import threading
import numpy as np
from collections import defaultdict
import grpc
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.parameter_utils import serialize_parameters
from src.utils.logging_config import get_logger

logger = get_logger()


class HeAggregationStrategy(AggregationStrategy):
    """
    基于 TenSEAL CKKS 的同态加密聚合策略。
    
    CKKS 方案的核心优势:
    1. SIMD 批处理: 密文上的加法/乘法自动应用于所有打包的元素
    2. 密文加法: 可以直接在密文空间进行 FedAvg 聚合
    3. 性能: 聚合操作比逐元素 Paillier 快数十倍
    """
    
    def __init__(self, server_instance):
        super().__init__(server_instance)
        import tenseal as ts
        self.ts = ts
        
        logger.info("启动HE模式：正在生成CKKS上下文...")
        
        # 从配置读取 CKKS 参数
        poly_mod_degree = config['encryption']['poly_modulus_degree']
        coeff_mod_bit_sizes = config['encryption']['coeff_mod_bit_sizes']
        global_scale = config['encryption']['global_scale']
        self.n_slots = config['encryption']['chunk_size']
        
        # 创建 CKKS 上下文 (包含公钥和私钥)
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = global_scale
        self.context.generate_galois_keys()  # 用于向量旋转操作
        
        # 创建只包含公钥的上下文 (发送给客户端)
        self.public_context = self.context.copy()
        self.public_context.make_context_public()  # 移除私钥
        
        logger.info(f"CKKS上下文生成完毕 (poly_modulus_degree={poly_mod_degree}, "
                   f"slots={self.n_slots})。")

    def prepare_setup_response(self, request):
        """向客户端发送公钥上下文。"""
        logger.info(f"向客户端 {request.client_id} 提供CKKS公钥上下文。")
        
        response = federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )
        # 序列化公钥上下文
        response.he_public_key = self.public_context.serialize()
        return response

    def aggregate(self, request, context):
        """处理非流式的客户端更新。"""
        payload = request.he
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'he' 模式不匹配。")
        
        try:
            client_id = request.client_id
            round_num = request.round
            
            with self.server.lock:
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(
                        code=400, 
                        message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}"
                    )

                params, metrics_data = self._process_encrypted_update(payload)
                self.server.clients[client_id].encrypted_metrics = metrics_data
                self.server.client_parameters[round_num][client_id] = params
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的CKKS密文更新。")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(
                        target=self.server.process_round_completion, 
                        args=(round_num,)
                    ).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="更新已收到"
                )

        except Exception as e:
            logger.error(f"处理CKKS密文更新时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def aggregate_stream(self, request_iterator, context):
        """处理来自客户端的加密参数流。"""
        client_id, round_num = None, None
        
        # 用于缓存每一层的所有 CKKS 密文块
        layer_cache = defaultdict(list)
        layer_shapes = {}

        try:
            for chunk in request_iterator:
                # --- 从第一个块中提取元数据 ---
                if chunk.layer_name == "metadata":
                    client_id = chunk.client_id
                    round_num = chunk.round
                    
                    with self.server.lock:
                        if round_num != self.server.current_round:
                            msg = f"轮次不匹配，服务器当前轮次为 {self.server.current_round}"
                            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                            context.set_details(msg)
                            return federation_pb2.ServerUpdate(code=400, message=msg)
                    
                    logger.info(f"[Round {round_num+1}] 开始接收来自客户端 {client_id} 的CKKS流式更新...")
                    
                    # 处理第一个块中的指标 (所有指标打包在一个 CKKS 向量中)
                    metrics_bytes = chunk.metrics.test_acc  # 整个向量存储在 test_acc 字段
                    if metrics_bytes:
                        encrypted_metrics_vector = self.ts.ckks_vector_from(
                            self.context, metrics_bytes
                        )
                        with self.server.lock:
                            self.server.clients[client_id].encrypted_metrics = encrypted_metrics_vector
                    continue

                # --- 累积参数块数据 ---
                layer_name = chunk.layer_name
                for key, enc_array in chunk.parameters_chunk.items():
                    # 每个 data 元素是一个序列化的 CKKS 向量
                    for serialized_vec in enc_array.data:
                        ckks_vector = self.ts.ckks_vector_from(self.context, serialized_vec)
                        layer_cache[layer_name].append(ckks_vector)
                    
                    if enc_array.shape:
                        layer_shapes[layer_name] = list(enc_array.shape)
                
                # --- 如果当前层的所有块都已接收完毕 ---
                if chunk.is_last_chunk_for_layer:
                    logger.info(f"[Round {round_num+1}] 客户端 {client_id} 的层 {layer_name} "
                               f"数据接收完毕 ({len(layer_cache[layer_name])} 个CKKS密文)。")
                    
                    # 存储该层的所有 CKKS 向量和形状
                    reconstructed_layer = {
                        layer_name: {
                            'vectors': layer_cache[layer_name],
                            'shape': layer_shapes[layer_name]
                        }
                    }

                    with self.server.lock:
                        if client_id not in self.server.client_parameters[round_num]:
                            self.server.client_parameters[round_num][client_id] = {}
                        self.server.client_parameters[round_num][client_id].update(reconstructed_layer)

                    # 清理已处理完的层的缓存
                    del layer_cache[layer_name]
                    if layer_name in layer_shapes:
                        del layer_shapes[layer_name]

            # --- 流处理结束，检查是否所有客户端都已提交 ---
            with self.server.lock:
                logger.info(f"[Round {round_num+1}] 已成功处理客户端 {client_id} 的所有CKKS流式数据。")
                self.server.completed_clients[round_num].add(client_id)
                
                completed_clients_count = len(self.server.completed_clients[round_num])
                if completed_clients_count >= self.server.expected_clients:
                    logger.info(f"[Round {round_num+1}] 所有客户端更新完毕，触发聚合。")
                    threading.Thread(
                        target=self.server.process_round_completion, 
                        args=(round_num,)
                    ).start()

            return federation_pb2.ServerUpdate(
                code=200, 
                current_round=self.server.current_round, 
                message="CKKS流式更新已成功接收"
            )

        except Exception as e:
            logger.error(f"处理CKKS流式密文更新时出错: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"服务器内部错误: {e}")
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def _process_encrypted_update(self, payload):
        """反序列化非流式的 CKKS 加密更新。"""
        params = {}
        
        for key, enc_array in payload.parameters_and_metrics.parameters.parameters.items():
            vectors = [
                self.ts.ckks_vector_from(self.context, b) 
                for b in enc_array.data
            ]
            params[key] = {
                'vectors': vectors,
                'shape': list(enc_array.shape)
            }
        
        # 反序列化指标 (所有指标打包在 test_acc 字段中)
        metrics_bytes = payload.parameters_and_metrics.metrics.test_acc
        if metrics_bytes:
            metrics_vector = self.ts.ckks_vector_from(self.context, metrics_bytes)
        else:
            metrics_vector = None
            
        return params, metrics_vector

    def aggregate_parameters(self, round_num):
        """
        在 CKKS 密文上聚合客户端参数，然后解密。
        
        CKKS 的优势: 密文加法直接对应明文加法，可以高效地进行 FedAvg。
        """
        logger.info(f"[Round {round_num+1}] 开始CKKS密文聚合...")
        
        client_ids = list(self.server.client_parameters[round_num].keys())
        num_clients = len(client_ids)
        
        if num_clients == 0:
            return self.server.global_model.get_parameters()
        
        # 获取第一个客户端的参数结构
        first_client_params = self.server.client_parameters[round_num][client_ids[0]]
        
        aggregated_params = {}
        
        for key in first_client_params.keys():
            logger.info(f"聚合参数层: {key}")
            
            # 获取该层的所有客户端的 CKKS 向量列表
            all_client_vectors = [
                self.server.client_parameters[round_num][cid][key]['vectors']
                for cid in client_ids
            ]
            shape = first_client_params[key]['shape']
            num_vectors = len(all_client_vectors[0])  # 每个客户端的向量块数
            
            # 在密文空间聚合 (CKKS 支持密文加法)
            aggregated_vectors = []
            for vec_idx in range(num_vectors):
                # 从第一个客户端开始
                summed_vector = all_client_vectors[0][vec_idx]
                
                # 加上其他客户端的对应向量
                for client_idx in range(1, num_clients):
                    summed_vector = summed_vector + all_client_vectors[client_idx][vec_idx]
                
                # 密文上除以客户端数量 (FedAvg 简单平均)
                avg_vector = summed_vector * (1.0 / num_clients)
                aggregated_vectors.append(avg_vector)
            
            # 解密并重构数组
            decrypted_flat = []
            for vec in aggregated_vectors:
                decrypted_flat.extend(vec.decrypt())
            
            # 截断到原始大小并重塑形状
            total_elements = np.prod(shape)
            decrypted_array = np.array(decrypted_flat[:total_elements]).reshape(shape)
            aggregated_params[key] = decrypted_array
            
            logger.debug(f"层 {key} 聚合完成, 形状: {shape}")

        logger.info(f"[Round {round_num+1}] CKKS密文参数聚合与解密完成。")
        return aggregated_params
        
    def evaluate_metrics(self, round_num):
        """解密并评估加密的指标。"""
        client_ids = list(self.server.client_parameters[round_num].keys())
        
        if not client_ids:
            return
        
        # 收集所有客户端的加密指标向量
        encrypted_metrics_list = []
        for cid in client_ids:
            client = self.server.clients[cid]
            if client.encrypted_metrics is not None:
                encrypted_metrics_list.append(client.encrypted_metrics)
        
        if not encrypted_metrics_list:
            return
        
        # 在密文空间求和
        summed_metrics = encrypted_metrics_list[0]
        for i in range(1, len(encrypted_metrics_list)):
            summed_metrics = summed_metrics + encrypted_metrics_list[i]
        
        # 解密
        decrypted_values = summed_metrics.decrypt()
        
        # 指标顺序: [test_acc, test_num, auc, loss, train_num]
        total_test_acc = decrypted_values[0]
        total_test_num = decrypted_values[1]
        total_auc = decrypted_values[2]  # 已经是加权和
        total_loss = decrypted_values[3]
        total_train_num = decrypted_values[4]

        # 清理本轮存储的加密指标
        for cid in client_ids:
            self.server.clients[cid].encrypted_metrics = None
        
        # 清理本轮的参数
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num]

        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0

        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        
        logger.info(f"[Round {round_num+1}] 全局评估 (CKKS): "
                   f"Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")
