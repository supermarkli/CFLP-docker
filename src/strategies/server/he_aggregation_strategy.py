import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import grpc
import zlib
import time
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.parameter_utils import serialize_parameters
from src.utils.logging_config import get_logger

logger = get_logger()

# 聚合时的并行线程数
AGGREGATION_WORKERS = 4

# 压缩配置 (需要与客户端保持一致)
ENABLE_COMPRESSION = True


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
        
        # 计算 CKKS 最大 slots 并校正 chunk_size
        max_slots = poly_mod_degree // 2
        configured_chunk_size = config['encryption']['chunk_size']
        
        if configured_chunk_size > max_slots:
            logger.warning(
                f"⚠️  配置错误: chunk_size ({configured_chunk_size}) 超过 CKKS 最大容量!\n"
                f"   CKKS slots 限制: poly_modulus_degree / 2 = {poly_mod_degree} / 2 = {max_slots}\n"
                f"   自动校正 chunk_size: {configured_chunk_size} → {max_slots}\n"
                f"   请修改 default.yaml 中的 chunk_size 配置以消除此警告。"
            )
            self.n_slots = max_slots
        else:
            self.n_slots = configured_chunk_size
        
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
        
        logger.info(f"CKKS上下文生成完毕: poly_mod={poly_mod_degree}, "
                   f"slots={self.n_slots}/{max_slots}, 压缩={'启用' if ENABLE_COMPRESSION else '禁用'}")

    def _decompress(self, data: bytes) -> bytes:
        """解压数据 (自动检测是否压缩)"""
        if not ENABLE_COMPRESSION:
            return data
        try:
            # zlib 压缩的数据以特定的魔数开头
            return zlib.decompress(data)
        except zlib.error:
            # 如果解压失败，假设数据未压缩
            return data
    
    def _decompress_and_deserialize(self, compressed_bytes: bytes):
        """解压并反序列化 CKKS 向量"""
        decompressed = self._decompress(compressed_bytes)
        return self.ts.ckks_vector_from(self.context, decompressed)

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
        start_time = time.time()
        total_bytes_received = 0
        layers_received = 0
        
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
                    
                    # 处理第一个块中的指标 (解压并反序列化)
                    metrics_bytes = chunk.metrics.test_acc
                    if metrics_bytes:
                        total_bytes_received += len(metrics_bytes)
                        encrypted_metrics_vector = self._decompress_and_deserialize(metrics_bytes)
                        with self.server.lock:
                            self.server.clients[client_id].encrypted_metrics = encrypted_metrics_vector
                    continue

                # --- 累积参数块数据 ---
                layer_name = chunk.layer_name
                for key, enc_array in chunk.parameters_chunk.items():
                    for compressed_vec in enc_array.data:
                        total_bytes_received += len(compressed_vec)
                        # 解压并反序列化
                        ckks_vector = self._decompress_and_deserialize(compressed_vec)
                        layer_cache[layer_name].append(ckks_vector)
                    
                    if enc_array.shape:
                        layer_shapes[layer_name] = list(enc_array.shape)
                
                # --- 如果当前层的所有块都已接收完毕 ---
                if chunk.is_last_chunk_for_layer:
                    layers_received += 1
                    num_vectors = len(layer_cache[layer_name])
                    logger.debug(f"[Round {round_num+1}] 客户端 {client_id}: 层 {layer_name} "
                                f"接收完毕 ({num_vectors} 个密文)")
                    
                    # 存储该层的所有 CKKS 向量和形状
                    shape = layer_shapes.get(layer_name, [1])
                    reconstructed_layer = {
                        layer_name: {
                            'vectors': layer_cache[layer_name],
                            'shape': shape
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

            # --- 流处理结束 ---
            elapsed = time.time() - start_time
            with self.server.lock:
                logger.info(f"[Round {round_num+1}] 客户端 {client_id} 数据接收完成: "
                           f"{layers_received} 层, {total_bytes_received/1024:.1f}KB, {elapsed:.2f}s")
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
        """反序列化非流式的 CKKS 加密更新 (支持压缩数据)。"""
        params = {}
        
        for key, enc_array in payload.parameters_and_metrics.parameters.parameters.items():
            vectors = [
                self._decompress_and_deserialize(b) 
                for b in enc_array.data
            ]
            shape = list(enc_array.shape) if enc_array.shape else [1]
            params[key] = {
                'vectors': vectors,
                'shape': shape
            }
        
        # 反序列化指标 (解压并反序列化)
        metrics_bytes = payload.parameters_and_metrics.metrics.test_acc
        if metrics_bytes:
            metrics_vector = self._decompress_and_deserialize(metrics_bytes)
        else:
            metrics_vector = None
            
        return params, metrics_vector

    def aggregate_parameters(self, round_num):
        """
        在 CKKS 密文上聚合客户端参数，然后解密。
        
        CKKS 的优势: 密文加法直接对应明文加法，可以高效地进行 FedAvg。
        使用多线程并行聚合不同的参数层以提高性能。
        按 data_size 加权聚合以保证 FedAvg 一致性。
        """
        start_time = time.time()
        logger.info(f"[Round {round_num+1}] 开始CKKS密文聚合...")
        
        client_ids = list(self.server.client_parameters[round_num].keys())
        num_clients = len(client_ids)
        
        if num_clients == 0:
            return self.server.global_model.get_parameters()
        
        # 计算各客户端的权重 (按 data_size 加权)
        total_data_size = sum(self.server.clients[cid].data_size for cid in client_ids)
        if total_data_size == 0:
            logger.warning(f"[Round {round_num+1}] 总数据量为0，使用简单平均")
            client_weights = [1.0 / num_clients for _ in client_ids]
        else:
            client_weights = [self.server.clients[cid].data_size / total_data_size for cid in client_ids]
        
        logger.info(f"[Round {round_num+1}] 客户端权重: {dict(zip(client_ids, [f'{w:.4f}' for w in client_weights]))}")
        
        # 获取第一个客户端的参数结构
        first_client_params = self.server.client_parameters[round_num][client_ids[0]]
        layer_keys = list(first_client_params.keys())
        
        logger.info(f"[Round {round_num+1}] 聚合 {len(layer_keys)} 个参数层, "
                   f"来自 {num_clients} 个客户端")
        
        def aggregate_single_layer(key):
            """聚合单个参数层 (可并行执行), 按 data_size 加权"""
            # 获取该层的所有客户端的 CKKS 向量列表
            all_client_vectors = [
                self.server.client_parameters[round_num][cid][key]['vectors']
                for cid in client_ids
            ]
            shape = first_client_params[key]['shape']
            num_vectors = len(all_client_vectors[0])
            
            # 在密文空间加权聚合 (CKKS 支持密文加法和标量乘法)
            aggregated_vectors = []
            for vec_idx in range(num_vectors):
                # 从第一个客户端开始，乘以其权重
                weighted_vector = all_client_vectors[0][vec_idx] * client_weights[0]
                
                # 加上其他客户端的加权向量
                for client_idx in range(1, num_clients):
                    weighted_vector = weighted_vector + all_client_vectors[client_idx][vec_idx] * client_weights[client_idx]
                
                # 加权求和后不需要再除，因为权重之和为1
                aggregated_vectors.append(weighted_vector)
            
            # 解密并重构数组
            decrypted_flat = []
            for vec in aggregated_vectors:
                decrypted_flat.extend(vec.decrypt())
            
            # 截断到原始大小并重塑形状
            total_elements = int(np.prod(shape))
            decrypted_array = np.array(decrypted_flat[:total_elements]).reshape(shape)
            
            return key, decrypted_array
        
        aggregated_params = {}
        
        # 使用线程池并行聚合不同层
        # 注意：TenSEAL 的某些操作可能有 GIL 限制，但解密操作通常是 CPU 密集型的
        with ThreadPoolExecutor(max_workers=AGGREGATION_WORKERS) as executor:
            futures = {executor.submit(aggregate_single_layer, key): key for key in layer_keys}
            
            completed = 0
            for future in as_completed(futures):
                key, decrypted_array = future.result()
                aggregated_params[key] = decrypted_array
                completed += 1
                
                # 每完成 10 层或全部完成时输出进度
                if completed % 10 == 0 or completed == len(layer_keys):
                    logger.debug(f"[Round {round_num+1}] 聚合进度: {completed}/{len(layer_keys)} 层")

        elapsed = time.time() - start_time
        logger.info(f"[Round {round_num+1}] CKKS密文聚合完成: {len(layer_keys)} 层, 耗时: {elapsed:.2f}s")
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
