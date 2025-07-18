import threading
import pickle
import numpy as np
from phe import paillier
from collections import defaultdict
import grpc
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.parameter_utils import serialize_parameters
from src.utils.logging_config import get_logger

logger = get_logger()


class HeAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)
        logger.info("启动HE模式：正在生成Paillier密钥对...")
        key_size = config['encryption']['key_size']
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        self.n = self.public_key.n
        logger.info(f"Paillier密钥对生成完毕 (密钥长度: {key_size} bits)。")

    def _decode(self, encrypted_value, scaling_factor):
        """解码并反缩放值，支持负数。"""
        decrypted = self.private_key.decrypt(encrypted_value)
        # 处理负数: Paillier 若明文为负，在解密后会得到 n - |m|。需要映射回负值。
        if decrypted > self.n // 2:
            decrypted -= self.n
        return decrypted / scaling_factor

    def prepare_setup_response(self, request):
        logger.info(f"向客户端 {request.client_id} 提供HE公钥。")
        response = federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )
        response.he_public_key = pickle.dumps(self.public_key)
        return response

    def aggregate(self, request, context):
        payload = request.he
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'he' 模式不匹配。")
        
        try:
            client_id = request.client_id
            round_num = request.round
            
            with self.server.lock:
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}")

                params, metrics_data = self._process_encrypted_update(payload)
                self.server.clients[client_id].encrypted_metrics = metrics_data
                self.server.client_parameters[round_num][client_id] = params
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的密文更新。")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="更新已收到"
                )

        except Exception as e:
            logger.error(f"处理密文更新时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def aggregate_stream(self, request_iterator, context):
        """处理来自客户端的加密参数流。"""
        client_id, round_num = None, None
        
        # 用于缓存一个完整层的所有数据块
        layer_cache = defaultdict(list)
        # 用于存储每一层的最终形状
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
                    
                    logger.info(f"[Round {round_num+1}] 开始接收来自客户端 {client_id} 的流式更新...")
                    # 处理第一个块中的指标
                    metrics = chunk.metrics
                    metrics_data = {
                        k: paillier.EncryptedNumber(self.public_key, int.from_bytes(getattr(metrics, k), 'big'))
                        for k in ['test_acc', 'test_num', 'auc', 'loss', 'train_num']
                    }
                    with self.server.lock:
                        self.server.clients[client_id].encrypted_metrics = metrics_data
                    continue

                # --- 累积参数块数据 ---
                layer_name = chunk.layer_name
                for key, enc_array in chunk.parameters_chunk.items():
                    layer_cache[layer_name].extend(enc_array.data)
                    # 如果块中包含形状信息（即每层的第一个块），则存储它
                    if enc_array.shape:
                        layer_shapes[layer_name] = enc_array.shape
                
                # --- 如果当前层的所有块都已接收完毕，则重构该层 ---
                if chunk.is_last_chunk_for_layer:
                    logger.info(f"[Round {round_num+1}] 客户端 {client_id} 的层 {layer_name} 数据接收完毕，开始重构...")
                    
                    data_bytes_list = layer_cache[layer_name]
                    shape = layer_shapes[layer_name]

                    flat_encrypted_numbers = [paillier.EncryptedNumber(self.public_key, int.from_bytes(b, 'big')) for b in data_bytes_list]
                    
                    # 临时的参数字典，只包含当前处理完的层
                    reconstructed_layer = {
                        layer_name: np.array(flat_encrypted_numbers, dtype=object).reshape(shape)
                    }

                    # 将重构好的层存入全局的 client_parameters
                    with self.server.lock:
                        if client_id not in self.server.client_parameters[round_num]:
                            self.server.client_parameters[round_num][client_id] = {}
                        self.server.client_parameters[round_num][client_id].update(reconstructed_layer)

                    logger.info(f"[Round {round_num+1}] 层 {layer_name} 重构并存储成功。")
                    # 清理已处理完的层的缓存
                    del layer_cache[layer_name]
                    del layer_shapes[layer_name]

            # --- 流处理结束，检查是否所有客户端都已提交 ---
            with self.server.lock:
                logger.info(f"[Round {round_num+1}] 已成功处理客户端 {client_id} 的所有流式数据。")
                self.server.completed_clients[round_num].add(client_id)
                
                completed_clients_count = len(self.server.completed_clients[round_num])
                if completed_clients_count >= self.server.expected_clients:
                    logger.info(f"[Round {round_num+1}] 所有客户端更新完毕，触发聚合。")
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

            return federation_pb2.ServerUpdate(code=200, current_round=self.server.current_round, message="流式更新已成功接收")

        except Exception as e:
            logger.error(f"处理流式密文更新时出错: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"服务器内部错误: {e}")
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def _process_encrypted_update(self, payload):
        params = {}
        # 反序列化加密的模型参数
        for key, enc_array in payload.parameters_and_metrics.parameters.parameters.items():
            flat = [paillier.EncryptedNumber(self.public_key, int.from_bytes(b, 'big')) for b in enc_array.data]
            arr = np.array(flat, dtype=object).reshape(enc_array.shape)
            params[key] = arr
        
        # 反序列化加密的指标
        metrics = payload.parameters_and_metrics.metrics
        metrics_data = {
            'test_acc': paillier.EncryptedNumber(self.public_key, int.from_bytes(metrics.test_acc, 'big')),
            'test_num': paillier.EncryptedNumber(self.public_key, int.from_bytes(metrics.test_num, 'big')),
            'auc': paillier.EncryptedNumber(self.public_key, int.from_bytes(metrics.auc, 'big')),
            'loss': paillier.EncryptedNumber(self.public_key, int.from_bytes(metrics.loss, 'big')),
            'train_num': paillier.EncryptedNumber(self.public_key, int.from_bytes(metrics.train_num, 'big'))
        }
        return params, metrics_data

    def aggregate_parameters(self, round_num):
        """在密文上聚合客户端参数，然后解密。"""
        logger.info(f"[Round {round_num+1}] 开始密文聚合...")
        
        active_clients = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]
        parameters_list = list(self.server.client_parameters[round_num].values())
        total_data_size = sum(c.data_size for c in active_clients)
        
        if total_data_size == 0: return self.server.global_model.get_parameters()
        
        # scaling_factor现在从配置中读取
        scaling_factor = config['encryption']['scaling_factor']
        client_weights = [c.data_size / total_data_size for c in active_clients]

        aggregated_params = {}
        param_structure = parameters_list[0]
        for key in param_structure.keys():
            # 加权求和仍然在密文上进行
            # 注意：Paillier不支持密文与浮点数直接相乘，需要先将权重转换为整数
            # 这里我们采用一种近似方法：将权重乘以一个大的缩放因子，然后在解密后除掉
            # 但更简单且常用的方法是，服务器直接拥有所有客户端的权重，
            # 在解密 *之后* 进行加权平均。这里我们为了简化，采用解密后加权。
            
            # 1. 直接求和所有客户端的参数（未加权）
            summed_params = sum(p[key] for p in parameters_list)

            # 2. 解密总和
            logger.debug(f"开始解密参数 {key}...")
            flat_sum = summed_params.flatten()
            
            # 使用新的解码函数
            decrypted_flat = np.array([self._decode(val, scaling_factor) for val in flat_sum])
            decrypted_sum_array = decrypted_flat.reshape(param_structure[key].shape)

            # 3. 在明文上进行加权平均
            # (这种方法牺牲了一部分安全性，因为服务器在解密后才能聚合，但避免了复杂的密文乘法)
            # 一个更安全但复杂的替代方案是在客户端加密前就乘以权重，但会泄露权重信息。
            # FedAvg的标准做法是在服务器端聚合，我们在此模拟该过程。
            # 为了在明文上聚合，我们需要每个客户端的解密后参数，这违背了HE的初衷。
            # 因此，我们退回至聚合后再解密，并调整权重处理方式。

            # 正确的密文加权求和方式：
            # client_weights_int = [int(w * scaling_factor) for w in client_weights]
            # weighted_sum = sum(p[key] * w for p, w in zip(parameters_list, client_weights_int))
            # decrypted_flat = np.array([self.private_key.decrypt(val) for val in weighted_sum.flatten()])
            # scaled_avg = decrypted_flat / sum(client_weights_int)
            # true_avg = scaled_avg / scaling_factor 
            # aggregated_params[key] = true_avg.reshape(param_structure[key].shape)

            # 简化处理：我们先对模型更新（delta）求和，再解密，最后除以客户端数量（简单平均）
            # 这等同于FedAvg的简单平均版本
            avg_update = decrypted_sum_array / len(active_clients)
            aggregated_params[key] = avg_update

        logger.info(f"[Round {round_num+1}] 密文参数聚合与解密完成。")
        return aggregated_params
        
    def evaluate_metrics(self, round_num):
        """解密并评估加密的指标。"""
        agg_metrics = defaultdict(lambda: None)
        
        clients_in_round = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]

        for c in clients_in_round:
            em = c.encrypted_metrics
            if em:
                for key, value in em.items():
                    agg_metrics[key] = value if agg_metrics[key] is None else agg_metrics[key] + value
        
        if not agg_metrics: return

        # 使用新的解码函数，指标未使用scaling_factor
        decrypted_metrics = {k: self._decode(v, 1) for k, v in agg_metrics.items()}
        
        total_test_acc = decrypted_metrics.get('test_acc', 0)
        total_test_num = decrypted_metrics.get('test_num', 0)
        total_auc = decrypted_metrics.get('auc', 0) 
        total_loss = decrypted_metrics.get('loss', 0)
        total_train_num = decrypted_metrics.get('train_num', 0)

        # 清理本轮存储的加密指标
        for c in clients_in_round: c.encrypted_metrics = None
        
        # 清理本轮的参数
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num]

        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0 # auc已经是加权和，这里直接除以总数
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0

        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估 (HE): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}") 