import threading
import pickle
import numpy as np
from phe import paillier
from collections import defaultdict
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
        logger.info(f"Paillier密钥对生成完毕 (密钥长度: {key_size} bits)。")

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
        
        # 将浮点数权重转换为定点整数以进行同态乘法
        scaling_factor = 1e6  # 缩放因子，用于保留精度
        client_weights_int = [int(w * scaling_factor) for c in active_clients for w in [c.data_size / total_data_size]]
        total_weight_int = sum(client_weights_int)

        aggregated_params = {}
        param_structure = parameters_list[0]
        for key in param_structure.keys():
            # 使用整数权重进行加权求和
            weighted_sum = sum(p[key] * w for p, w in zip(parameters_list, client_weights_int))
            
            logger.debug(f"开始解密参数 {key}...")
            flat_sum = weighted_sum.flatten()
            decrypted_flat = np.array([self.private_key.decrypt(val) for val in flat_sum])
            
            # 解密和反缩放
            if total_weight_int > 0:
                scaled_avg = decrypted_flat / total_weight_int
            else:
                scaled_avg = decrypted_flat # Should not happen if total_data_size > 0
            
            aggregated_params[key] = scaled_avg.reshape(param_structure[key].shape)
        
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

        decrypted_metrics = {k: self.private_key.decrypt(v) for k, v in agg_metrics.items()}
        
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