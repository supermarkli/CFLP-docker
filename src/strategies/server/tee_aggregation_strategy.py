import threading
import json
import hashlib
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.utils.logging_config import get_logger

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = get_logger()


class TeeAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)
        logger.info("启动TEE模式：正在生成RSA密钥对用于模拟Enclave...")
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        
        logger.info("启动TEE模式：正在模拟生成身份指纹(MRENCLAVE)...")
        model_str = str(self.server.global_model.state_dict())
        self.mrenclave = hashlib.sha256(model_str.encode()).hexdigest()

    def prepare_setup_response(self, request):
        logger.info(f"向客户端 {request.client_id} 提供模拟的证明报告和公钥。")
        response = federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )
        report_data = {"mrenclave": self.mrenclave}
        response.tee_attestation_report = json.dumps(report_data).encode('utf-8')
        response.tee_public_key = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return response

    def aggregate(self, request, context):
        payload = request.tee
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'tee' 模式不匹配。")

        try:
            # 1. 解密载荷
            symmetric_key = self.private_key.decrypt(
                payload.encrypted_symmetric_key,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
            aesgcm = AESGCM(symmetric_key)
            decrypted_payload_bytes = aesgcm.decrypt(payload.nonce, payload.encrypted_payload, None)
            
            params_and_metrics = federation_pb2.ParametersAndMetrics()
            params_and_metrics.ParseFromString(decrypted_payload_bytes)

            # 2. 调用与 'none' 模式类似的明文处理逻辑
            with self.server.lock:
                round_num = request.round
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}")
                
                params, metrics_data = self._process_plaintext_update(params_and_metrics)
                self.server.clients[request.client_id].metrics = metrics_data
                self.server.client_parameters[round_num][request.client_id] = params
                
                logger.info(f"[Round {round_num+1}] 收到并解密了客户端 {request.client_id} 的TEE模式更新。")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="更新已收到"
                )

        except Exception as e:
            logger.error(f"处理TEE更新失败: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message="解密或处理TEE载荷时发生错误")

    def _process_plaintext_update(self, params_and_metrics):
        """复用与 'none' 模式相同的明文处理逻辑。"""
        parameters = deserialize_parameters(params_and_metrics.parameters.parameters)
        metrics = params_and_metrics.metrics
        metrics_data = {
            'test_acc': metrics.test_acc, 'test_num': metrics.test_num, 'auc': metrics.auc,
            'loss': metrics.loss, 'train_num': metrics.train_num
        }
        return parameters, metrics_data 

    def aggregate_parameters(self, round_num):
        """聚合明文客户端参数 (FedAvg)"""
        active_clients = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]
        parameters_list = list(self.server.client_parameters[round_num].values())
        
        total_data_size = sum(client.data_size for client in active_clients)
        if total_data_size == 0: return self.server.global_model.get_parameters()
        
        client_weights = [client.data_size / total_data_size for client in active_clients]
        
        aggregated = {}
        param_structure = parameters_list[0]
        for param_name in param_structure.keys():
            aggregated[param_name] = sum(weight * params[param_name] for params, weight in zip(parameters_list, client_weights))
        
        logger.info(f"[Round {round_num+1}] TEE模式下明文参数聚合完成。")
        return aggregated

    def evaluate_metrics(self, round_num):
        """评估明文指标"""
        total_test_acc, total_test_num = 0, 0
        total_auc, total_loss, total_train_num = 0, 0, 0
        
        clients_in_round = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]

        for c in clients_in_round:
            m = c.metrics
            if m:
                total_test_acc += m['test_acc']
                total_test_num += m['test_num']
                total_auc += m['auc'] * m['test_num'] # AUC需要加权
                total_loss += m['loss']
                total_train_num += m['train_num']
        
        # 清理本轮存储的指标
        for c in clients_in_round: c.metrics = None
        
        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0
        
        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估 (TEE): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")

        # 清理本轮的参数
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num] 