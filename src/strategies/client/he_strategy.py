import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
import numpy as np
import pickle
import gc

logger = get_logger()


class HeClientStrategy(ClientStrategy):
    def __init__(self, client_instance, he_public_key_bytes):
        super().__init__(client_instance)
        self.he_public_key = pickle.loads(he_public_key_bytes)
        logger.info(f"客户端 {self.client.client_id} 的 HE 策略已初始化并加载了公钥。")

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（HE密文）"""
        encrypted_parameters = {}
        chunk_size = config['encryption']['chunk_size']
        ciphertext_bytes_len = (self.he_public_key.nsquare.bit_length() + 7) // 8

        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten()
                encrypted_chunks = []
                total = len(flat)
                logger.info(f"HE策略: 开始加密参数 {key}, 形状: {value.shape}, 总量: {total}, 分块大小: {chunk_size}")

                for i in range(0, total, chunk_size):
                    chunk = flat[i:i+chunk_size]
                    encrypted_part = [self.he_public_key.encrypt(float(v)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') for v in chunk]
                    encrypted_chunks.extend(encrypted_part)
                    
                    progress = min(i + chunk_size, total)
                    logger.info(f"参数 {key} 加密进度: {progress}/{total} ({(progress/total)*100:.1f}%)")

                    del chunk, encrypted_part
                    gc.collect()
                
                encrypted_parameters[key] = {
                    'data': encrypted_chunks, 'shape': list(value.shape)
                }
            else: # scalar
                encrypted_value = self.he_public_key.encrypt(float(value))
                encrypted_parameters[key] = {
                    'data': [encrypted_value.ciphertext().to_bytes(ciphertext_bytes_len, 'big')], 'shape': [1]
                }
        
        proto_params = {k: federation_pb2.EncryptedNumpyArray(data=v['data'], shape=v['shape']) for k, v in encrypted_parameters.items()}
        encrypted_model_params = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        # --- 加密训练指标 ---
        encrypted_metrics_dict = {k: self.he_public_key.encrypt(float(v)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') for k, v in metrics.items()}
        encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(**encrypted_metrics_dict)
        
        # --- 组装 Payload ---
        he_payload = federation_pb2.HePayload(
            parameters_and_metrics=federation_pb2.EncryptedParametersAndMetrics(
                parameters=encrypted_model_params, 
                metrics=encrypted_metrics_proto
            )
        )

        return federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            he=he_payload
        ) 