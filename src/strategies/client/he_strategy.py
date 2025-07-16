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

    def _encrypt_layer_generator(self, layer_flat_values, chunk_size, scaling_factor, ciphertext_bytes_len):
        """一个生成器，逐块加密一个扁平化的层。"""
        total = len(layer_flat_values)
        for i in range(0, total, chunk_size):
            chunk = layer_flat_values[i:i + chunk_size]
            encrypted_part = [
                self.he_public_key.encrypt(int(v * scaling_factor)).ciphertext().to_bytes(ciphertext_bytes_len, 'big')
                for v in chunk
            ]
            for item in encrypted_part:
                yield item
            
            progress = min(i + chunk_size, total)
            logger.info(f"加密进度: {progress}/{total} ({(progress/total)*100:.1f}%)")
            del chunk, encrypted_part
            gc.collect()

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（HE密文）"""
        proto_params = {}
        chunk_size = config['encryption']['chunk_size']
        ciphertext_bytes_len = (self.he_public_key.nsquare.bit_length() + 7) // 8
        scaling_factor = 1e6 # 与服务器端一致

        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten()
                total = len(flat)
                logger.info(f"HE策略: 开始流式加密参数 {key}, 形状: {value.shape}, 总量: {total}")

                encryption_generator = self._encrypt_layer_generator(
                    flat, chunk_size, scaling_factor, ciphertext_bytes_len
                )
                
                # 直接从生成器创建Protobuf消息，避免在内存中创建完整的列表
                encrypted_array_proto = federation_pb2.EncryptedNumpyArray(
                    data=encryption_generator,
                    shape=list(value.shape)
                )
                proto_params[key] = encrypted_array_proto
                
                logger.info(f"参数 {key} 已完成流式加密。")
                del flat, encryption_generator, encrypted_array_proto
                gc.collect()

            else: # scalar
                encrypted_value = self.he_public_key.encrypt(int(value * scaling_factor))
                proto_params[key] = federation_pb2.EncryptedNumpyArray(
                    data=[encrypted_value.ciphertext().to_bytes(ciphertext_bytes_len, 'big')], 
                    shape=[1]
                )
        
        encrypted_model_params = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        # --- 加密训练指标 ---
        # 预处理指标，将auc乘以样本数
        metrics_to_encrypt = metrics.copy()
        if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
            metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)

        encrypted_metrics_dict = {k: self.he_public_key.encrypt(float(v)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') for k, v in metrics_to_encrypt.items()}
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