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
        self.n = self.he_public_key.n
        logger.info(f"客户端 {self.client.client_id} 的 HE 策略已初始化并加载了公钥。")

    def _encode(self, value, scaling_factor):
        """将浮点数编码为整数，保持符号。确保绝对值远小于 n/2。"""
        return int(round(value * scaling_factor))

    def _encrypt_chunk(self, chunk, scaling_factor):
        """加密一个数据块。"""
        # 注意：paillier库本身会处理编码，但为了统一负数处理，我们手动编码
        return [self.he_public_key.encrypt(self._encode(v, scaling_factor)) for v in chunk]

    def _serialize_encrypted_chunk(self, encrypted_chunk, ciphertext_bytes_len):
        """将加密块序列化为字节列表。"""
        return [num.ciphertext().to_bytes(ciphertext_bytes_len, 'big') for num in encrypted_chunk]
    
    def _encrypt_layer_generator(self, layer_flat_values, chunk_size, scaling_factor, ciphertext_bytes_len):
        """逐块加密扁平化层，保持向后兼容给非流式接口使用。"""
        total = len(layer_flat_values)
        for i in range(0, total, chunk_size):
            chunk = layer_flat_values[i:i + chunk_size]
            encrypted_chunk = self._encrypt_chunk(chunk, scaling_factor)
            serialized_chunk = self._serialize_encrypted_chunk(encrypted_chunk, ciphertext_bytes_len)
            for ct in serialized_chunk:
                yield ct
            progress = min(i + chunk_size, total)
            logger.info(f"加密进度: {progress}/{total} ({(progress/total)*100:.1f}%)")
            del chunk, encrypted_chunk, serialized_chunk
            gc.collect()

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（HE密文），供非流式回退使用。"""
        proto_params = {}
        chunk_size = config['encryption']['chunk_size']
        ciphertext_bytes_len = (self.he_public_key.nsquare.bit_length() + 7) // 8
        scaling_factor = config['encryption']['scaling_factor']

        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten()
                logger.info(f"HE策略: 开始加密参数 {key}, 形状: {value.shape}, 总量: {len(flat)}")

                encryption_generator = self._encrypt_layer_generator(
                    flat, chunk_size, scaling_factor, ciphertext_bytes_len
                )
                encrypted_array_proto = federation_pb2.EncryptedNumpyArray(
                    shape=list(value.shape)
                )
                encrypted_array_proto.data.extend(encryption_generator)
                proto_params[key] = encrypted_array_proto
                del flat, encryption_generator, encrypted_array_proto
                gc.collect()
            else:
                encrypted_value = self.he_public_key.encrypt(self._encode(value, scaling_factor))
                proto_params[key] = federation_pb2.EncryptedNumpyArray(
                    data=[encrypted_value.ciphertext().to_bytes(ciphertext_bytes_len, 'big')],
                    shape=[1]
                )

        encrypted_model_params = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        # --- 加密训练指标 ---
        metrics_to_encrypt = metrics.copy()
        if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
            metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)
        encrypted_metrics_dict = {
            k: self.he_public_key.encrypt(self._encode(v, scaling_factor)).ciphertext().to_bytes(ciphertext_bytes_len, 'big')
            for k, v in metrics_to_encrypt.items()
        }
        encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(**encrypted_metrics_dict)

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
    
    def prepare_stream_update_request(self, current_round, model_parameters, metrics):
        """返回一个生成器，用于流式传输加密的模型更新。"""
        
        def update_generator():
            logger.info(f"客户端{self.client.client_id} 开始准备HE流式更新...")
            
            # --- 1. 准备和加密指标 ---
            ciphertext_bytes_len = (self.he_public_key.nsquare.bit_length() + 7) // 8
            scaling_factor = config['encryption']['scaling_factor']
            
            metrics_to_encrypt = metrics.copy()
            if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
                metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)

            # 使用新的编码函数加密指标
            encrypted_metrics_dict = {
                k: self.he_public_key.encrypt(self._encode(v, 1)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') 
                for k, v in metrics_to_encrypt.items()
            }
            encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(**encrypted_metrics_dict)
            
            # --- 2. 发送第一个包含元数据和指标的块 ---
            initial_chunk = federation_pb2.HeClientUpdateChunk(
                client_id=self.client.client_id,
                round=current_round,
                metrics=encrypted_metrics_proto,
                parameters_chunk={},
                is_last_chunk_for_layer=True, # 标记元数据块
                layer_name="metadata"
            )
            logger.info(f"客户端{self.client.client_id} 将发送包含元数据和指标的第一个块。")
            yield initial_chunk

            # --- 3. 逐层、逐块加密并流式传输模型参数 ---
            chunk_size = config['encryption']['chunk_size']
            
            for key, value in model_parameters.items():
                if isinstance(value, np.ndarray):
                    flat = value.flatten()
                    total_elements = len(flat)
                    logger.info(f"客户端{self.client.client_id} HE策略: 开始流式加密参数层 {key}, 形状: {value.shape}, 总元素: {total_elements}")

                    for i in range(0, total_elements, chunk_size):
                        chunk_data = flat[i:i + chunk_size]
                        
                        encrypted_chunk = self._encrypt_chunk(chunk_data, scaling_factor)
                        serialized_chunk = self._serialize_encrypted_chunk(encrypted_chunk, ciphertext_bytes_len)
                        
                        is_first_chunk = (i == 0)
                        is_last_chunk = (i + chunk_size >= total_elements)

                        # 仅在每层的第一个块中发送形状信息
                        shape_info = list(value.shape) if is_first_chunk else []
                        
                        enc_array_proto = federation_pb2.EncryptedNumpyArray(
                            shape=shape_info
                        )
                        enc_array_proto.data.extend(serialized_chunk)

                        param_chunk = federation_pb2.HeClientUpdateChunk(
                            parameters_chunk={key: enc_array_proto},
                            layer_name=key,
                            is_last_chunk_for_layer=is_last_chunk
                        )
                        
                        progress = min(i + chunk_size, total_elements)
                        logger.info(f"客户端{self.client.client_id} 正在发送层 {key} 的块: {progress}/{total_elements} ({(progress/total_elements)*100:.1f}%)")
                        yield param_chunk
                        
                    logger.info(f"客户端{self.client.client_id} 参数层 {key} 已全部发送。")
                    del flat, chunk_data, encrypted_chunk, serialized_chunk
                    gc.collect()

                else: # 处理标量
                    encrypted_value = self.he_public_key.encrypt(self._encode(value, scaling_factor))
                    scalar_proto = federation_pb2.EncryptedNumpyArray(
                        data=[encrypted_value.ciphertext().to_bytes(ciphertext_bytes_len, 'big')], 
                        shape=[1]
                    )
                    param_chunk = federation_pb2.HeClientUpdateChunk(
                        parameters_chunk={key: scalar_proto},
                        layer_name=key,
                        is_last_chunk_for_layer=True
                    )
                    yield param_chunk
            
            logger.info(f"客户端{self.client.client_id} 所有参数块已发送完毕。")

        return update_generator 