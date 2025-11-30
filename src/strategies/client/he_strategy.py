import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
import numpy as np
import gc

logger = get_logger()


class HeClientStrategy(ClientStrategy):
    """
    基于 TenSEAL CKKS 的同态加密客户端策略。
    
    CKKS 方案相比 Paillier 的优势:
    1. SIMD 批处理: 一个密文可以打包数千个浮点数
    2. 原生浮点支持: 无需手动缩放
    3. C++ 后端: 性能远超纯 Python
    """
    
    def __init__(self, client_instance, context_bytes):
        super().__init__(client_instance)
        import tenseal as ts
        self.ts = ts
        
        # 从服务器接收的序列化上下文中恢复公钥上下文
        self.context = ts.context_from(context_bytes)
        
        # CKKS slots 数量 (每个密文可打包的元素数)
        self.n_slots = config['encryption']['chunk_size']
        
        logger.info(f"客户端 {self.client.client_id} 的 CKKS 策略已初始化, "
                   f"每个密文可打包 {self.n_slots} 个元素。")

    def _encrypt_vector(self, flat_array):
        """
        使用 CKKS 批量加密一个扁平化数组。
        
        利用 SIMD 特性，将数组分成多个 chunk，每个 chunk 打包到一个密文中。
        这比 Paillier 的逐元素加密快 10-100 倍。
        """
        encrypted_chunks = []
        total = len(flat_array)
        
        for i in range(0, total, self.n_slots):
            chunk = flat_array[i:i + self.n_slots].tolist()
            
            # 一次调用加密整个 chunk (SIMD)
            encrypted_vector = self.ts.ckks_vector(self.context, chunk)
            # 序列化为 bytes
            encrypted_chunks.append(encrypted_vector.serialize())
            
            progress = min(i + self.n_slots, total)
            if progress % (self.n_slots * 10) == 0 or progress == total:
                logger.info(f"CKKS 加密进度: {progress}/{total} ({(progress/total)*100:.1f}%)")
        
        return encrypted_chunks

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（CKKS密文），供非流式接口使用。"""
        proto_params = {}
        
        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten().astype(np.float64)
                logger.info(f"CKKS策略: 开始加密参数 {key}, 形状: {value.shape}, "
                           f"总量: {len(flat)}, 预计密文数: {(len(flat) + self.n_slots - 1) // self.n_slots}")
                
                encrypted_chunks = self._encrypt_vector(flat)
                
                encrypted_array_proto = federation_pb2.EncryptedNumpyArray(
                    shape=list(value.shape),
                    data=encrypted_chunks  # 每个 chunk 是一个 bytes
                )
                proto_params[key] = encrypted_array_proto
                
                del flat, encrypted_chunks
                gc.collect()
            else:
                # 标量处理
                encrypted_vector = self.ts.ckks_vector(self.context, [float(value)])
                proto_params[key] = federation_pb2.EncryptedNumpyArray(
                    data=[encrypted_vector.serialize()],
                    shape=[1]
                )

        encrypted_model_params = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        # --- 加密训练指标 ---
        metrics_to_encrypt = metrics.copy()
        if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
            metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)
        
        # 将所有指标打包到一个密文中 (充分利用 SIMD)
        metrics_values = [
            float(metrics_to_encrypt.get('test_acc', 0)),
            float(metrics_to_encrypt.get('test_num', 0)),
            float(metrics_to_encrypt.get('auc', 0)),
            float(metrics_to_encrypt.get('loss', 0)),
            float(metrics_to_encrypt.get('train_num', 0))
        ]
        encrypted_metrics_vector = self.ts.ckks_vector(self.context, metrics_values)
        encrypted_metrics_bytes = encrypted_metrics_vector.serialize()
        
        # 使用单个 bytes 字段存储所有指标
        encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(
            test_acc=encrypted_metrics_bytes,  # 实际存储的是整个向量
            test_num=b'',  # 空，因为所有值都在 test_acc 中
            auc=b'',
            loss=b'',
            train_num=b''
        )

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
            logger.info(f"客户端{self.client.client_id} 开始准备CKKS流式更新...")
            
            # --- 1. 准备和加密指标 ---
            metrics_to_encrypt = metrics.copy()
            if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
                metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)

            # 将所有指标打包到一个 CKKS 向量中
            metrics_values = [
                float(metrics_to_encrypt.get('test_acc', 0)),
                float(metrics_to_encrypt.get('test_num', 0)),
                float(metrics_to_encrypt.get('auc', 0)),
                float(metrics_to_encrypt.get('loss', 0)),
                float(metrics_to_encrypt.get('train_num', 0))
            ]
            encrypted_metrics_vector = self.ts.ckks_vector(self.context, metrics_values)
            encrypted_metrics_bytes = encrypted_metrics_vector.serialize()
            
            encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(
                test_acc=encrypted_metrics_bytes,
                test_num=b'', auc=b'', loss=b'', train_num=b''
            )
            
            # --- 2. 发送第一个包含元数据和指标的块 ---
            initial_chunk = federation_pb2.HeClientUpdateChunk(
                client_id=self.client.client_id,
                round=current_round,
                metrics=encrypted_metrics_proto,
                parameters_chunk={},
                is_last_chunk_for_layer=True,
                layer_name="metadata"
            )
            logger.info(f"客户端{self.client.client_id} 发送包含元数据和指标的第一个块。")
            yield initial_chunk

            # --- 3. 逐层加密并流式传输模型参数 ---
            for key, value in model_parameters.items():
                if isinstance(value, np.ndarray):
                    flat = value.flatten().astype(np.float64)
                    total_elements = len(flat)
                    num_chunks = (total_elements + self.n_slots - 1) // self.n_slots
                    
                    logger.info(f"客户端{self.client.client_id} CKKS策略: 开始流式加密参数层 {key}, "
                               f"形状: {value.shape}, 总元素: {total_elements}, 密文块数: {num_chunks}")

                    for chunk_idx, i in enumerate(range(0, total_elements, self.n_slots)):
                        chunk_data = flat[i:i + self.n_slots].tolist()
                        
                        # CKKS 批量加密
                        encrypted_vector = self.ts.ckks_vector(self.context, chunk_data)
                        serialized_chunk = encrypted_vector.serialize()
                        
                        is_first_chunk = (chunk_idx == 0)
                        is_last_chunk = (chunk_idx == num_chunks - 1)

                        shape_info = list(value.shape) if is_first_chunk else []
                        
                        enc_array_proto = federation_pb2.EncryptedNumpyArray(
                            shape=shape_info,
                            data=[serialized_chunk]  # 单个 CKKS 密文
                        )

                        param_chunk = federation_pb2.HeClientUpdateChunk(
                            parameters_chunk={key: enc_array_proto},
                            layer_name=key,
                            is_last_chunk_for_layer=is_last_chunk
                        )
                        
                        progress = min(i + self.n_slots, total_elements)
                        logger.info(f"客户端{self.client.client_id} 层 {key}: "
                                   f"块 {chunk_idx+1}/{num_chunks} ({(progress/total_elements)*100:.1f}%)")
                        yield param_chunk
                        
                    logger.info(f"客户端{self.client.client_id} 参数层 {key} 已全部发送。")
                    del flat
                    gc.collect()

                else:  # 处理标量
                    encrypted_value = self.ts.ckks_vector(self.context, [float(value)])
                    scalar_proto = federation_pb2.EncryptedNumpyArray(
                        data=[encrypted_value.serialize()], 
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
