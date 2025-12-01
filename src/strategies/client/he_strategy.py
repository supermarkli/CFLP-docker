import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
import numpy as np
import gc
import zlib
import time

logger = get_logger()

# 不需要分块加密的参数名模式 (这些通常是统计量或小型参数)
# 这些参数会用单个密文加密，减少密文数量开销
SKIP_CHUNKING_PATTERNS = {
    'num_batches_tracked',  # BatchNorm 计数器
    'running_mean',         # BatchNorm 运行均值
    'running_var',          # BatchNorm 运行方差
}

# 压缩配置
ENABLE_COMPRESSION = True
COMPRESSION_LEVEL = 6  # zlib 压缩级别 (1-9, 6 是默认值, 平衡速度和压缩率)


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
        
        # 获取 CKKS 参数限制
        poly_mod_degree = config['encryption']['poly_modulus_degree']
        max_slots = poly_mod_degree // 2  # CKKS 最大 slots = poly_modulus_degree / 2
        
        # 校正 chunk_size，确保不超过 max_slots
        configured_chunk_size = config['encryption']['chunk_size']
        if configured_chunk_size > max_slots:
            logger.warning(f"配置的 chunk_size ({configured_chunk_size}) 超过 CKKS 最大 slots ({max_slots})，"
                          f"自动校正为 {max_slots}。建议修改配置文件。")
            self.n_slots = max_slots
        else:
            self.n_slots = configured_chunk_size
        
        logger.info(f"客户端 {self.client.client_id} 的 CKKS 策略已初始化, "
                   f"每个密文打包 {self.n_slots} 个元素 (最大: {max_slots}), "
                   f"压缩: {'启用' if ENABLE_COMPRESSION else '禁用'}")
    
    def _should_skip_chunking(self, param_name):
        """检查参数是否应该跳过分块加密 (如 num_batches_tracked 等统计量)"""
        return any(pattern in param_name for pattern in SKIP_CHUNKING_PATTERNS)
    
    def _compress(self, data: bytes) -> bytes:
        """压缩数据 (如果启用压缩)"""
        if ENABLE_COMPRESSION:
            return zlib.compress(data, level=COMPRESSION_LEVEL)
        return data
    
    def _serialize_and_compress(self, encrypted_vector) -> bytes:
        """序列化并压缩 CKKS 向量"""
        serialized = encrypted_vector.serialize()
        return self._compress(serialized)

    def _encrypt_vector(self, flat_array, layer_name=""):
        """
        使用 CKKS 批量加密一个扁平化数组。
        
        利用 SIMD 特性，将数组分成多个 chunk，每个 chunk 打包到一个密文中。
        这比 Paillier 的逐元素加密快 10-100 倍。
        
        返回: (encrypted_chunks, stats_dict)
        """
        encrypted_chunks = []
        total = len(flat_array)
        num_chunks = (total + self.n_slots - 1) // self.n_slots
        
        total_raw_size = 0
        total_compressed_size = 0
        
        for i in range(0, total, self.n_slots):
            chunk = flat_array[i:i + self.n_slots]
            
            # CKKS 批量加密
            encrypted_vector = self.ts.ckks_vector(self.context, chunk.tolist())
            
            # 序列化并压缩
            serialized = encrypted_vector.serialize()
            compressed = self._compress(serialized)
            
            total_raw_size += len(serialized)
            total_compressed_size += len(compressed)
            
            encrypted_chunks.append(compressed)
        
        # 计算压缩比
        compression_ratio = (1 - total_compressed_size / total_raw_size) * 100 if total_raw_size > 0 else 0
        
        logger.debug(f"CKKS 加密: {layer_name} ({num_chunks} 块, "
                    f"原始: {total_raw_size/1024:.1f}KB, "
                    f"压缩后: {total_compressed_size/1024:.1f}KB, "
                    f"压缩率: {compression_ratio:.1f}%)")
        
        return encrypted_chunks

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（CKKS密文），供非流式接口使用。"""
        start_time = time.time()
        proto_params = {}
        skipped_params = []
        
        for key, value in model_parameters.items():
            # 跳过分块加密的统计量参数 (用单个密文)
            if self._should_skip_chunking(key):
                skipped_params.append(key)
                if isinstance(value, np.ndarray):
                    flat_value = value.flatten().astype(np.float64).tolist()
                else:
                    flat_value = [float(value)]
                encrypted_vector = self.ts.ckks_vector(self.context, flat_value)
                shape = list(value.shape) if isinstance(value, np.ndarray) and value.shape else [1]
                proto_params[key] = federation_pb2.EncryptedNumpyArray(
                    data=[self._serialize_and_compress(encrypted_vector)],
                    shape=shape
                )
                continue
                
            if isinstance(value, np.ndarray):
                flat = value.flatten().astype(np.float64)
                num_chunks = (len(flat) + self.n_slots - 1) // self.n_slots
                logger.info(f"CKKS策略: 加密参数 {key}, 形状: {value.shape}, "
                           f"总量: {len(flat)}, 密文数: {num_chunks}")
                
                encrypted_chunks = self._encrypt_vector(flat, key)
                
                encrypted_array_proto = federation_pb2.EncryptedNumpyArray(
                    shape=list(value.shape) if value.shape else [1],
                    data=encrypted_chunks
                )
                proto_params[key] = encrypted_array_proto
                
                del flat, encrypted_chunks
                gc.collect()
            else:
                # 标量处理
                encrypted_vector = self.ts.ckks_vector(self.context, [float(value)])
                proto_params[key] = federation_pb2.EncryptedNumpyArray(
                    data=[self._serialize_and_compress(encrypted_vector)],
                    shape=[1]
                )
        
        if skipped_params:
            logger.debug(f"跳过分块加密的参数: {skipped_params}")
        
        encrypt_time = time.time() - start_time
        logger.info(f"参数加密完成，耗时: {encrypt_time:.2f}s")

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
        
        # 使用闭包捕获 self 引用
        client_id = self.client.client_id
        n_slots = self.n_slots
        ts = self.ts
        context = self.context
        compress = self._compress
        should_skip = self._should_skip_chunking
        serialize_compress = self._serialize_and_compress
        
        def update_generator():
            start_time = time.time()
            total_bytes_sent = 0
            
            logger.info(f"客户端{client_id} 开始准备CKKS流式更新...")
            
            # --- 1. 准备和加密指标 ---
            metrics_to_encrypt = metrics.copy()
            if 'auc' in metrics_to_encrypt and 'test_num' in metrics_to_encrypt:
                metrics_to_encrypt['auc'] = metrics_to_encrypt['auc'] * metrics_to_encrypt.get('test_num', 1)

            metrics_values = [
                float(metrics_to_encrypt.get('test_acc', 0)),
                float(metrics_to_encrypt.get('test_num', 0)),
                float(metrics_to_encrypt.get('auc', 0)),
                float(metrics_to_encrypt.get('loss', 0)),
                float(metrics_to_encrypt.get('train_num', 0))
            ]
            encrypted_metrics_vector = ts.ckks_vector(context, metrics_values)
            encrypted_metrics_bytes = serialize_compress(encrypted_metrics_vector)
            
            encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(
                test_acc=encrypted_metrics_bytes,
                test_num=b'', auc=b'', loss=b'', train_num=b''
            )
            
            # --- 2. 发送第一个包含元数据和指标的块 ---
            initial_chunk = federation_pb2.HeClientUpdateChunk(
                client_id=client_id,
                round=current_round,
                metrics=encrypted_metrics_proto,
                parameters_chunk={},
                is_last_chunk_for_layer=True,
                layer_name="metadata"
            )
            total_bytes_sent += len(encrypted_metrics_bytes)
            yield initial_chunk

            # --- 3. 逐层加密并流式传输模型参数 ---
            total_layers = len(model_parameters)
            skipped_layers = []
            
            for layer_idx, (key, value) in enumerate(model_parameters.items()):
                skip_chunking = should_skip(key)
                
                if isinstance(value, np.ndarray):
                    flat = value.flatten().astype(np.float64)
                    total_elements = len(flat)
                    original_shape = list(value.shape) if value.shape else [1]
                    
                    if skip_chunking:
                        # 统计量参数用单个密文
                        skipped_layers.append(key)
                        encrypted_vector = ts.ckks_vector(context, flat.tolist())
                        compressed_data = serialize_compress(encrypted_vector)
                        total_bytes_sent += len(compressed_data)
                        
                        enc_array_proto = federation_pb2.EncryptedNumpyArray(
                            shape=original_shape,
                            data=[compressed_data]
                        )
                        param_chunk = federation_pb2.HeClientUpdateChunk(
                            parameters_chunk={key: enc_array_proto},
                            layer_name=key,
                            is_last_chunk_for_layer=True
                        )
                        yield param_chunk
                    else:
                        num_chunks = (total_elements + n_slots - 1) // n_slots
                        logger.info(f"客户端{client_id} 加密层 {key} "
                                   f"[{layer_idx+1}/{total_layers}]: {value.shape}, {num_chunks} 块")

                        for chunk_idx, i in enumerate(range(0, total_elements, n_slots)):
                            chunk_data = flat[i:i + n_slots].tolist()
                            
                            encrypted_vector = ts.ckks_vector(context, chunk_data)
                            compressed_chunk = serialize_compress(encrypted_vector)
                            total_bytes_sent += len(compressed_chunk)
                            
                            is_first_chunk = (chunk_idx == 0)
                            is_last_chunk = (chunk_idx == num_chunks - 1)
                            shape_info = original_shape if is_first_chunk else []
                            
                            enc_array_proto = federation_pb2.EncryptedNumpyArray(
                                shape=shape_info,
                                data=[compressed_chunk]
                            )

                            param_chunk = federation_pb2.HeClientUpdateChunk(
                                parameters_chunk={key: enc_array_proto},
                                layer_name=key,
                                is_last_chunk_for_layer=is_last_chunk
                            )
                            yield param_chunk
                    
                    del flat
                    gc.collect()

                else:  # 处理标量
                    encrypted_value = ts.ckks_vector(context, [float(value)])
                    compressed_data = serialize_compress(encrypted_value)
                    total_bytes_sent += len(compressed_data)
                    
                    scalar_proto = federation_pb2.EncryptedNumpyArray(
                        data=[compressed_data], 
                        shape=[1]
                    )
                    param_chunk = federation_pb2.HeClientUpdateChunk(
                        parameters_chunk={key: scalar_proto},
                        layer_name=key,
                        is_last_chunk_for_layer=True
                    )
                    yield param_chunk
            
            elapsed = time.time() - start_time
            logger.info(f"客户端{client_id} 流式加密完成: {total_layers} 层, "
                       f"总数据量: {total_bytes_sent/1024:.1f}KB, 耗时: {elapsed:.2f}s")
            if skipped_layers:
                logger.debug(f"跳过分块的统计量层: {skipped_layers}")

        return update_generator
