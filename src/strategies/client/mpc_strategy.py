import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
from src.utils.fast_shamir import (
    fast_batch_share, 
    fast_batch_share_variable,
    vectorized_secret_to_shares
)
import numpy as np
import gc
import time

logger = get_logger()


class MpcClientStrategy(ClientStrategy):
    def __init__(self, client_instance):
        super().__init__(client_instance)
        self.shamir_k = int(config['mpc']['shamir_k'])
        self.shamir_n = int(config['mpc']['shamir_n'])
        self.scaling_factor = int(config['mpc']['scaling_factor'])
        self.chunk_size = int(config['mpc'].get('chunk_size', 50000))
        self.prime_mod = int(config['mpc']['prime_mod'])
        
        # 使用固定 8 字节编码（适用于 prime < 2^64）
        self.use_fixed_encoding = self.prime_mod < (1 << 64)
        
        logger.info(
            f"客户端 {self.client.client_id} 的 MPC 策略已初始化 "
            f"(k={self.shamir_k}, n={self.shamir_n}, chunk_size={self.chunk_size}, "
            f"encoding={'fixed-8byte' if self.use_fixed_encoding else 'variable'})。"
        )

    def _share_scalar_binary(self, value):
        """为标量值生成二进制编码的秘密共享（变长格式，用于 metrics）"""
        integer_value = int(value * self.scaling_factor)
        
        shares = vectorized_secret_to_shares(
            np.array([integer_value]), 
            self.shamir_k, 
            self.shamir_n, 
            self.prime_mod
        )
        
        # 变长编码（metrics 数据量小，保持兼容性）
        byte_parts = []
        for party_idx in range(self.shamir_n):
            y_val = int(shares[party_idx][0])
            y_bytes = y_val.to_bytes(
                max(1, (y_val.bit_length() + 7) // 8),
                byteorder='big',
                signed=False
            )
            byte_parts.append(len(y_bytes).to_bytes(2, 'big') + y_bytes)
        
        return b''.join(byte_parts)

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（MPC份额）- 高性能版本 v2"""
        total_start = time.time()
        logger.info(f"MPC策略: 开始创建秘密共享（v2, {'固定8字节' if self.use_fixed_encoding else '变长'}编码）...")
        
        shared_parameters = {}
        
        # 计算总参数量
        total_params = sum(
            np.prod(v.shape) if isinstance(v, np.ndarray) else 1 
            for v in model_parameters.values()
        )
        logger.info(f"MPC策略: 总参数量: {total_params:,}")
        
        processed_count = 0
        
        # 选择编码函数
        share_func = fast_batch_share if self.use_fixed_encoding else fast_batch_share_variable
        
        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                layer_start = time.time()
                num_elements = int(np.prod(value.shape))
                
                # 使用高性能批量共享
                encoded_shares = share_func(
                    value,
                    self.shamir_k,
                    self.shamir_n,
                    self.scaling_factor,
                    self.prime_mod,
                    self.chunk_size
                )
                
                shared_parameters[key] = {
                    'data': encoded_shares,
                    'shape': list(value.shape)
                }
                
                processed_count += num_elements
                layer_time = time.time() - layer_start
                
                if num_elements > 10000:
                    speed = num_elements / layer_time if layer_time > 0 else 0
                    data_size = sum(len(d) for d in encoded_shares) / 1024 / 1024
                    logger.info(
                        f"参数 {key} ({num_elements:,} 元素) 完成，"
                        f"耗时 {layer_time:.2f}s，速度 {speed:.0f} 元素/秒，"
                        f"数据量 {data_size:.2f} MB，"
                        f"进度: {processed_count:,}/{total_params:,} ({100*processed_count/total_params:.1f}%)"
                    )
            else:
                # 标量
                shares = vectorized_secret_to_shares(
                    np.array([int(value * self.scaling_factor)]),
                    self.shamir_k,
                    self.shamir_n,
                    self.prime_mod
                )
                
                if self.use_fixed_encoding:
                    # 固定 8 字节编码
                    byte_data = np.array([int(shares[party_idx][0]) for party_idx in range(self.shamir_n)], 
                                        dtype='>u8').tobytes()
                    shared_parameters[key] = {
                        'data': [byte_data],
                        'shape': [1]
                    }
                else:
                    # 变长编码
                    byte_parts = []
                    for party_idx in range(self.shamir_n):
                        y_val = int(shares[party_idx][0])
                        y_bytes = y_val.to_bytes(
                            max(1, (y_val.bit_length() + 7) // 8),
                            byteorder='big',
                            signed=False
                        )
                        byte_parts.append(len(y_bytes).to_bytes(2, 'big') + y_bytes)
                    
                    shared_parameters[key] = {
                        'data': [b''.join(byte_parts)],
                        'shape': [1]
                    }
                processed_count += 1
        
        # 转换为 Protobuf 格式
        proto_params = {
            k: federation_pb2.SharedNumpyArray(data=v['data'], shape=v['shape']) 
            for k, v in shared_parameters.items()
        }
        shared_model_params = federation_pb2.SharedModelParameters(parameters=proto_params)

        # --- 对训练指标进行秘密共享（始终使用变长编码）---
        metrics_to_share = metrics.copy()
        
        if 'test_num' in metrics_to_share:
            test_num = metrics_to_share.get('test_num', 1)
            if 'auc' in metrics_to_share:
                metrics_to_share['auc'] *= test_num
        if 'train_num' in metrics_to_share:
            train_num = metrics_to_share.get('train_num', 1)
            if 'loss' in metrics_to_share:
                metrics_to_share['loss'] *= train_num

        shared_metrics_dict = {}
        for key, value in metrics_to_share.items():
            if isinstance(value, (int, float)):
                shared_metrics_dict[key] = self._share_scalar_binary(value)
            else:
                logger.warning(f"跳过对非标量指标 '{key}' 的秘密共享。")

        shared_metrics_proto = federation_pb2.SharedTrainingMetrics(**shared_metrics_dict)
        
        # 组装 Payload
        mpc_payload = federation_pb2.MpcPayload(
            parameters_and_metrics=federation_pb2.SharedParametersAndMetrics(
                parameters=shared_model_params, 
                metrics=shared_metrics_proto
            )
        )
        
        # 计算总数据量
        total_data_size = sum(
            sum(len(d) for d in v['data']) 
            for v in shared_parameters.values()
        )
        
        # 清理内存
        del shared_parameters
        gc.collect()

        total_time = time.time() - total_start
        speed = total_params / total_time if total_time > 0 else 0
        logger.info(
            f"MPC策略: 秘密共享完成，总耗时 {total_time:.2f}s，"
            f"平均速度 {speed:.0f} 元素/秒，"
            f"总数据量 {total_data_size / 1024 / 1024:.2f} MB"
        )

        return federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            mpc=mpc_payload
        )
