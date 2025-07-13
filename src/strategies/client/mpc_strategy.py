import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
import numpy as np
from pyseltongue import secret_int_to_points
import gc

logger = get_logger()

class MpcClientStrategy(ClientStrategy):
    def __init__(self, client_instance):
        super().__init__(client_instance)
        self.shamir_k = int(config['mpc']['shamir_k'])
        self.shamir_n = int(config['mpc']['shamir_n'])
        self.scaling_factor = int(config['mpc']['scaling_factor'])
        self.chunk_size = int(config['mpc'].get('chunk_size', 10000)) # 从配置或使用默认值
        self.prime_mod = int(config['mpc']['prime_mod'])
        self.share_separator = ";"
        logger.info(f"客户端 {self.client.client_id} 的 MPC 策略已初始化 (k={self.shamir_k}, n={self.shamir_n}, chunk_size={self.chunk_size})。")

    def _float_to_int_shares(self, value):
        """将浮点数转换为缩放后的整数，再进行秘密共享。"""
        integer_value = int(value * self.scaling_factor)
        # secret_int_to_points 返回一个 (x, y) 元组的列表
        points = secret_int_to_points(integer_value, self.shamir_k, self.shamir_n, prime=self.prime_mod)
        # 为了传输，将点转换为 "x:y" 格式的字符串
        return [f"{p[0]}:{p[1]}" for p in points]

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（MPC份额）"""
        shared_parameters = {}
        logger.info("MPC策略: 开始创建秘密共享...")

        for key, value in model_parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten()
                total = len(flat)
                shares_by_party_chunks = [[] for _ in range(self.shamir_n)]
                logger.info(f"MPC策略: 开始处理参数 {key}, 形状: {value.shape}, 总量: {total}, 分块大小: {self.chunk_size}")

                for i in range(0, total, self.chunk_size):
                    chunk = flat[i:i+self.chunk_size]
                    
                    shares_by_pos = [self._float_to_int_shares(v) for v in chunk]
                    shares_by_party = list(zip(*shares_by_pos))

                    for party_idx, party_shares in enumerate(shares_by_party):
                        shares_by_party_chunks[party_idx].extend(party_shares)

                    progress = min(i + self.chunk_size, total)
                    logger.info(f"参数 {key} 秘密共享进度: {progress}/{total} ({(progress/total)*100:.1f}%)")

                    del chunk, shares_by_pos, shares_by_party
                    gc.collect()

                encoded_shares = [self.share_separator.join(party_shares).encode('utf-8') for party_shares in shares_by_party_chunks]

                shared_parameters[key] = {
                    'data': encoded_shares, 'shape': list(value.shape)
                }
            else: # scalar
                shares = self._float_to_int_shares(value)
                encoded_shares = [self.share_separator.join(shares).encode('utf-8')]
                shared_parameters[key] = {
                    'data': encoded_shares, 'shape': [1]
                }
            logger.info(f"参数 {key} 已完成秘密共享。")

        proto_params = {k: federation_pb2.SharedNumpyArray(data=v['data'], shape=v['shape']) for k, v in shared_parameters.items()}
        shared_model_params = federation_pb2.SharedModelParameters(parameters=proto_params)

        # --- 对训练指标进行秘密共享 ---
        metrics_to_share = metrics.copy()
        # 对auc, acc, loss进行加权，使其在聚合时可以正确平均
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
            # 确保我们只处理标量值，而不是数组或列表
            if isinstance(value, (int, float)):
                shares = self._float_to_int_shares(value)
                shared_metrics_dict[key] = self.share_separator.join(shares).encode('utf-8')
            else:
                logger.warning(f"跳过对非标量指标 '{key}' 的秘密共享。")


        shared_metrics_proto = federation_pb2.SharedTrainingMetrics(**shared_metrics_dict)
        
        # --- 组装 Payload ---
        mpc_payload = federation_pb2.MpcPayload(
            parameters_and_metrics=federation_pb2.SharedParametersAndMetrics(
                parameters=shared_model_params, 
                metrics=shared_metrics_proto
            )
        )

        return federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            mpc=mpc_payload
        ) 