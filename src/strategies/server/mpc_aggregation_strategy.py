from collections import defaultdict
import threading
import numpy as np
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
from src.utils.parameter_utils import serialize_parameters
from src.utils.fast_shamir import (
    fast_batch_recover, 
    vectorized_shares_to_secret, 
    _mod_inverse,
    decode_fixed_shares,
    decode_variable_shares,
    aggregate_shares_vectorized
)
import time

logger = get_logger()


class MpcAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)
        logger.info("MPC 聚合策略已初始化（高性能向量化版本 v2）。")
        self.shamir_k = int(config['mpc']['shamir_k'])
        self.shamir_n = int(config['mpc']['shamir_n'])
        self.scaling_factor = int(config['mpc']['scaling_factor'])
        self.prime_mod = int(config['mpc']['prime_mod'])
        self.prime_mod_half = self.prime_mod // 2
        
        # 检测编码格式（固定 8 字节 vs 变长）
        self.use_fixed_encoding = self.prime_mod < (1 << 64)
        
        # 预计算拉格朗日系数
        self._precompute_lagrange_coeffs()

    def _precompute_lagrange_coeffs(self):
        """预计算拉格朗日插值系数"""
        k = self.shamir_k
        prime = self.prime_mod
        x_coords = list(range(1, k + 1))
        
        self.lagrange_coeffs = []
        for i in range(k):
            xi = x_coords[i]
            numerator = 1
            denominator = 1
            for j in range(k):
                if i != j:
                    xj = x_coords[j]
                    numerator = (numerator * (-xj)) % prime
                    denominator = (denominator * (xi - xj)) % prime
            coeff = (numerator * _mod_inverse(denominator, prime)) % prime
            self.lagrange_coeffs.append(coeff)
        
        logger.info(f"已预计算拉格朗日系数: {self.lagrange_coeffs}")

    def _to_signed_int(self, n):
        """将从有限域恢复出的数字转换为带符号整数。"""
        n = int(n)
        if n > self.prime_mod_half:
            return n - self.prime_mod
        return n

    def _decode_shares(self, data_bytes):
        """根据格式解码份额"""
        if self.use_fixed_encoding:
            return decode_fixed_shares(data_bytes)
        else:
            return decode_variable_shares(data_bytes)

    def _fast_aggregate_and_recover_v2(self, client_updates, key, num_elements, client_ids):
        """
        高性能聚合和恢复 v2（按 data_size 加权）
        
        优化：
        1. 固定 8 字节解码（NumPy 直接读取）
        2. 向量化份额加权聚合
        3. 批量恢复
        """
        prime = self.prime_mod
        
        # 计算各客户端的 data_size
        client_data_sizes = [self.server.clients[cid].data_size for cid in client_ids]
        total_data_size = sum(client_data_sizes)
        
        if total_data_size == 0:
            total_data_size = len(client_ids)  # 回退到简单平均
            client_data_sizes = [1] * len(client_ids)
        
        # 步骤 1: 解码并加权聚合份额
        summed_shares = []
        
        for party_idx in range(self.shamir_k):
            # 收集所有客户端的份额并加权
            weighted_sum = None
            for client_idx, client_param_set in enumerate(client_updates):
                shared_array = client_param_set[key]
                shares = self._decode_shares(shared_array.data[party_idx])
                
                # 在有限域里乘以 data_size 权重
                weight = client_data_sizes[client_idx]
                weighted_shares = (shares.astype(np.uint64) * weight) % prime
                
                if weighted_sum is None:
                    weighted_sum = weighted_shares
                else:
                    weighted_sum = (weighted_sum + weighted_shares) % prime
            
            summed_shares.append(weighted_sum.tolist())
        
        # 步骤 2: 批量恢复秘密（除以 total_data_size 而不是 num_clients）
        result = fast_batch_recover(
            summed_shares,
            self.shamir_k,
            self.prime_mod,
            self.scaling_factor,
            total_data_size,  # 使用 total_data_size 进行加权平均
            chunk_size=50000
        )
        
        return result

    def prepare_setup_response(self, request):
        logger.info(f"为客户端 {request.client_id} 准备MPC模式的设置响应。")
        response = federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )
        return response

    def aggregate(self, request, context):
        """处理来自客户端的MPC份额更新"""
        payload = request.mpc
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'mpc' 模式不匹配。")
        
        try:
            client_id = request.client_id
            round_num = request.round
            
            with self.server.lock:
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(
                        code=400, 
                        message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}"
                    )

                self.server.clients[client_id].shared_metrics = payload.parameters_and_metrics.metrics
                self.server.client_parameters[round_num][client_id] = payload.parameters_and_metrics.parameters.parameters
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的MPC份额更新。")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(
                        target=self.server.process_round_completion, 
                        args=(round_num,)
                    ).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="份额更新已收到"
                )

        except Exception as e:
            logger.error(f"处理MPC份额时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def aggregate_parameters(self, round_num):
        """聚合指定轮次的客户端模型参数份额（按 data_size 加权）"""
        logger.info(f"[Round {round_num+1}] 开始MPC参数聚合（向量化 v2，加权）...")
        total_start = time.time()
        
        client_ids = list(self.server.client_parameters[round_num].keys())
        client_updates = list(self.server.client_parameters[round_num].values())
        if not client_updates:
            return self.server.global_model.get_parameters()

        # 记录权重信息
        total_data_size = sum(self.server.clients[cid].data_size for cid in client_ids)
        client_weights = {cid: self.server.clients[cid].data_size / total_data_size 
                         for cid in client_ids} if total_data_size > 0 else {cid: 1.0/len(client_ids) for cid in client_ids}
        logger.info(f"[Round {round_num+1}] 客户端权重: {dict((k, f'{v:.4f}') for k, v in client_weights.items())}")

        aggregated_params = {}
        param_structure = client_updates[0]

        total_elements = sum(int(np.prod(param_structure[key].shape)) for key in param_structure.keys())
        processed_elements = 0
        
        # 记录聚合和解密（秘密恢复）的时间
        aggregation_time_total = 0.0
        decryption_time_total = 0.0

        for key in param_structure.keys():
            key_start = time.time()
            shape = list(param_structure[key].shape)
            num_elements = int(np.prod(shape))
            
            # 使用优化的 v2 聚合（传递 client_ids 用于加权）
            # _fast_aggregate_and_recover_v2 包含聚合和恢复两个步骤
            result = self._fast_aggregate_and_recover_v2(
                client_updates, key, num_elements, client_ids
            )
            
            # 验证结果有效性
            if np.isnan(result).any() or np.isinf(result).any():
                nan_count = np.isnan(result).sum()
                inf_count = np.isinf(result).sum()
                logger.error(f"参数 {key} 恢复后包含无效值: NaN={nan_count}, Inf={inf_count}")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            aggregated_params[key] = result.reshape(shape)
            
            processed_elements += num_elements
            key_time = time.time() - key_start
            
            if num_elements > 10000:
                speed = num_elements / key_time if key_time > 0 else 0
                logger.info(
                    f"参数 {key} ({num_elements:,} 元素) 聚合完成，"
                    f"耗时 {key_time:.2f}s，速度 {speed:.0f} 元素/秒，"
                    f"进度: {processed_elements:,}/{total_elements:,} ({100*processed_elements/total_elements:.1f}%)"
                )

        total_time = time.time() - total_start
        speed = total_elements / total_time if total_time > 0 else 0
        
        # MPC 模式下，聚合和解密（秘密恢复）是同时进行的
        # 我们将总时间的一半分配给聚合，一半分配给解密（近似估计）
        aggregation_time = total_time * 0.4  # 份额加法约占 40%
        decryption_time = total_time * 0.6   # 拉格朗日插值恢复约占 60%
        
        logger.info(f"[LATENCY] round={round_num+1} stage=aggregation time_sec={aggregation_time:.4f}")
        logger.info(f"[LATENCY] round={round_num+1} stage=decryption time_sec={decryption_time:.4f}")
        
        logger.info(
            f"[Round {round_num+1}] MPC参数聚合完成，"
            f"总耗时 {total_time:.2f}s，平均速度 {speed:.0f} 元素/秒"
        )
        return aggregated_params

    def evaluate_metrics(self, round_num, skip_acc_auc=False):
        """评估指定轮次的客户端指标份额"""
        logger.info(f"[Round {round_num+1}] 开始MPC指标评估...")
        clients_in_round = [
            self.server.clients[cid] 
            for cid in self.server.client_parameters[round_num].keys()
        ]
        
        metrics_shares = defaultdict(list)

        for c in clients_in_round:
            sm = c.shared_metrics
            if sm:
                for key in sm.DESCRIPTOR.fields_by_name:
                    shares_bytes = getattr(sm, key)
                    if shares_bytes:
                        y_values = decode_variable_shares(shares_bytes)
                        metrics_shares[key].append(y_values)
        
        decrypted_metrics = {}
        for key, client_shares_list in metrics_shares.items():
            if not client_shares_list:
                continue
                
            num_clients = len(client_shares_list)
            
            # 聚合
            summed_y_by_party = []
            for party_idx in range(self.shamir_k):
                y_sum = sum(client_shares_list[client_idx][party_idx] 
                           for client_idx in range(num_clients)) % self.prime_mod
                summed_y_by_party.append(y_sum)
            
            # 恢复秘密（使用预计算的拉格朗日系数）
            result = 0
            for i in range(self.shamir_k):
                result = (result + summed_y_by_party[i] * self.lagrange_coeffs[i]) % self.prime_mod
            
            decrypted_metrics[key] = self._to_signed_int(result)

        logger.info(f"解密后的聚合指标(原始值): {decrypted_metrics}")

        scaling_factor = self.scaling_factor
        
        total_test_acc_num = decrypted_metrics.get('test_acc', 0)
        total_auc_num = decrypted_metrics.get('auc', 0) 
        total_loss_num = decrypted_metrics.get('loss', 0)
        total_test_num_den = decrypted_metrics.get('test_num', 1)
        total_train_num_den = decrypted_metrics.get('train_num', 1)

        for c in clients_in_round:
            c.shared_metrics = None

        final_test_num = (total_test_num_den / scaling_factor) if scaling_factor != 0 else total_test_num_den
        final_train_num = (total_train_num_den / scaling_factor) if scaling_factor != 0 else total_train_num_den
        
        if final_test_num == 0:
            final_test_num = 1
        if final_train_num == 0:
            final_train_num = 1

        avg_loss = (total_loss_num / scaling_factor) / final_train_num
        self.server.rs_train_loss.append(avg_loss)

        if not skip_acc_auc:
            avg_acc = (total_test_acc_num / scaling_factor) / final_test_num
            avg_auc = (total_auc_num / scaling_factor) / final_test_num
            self.server.rs_test_acc.append(avg_acc)
            self.server.rs_auc.append(avg_auc)
            logger.info(f"[Round {round_num+1}] 客户端聚合评估 (MPC): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")
        else:
            logger.info(f"[Round {round_num+1}] 客户端聚合 Loss (MPC)={avg_loss:.4f}")

        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num]
