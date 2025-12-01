from collections import defaultdict
import threading
import numpy as np
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
from src.utils.parameter_utils import serialize_parameters
from src.utils.fast_shamir import fast_batch_recover, vectorized_shares_to_secret, _mod_inverse
import time

logger = get_logger()


class MpcAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)
        logger.info("MPC 聚合策略已初始化（高性能多进程版本）。")
        self.shamir_k = int(config['mpc']['shamir_k'])
        self.shamir_n = int(config['mpc']['shamir_n'])
        self.scaling_factor = int(config['mpc']['scaling_factor'])
        self.prime_mod = int(config['mpc']['prime_mod'])
        self.prime_mod_half = self.prime_mod // 2
        
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

    def _decode_binary_shares(self, data_bytes):
        """解码二进制格式的份额数据"""
        y_values = []
        offset = 0
        data_len = len(data_bytes)
        while offset < data_len:
            length = int.from_bytes(data_bytes[offset:offset + 2], 'big')
            offset += 2
            y_bytes = data_bytes[offset:offset + length]
            y_val = int.from_bytes(y_bytes, byteorder='big', signed=False)
            y_values.append(y_val)
            offset += length
        return y_values

    def _fast_aggregate_and_recover(self, client_updates, key, num_elements, num_clients):
        """
        快速聚合和恢复参数
        
        优化：
        1. 边解码边聚合，减少内存占用
        2. 使用预计算的拉格朗日系数
        3. 批量处理恢复
        """
        prime = self.prime_mod
        
        # 步骤 1: 解码并聚合份额
        # summed_y[party_idx] = list of summed y values
        summed_y_by_party = [None] * self.shamir_k  # 只需要 k 个份额
        
        for party_idx in range(self.shamir_k):
            # 初始化为零
            summed_y = [0] * num_elements
            
            # 累加所有客户端的份额
            for client_param_set in client_updates:
                shared_array = client_param_set[key]
                y_values = self._decode_binary_shares(shared_array.data[party_idx])
                
                for elem_idx in range(num_elements):
                    summed_y[elem_idx] = (summed_y[elem_idx] + y_values[elem_idx]) % prime
            
            summed_y_by_party[party_idx] = summed_y
        
        # 步骤 2: 批量恢复秘密
        result = fast_batch_recover(
            summed_y_by_party,
            self.shamir_k,
            self.prime_mod,
            self.scaling_factor,
            num_clients,
            chunk_size=50000
        )
        
        # 调试：检查恢复结果的统计信息
        if num_elements > 10000:
            result_arr = np.array(result)
            logger.debug(
                f"参数 {key} 恢复统计: min={result_arr.min():.6f}, max={result_arr.max():.6f}, "
                f"mean={result_arr.mean():.6f}, std={result_arr.std():.6f}, "
                f"nan_count={np.isnan(result_arr).sum()}, inf_count={np.isinf(result_arr).sum()}"
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
        """聚合指定轮次的客户端模型参数份额"""
        logger.info(f"[Round {round_num+1}] 开始MPC参数聚合（多进程并行版本）...")
        total_start = time.time()
        
        client_updates = list(self.server.client_parameters[round_num].values())
        if not client_updates:
            return self.server.global_model.get_parameters()

        aggregated_params = {}
        param_structure = client_updates[0]
        num_clients = len(client_updates)

        total_elements = sum(int(np.prod(param_structure[key].shape)) for key in param_structure.keys())
        processed_elements = 0

        for key in param_structure.keys():
            key_start = time.time()
            shape = list(param_structure[key].shape)
            num_elements = int(np.prod(shape))
            
            # 使用优化的聚合和恢复
            result = self._fast_aggregate_and_recover(
                client_updates, key, num_elements, num_clients
            )
            
            # 验证结果有效性
            if np.isnan(result).any() or np.isinf(result).any():
                nan_count = np.isnan(result).sum()
                inf_count = np.isinf(result).sum()
                logger.error(f"参数 {key} 恢复后包含无效值: NaN={nan_count}, Inf={inf_count}")
                # 用零替换无效值（临时修复）
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
        logger.info(
            f"[Round {round_num+1}] MPC参数聚合完成，"
            f"总耗时 {total_time:.2f}s，平均速度 {speed:.0f} 元素/秒"
        )
        return aggregated_params

    def evaluate_metrics(self, round_num):
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
                        y_values = self._decode_binary_shares(shares_bytes)
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

        avg_acc = (total_test_acc_num / scaling_factor) / final_test_num
        avg_auc = (total_auc_num / scaling_factor) / final_test_num
        avg_loss = (total_loss_num / scaling_factor) / final_train_num

        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估 (MPC): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")

        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num]
