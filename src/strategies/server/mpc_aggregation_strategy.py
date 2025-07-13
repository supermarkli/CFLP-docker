from collections import defaultdict
import threading
import numpy as np
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.config_utils import config
from src.utils.logging_config import get_logger
from src.utils.parameter_utils import serialize_parameters
from pyseltongue import points_to_secret_int

logger = get_logger()

class MpcAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)
        logger.info("MPC 聚合策略已初始化。")
        self.shamir_k = int(config['mpc']['shamir_k'])
        self.shamir_n = int(config['mpc']['shamir_n'])
        self.scaling_factor = int(config['mpc']['scaling_factor'])
        self.prime_mod = int(config['mpc']['prime_mod'])
        self.prime_mod_half = self.prime_mod // 2
        self.share_separator = ";"

    def _to_signed_int(self, n):
        """将从有限域恢复出的数字转换为带符号整数。"""
        if n > self.prime_mod_half:
            return n - self.prime_mod
        return n

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
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}")

                self.server.clients[client_id].shared_metrics = payload.parameters_and_metrics.metrics
                self.server.client_parameters[round_num][client_id] = payload.parameters_and_metrics.parameters.parameters
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的MPC份额更新。")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

                return federation_pb2.ServerUpdate(code=200, current_round=self.server.current_round, message="份额更新已收到")

        except Exception as e:
            logger.error(f"处理MPC份额时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def aggregate_parameters(self, round_num):
        """聚合指定轮次的客户端模型参数份额"""
        logger.info(f"[Round {round_num+1}] 开始MPC参数聚合...")
        
        client_updates = list(self.server.client_parameters[round_num].values())
        if not client_updates:
            return self.server.global_model.get_parameters()

        aggregated_params = {}
        param_structure = client_updates[0]
        num_clients = len(client_updates)

        for key in param_structure.keys():
            shape = param_structure[key].shape
            num_elements = int(np.prod(shape))
            reconstructed_flat_array = np.zeros(num_elements)

            # Pre-process and combine shares
            processed_shares = []
            for client_param_set in client_updates:
                shared_array = client_param_set[key]
                party_shares_by_element = []
                for party_idx in range(self.shamir_n):
                    party_shares_str = shared_array.data[party_idx].decode('utf-8')
                    shares_for_party = party_shares_str.split(self.share_separator)
                    party_shares_by_element.append(shares_for_party)
                
                element_shares_by_party = list(zip(*party_shares_by_element))
                processed_shares.append(element_shares_by_party)

            # Aggregate element by element
            for i in range(num_elements):
                element_shares_from_all_clients = []
                for client_idx in range(num_clients):
                    element_shares_from_all_clients.append(processed_shares[client_idx][i])
                
                # Transpose to group by party index
                shares_by_party = list(zip(*element_shares_from_all_clients))
                
                summed_points_for_recovery = []
                for party_shares in shares_by_party:
                    # party_shares is a list of "x:y" strings, e.g., ["1:123", "1:456"]
                    # All shares for a party have the same x-coordinate.
                    point_x = int(party_shares[0].split(':')[0])
                    # Parse the points and sum the y-coordinates.
                    point_y_values = [int(s.split(':')[1]) for s in party_shares]
                    # Perform summation in the specified prime field
                    summed_point_y = sum(point_y_values) % self.prime_mod
                    
                    summed_points_for_recovery.append((point_x, summed_point_y))

                # Use k of the summed points to recover the summed secret integer.
                reconstructed_unsigned_int = points_to_secret_int(summed_points_for_recovery[:self.shamir_k], prime=self.prime_mod)
                reconstructed_int = self._to_signed_int(reconstructed_unsigned_int)
                
                # Averaging
                reconstructed_value = float(reconstructed_int) / (self.scaling_factor * num_clients)
                reconstructed_flat_array[i] = reconstructed_value
            
            aggregated_params[key] = reconstructed_flat_array.reshape(shape)
            logger.info(f"参数 {key} 已通过MPC聚合。")

        logger.info(f"[Round {round_num+1}] MPC参数聚合完成。")
        return aggregated_params

    def evaluate_metrics(self, round_num):
        """评估指定轮次的客户端指标份额"""
        logger.info(f"[Round {round_num+1}] 开始MPC指标评估...")
        clients_in_round = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]
        
        agg_metrics_shares_by_party = defaultdict(list)

        for c in clients_in_round:
            sm = c.shared_metrics
            if sm:
                for key in sm.DESCRIPTOR.fields_by_name:
                    shares_str = getattr(sm, key).decode('utf-8')
                    shares = shares_str.split(self.share_separator)
                    agg_metrics_shares_by_party[key].append(shares)
        
        decrypted_metrics = {}
        for key, client_shares_list in agg_metrics_shares_by_party.items():
            # Transpose to group by party index
            shares_by_party = list(zip(*client_shares_list))
            
            summed_points_for_recovery = []
            for party_shares in shares_by_party:
                point_x = int(party_shares[0].split(':')[0])
                point_y_values = [int(s.split(':')[1]) for s in party_shares]
                # Perform summation in the specified prime field
                summed_point_y = sum(point_y_values) % self.prime_mod
                summed_points_for_recovery.append((point_x, summed_point_y))

            reconstructed_unsigned_int = points_to_secret_int(summed_points_for_recovery[:self.shamir_k], prime=self.prime_mod)
            decrypted_metrics[key] = self._to_signed_int(reconstructed_unsigned_int)

        logger.info(f"解密后的聚合指标(原始值): {decrypted_metrics}")

        scaling_factor = self.scaling_factor
        
        # 分子 (numerator) 是加权并缩放后的值
        total_test_acc_num = decrypted_metrics.get('test_acc', 0)
        total_auc_num = decrypted_metrics.get('auc', 0) 
        total_loss_num = decrypted_metrics.get('loss', 0)

        # 分母 (denominator) 也是缩放后的值
        total_test_num_den = decrypted_metrics.get('test_num', 1)
        total_train_num_den = decrypted_metrics.get('train_num', 1)

        # 清理本轮存储的加密指标
        for c in clients_in_round: c.shared_metrics = None

        # 正确的加权平均计算方法：
        # 分子和分母都需要先除以scaling_factor来“解缩放”
        # 然后再进行除法。或者，直接相除，scaling_factor会抵消。
        # 之前的逻辑是正确的，但结果却有问题，这暗示着可能存在数值精度问题
        # 我们采用更稳健的 HE 策略的计算方式：先对分子解缩放，再除以分母
        
        # 为了避免除以零，我们确保分母不为零
        # 由于分母也被缩放了，我们需要先把它解缩放回来
        final_test_num = (total_test_num_den / scaling_factor) if scaling_factor != 0 else total_test_num_den
        final_train_num = (total_train_num_den / scaling_factor) if scaling_factor != 0 else total_train_num_den
        
        if final_test_num == 0: final_test_num = 1
        if final_train_num == 0: final_train_num = 1

        avg_acc = (total_test_acc_num / scaling_factor) / final_test_num
        avg_auc = (total_auc_num / scaling_factor) / final_test_num
        avg_loss = (total_loss_num / scaling_factor) / final_train_num

        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估 (MPC): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")

        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num] 