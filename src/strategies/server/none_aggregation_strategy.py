import threading
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.utils.logging_config import get_logger

logger = get_logger()


class NoneAggregationStrategy(AggregationStrategy):
    def __init__(self, server_instance):
        super().__init__(server_instance)

    def prepare_setup_response(self, request):
        return federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )

    def aggregate(self, request, context):
        payload = request.plaintext
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'none' 模式不匹配。")
        
        try:
            client_id = request.client_id
            round_num = request.round
            
            with self.server.lock:
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}")

                # 1. 处理明文更新
                params, metrics_data = self._process_plaintext_update(payload)
                self.server.clients[client_id].metrics = metrics_data
                self.server.client_parameters[round_num][client_id] = params
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的明文更新。")

                # 2. 检查是否所有客户端都已提交，如果是则启动聚合
                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    # 使用线程以避免阻塞gRPC调用
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="更新已收到"
                )

        except Exception as e:
            logger.error(f"处理明文更新时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def _process_plaintext_update(self, payload):
        parameters = deserialize_parameters(payload.parameters_and_metrics.parameters.parameters)
        metrics = payload.parameters_and_metrics.metrics
        metrics_data = {
            'test_acc': metrics.test_acc, 'test_num': metrics.test_num, 'auc': metrics.auc,
            'loss': metrics.loss, 'train_num': metrics.train_num
        }
        return parameters, metrics_data 

    def aggregate_parameters(self, round_num):
        """聚合明文客户端参数 (FedAvg)"""
        active_clients = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]
        parameters_list = list(self.server.client_parameters[round_num].values())
        
        total_data_size = sum(client.data_size for client in active_clients)
        if total_data_size == 0: return self.server.global_model.get_parameters()
        
        client_weights = [client.data_size / total_data_size for client in active_clients]
        
        aggregated = {}
        param_structure = parameters_list[0]
        for param_name in param_structure.keys():
            aggregated[param_name] = sum(weight * params[param_name] for params, weight in zip(parameters_list, client_weights))
        
        logger.info(f"[Round {round_num+1}] 明文参数聚合完成。")
        return aggregated

    def evaluate_metrics(self, round_num):
        """评估明文指标"""
        total_test_acc, total_test_num = 0, 0
        total_auc, total_loss, total_train_num = 0, 0, 0
        
        clients_in_round = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]

        for c in clients_in_round:
            m = c.metrics
            if m:
                total_test_acc += m['test_acc']
                total_test_num += m['test_num']
                total_auc += m['auc'] * m['test_num']
                total_loss += m['loss']
                total_train_num += m['train_num']
        
        # 清理本轮存储的指标
        for c in clients_in_round: c.metrics = None
        
        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0
        
        self.server.rs_test_acc.append(avg_acc)
        self.server.rs_train_loss.append(avg_loss)
        self.server.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估: Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")

        # 清理本轮的参数
        # Note: HE模式下这个清理需要晚于指标评估，所以放在这里统一处理
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num] 