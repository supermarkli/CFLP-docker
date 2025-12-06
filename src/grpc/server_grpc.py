import grpc
from concurrent import futures
import os
import sys
import numpy as np
from collections import defaultdict
import pandas as pd
import threading
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import hashlib
import json
import pickle
import gc
import psutil

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging_config import get_logger
from src.models.models import get_model
from src.grpc.generated import federation_pb2
from src.grpc.generated import federation_pb2_grpc
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
# from src.utils.draw import plot_global_convergence_curve
from src.utils.config_utils import config
from src.strategies.server.none_aggregation_strategy import NoneAggregationStrategy
from src.strategies.server.he_aggregation_strategy import HeAggregationStrategy
from src.strategies.server.tee_aggregation_strategy import TeeAggregationStrategy
from src.strategies.server.mpc_aggregation_strategy import MpcAggregationStrategy
from src.strategies.server.sgx_aggregation_strategy import SgxAggregationStrategy

logger = get_logger(create_file=True)
set_seed(config['base']['random_seed'])

class ClientState:
    def __init__(self, client_id, model_type, data_size):
        self.client_id = client_id
        self.model_type = model_type
        self.data_size = data_size
        self.current_round = 0
        self.metrics = None
        self.encrypted_metrics = None

class FederatedLearningServicer(federation_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从配置中读取数据集和模型名称
        self.dataset_name = config['data']['dataset']
        self.model_name = config['model']['name']
        logger.info(f"[Server] 配置: Dataset={self.dataset_name}, Model={self.model_name}")

        self.global_model = get_model(self.model_name, self.dataset_name).to(self.device)
        self.clients = {}
        self.current_round = 0
        self.lock = threading.Lock()
        # 新增：使用 Condition 替代单独的 Lock，用于状态变化通知
        self.status_condition = threading.Condition(self.lock)
        self.client_parameters = defaultdict(dict) 
        self.completed_clients = defaultdict(set) 
        self.converged = False
        self.start_time = None
        self.end_time = None
        self.rs_test_acc, self.rs_train_loss, self.rs_auc = [], [], []
        self.count = 0
        self.next_step = False
        self.privacy_mode = config['federation']['privacy_mode']
        self.expected_clients = config['federation']['expected_clients']
        self.max_rounds = config['federation']['max_rounds']
        self.acc_delta_threshold = config['federation']['convergence']['acc_delta_threshold']
        self.converge_window = config['federation']['convergence']['window']
        self.logger=logger
        self.communication_costs = {} # <-- 新增：用于存储每轮各客户端的通信开销
        
        # 加载全局测试集用于评估全局模型
        self.global_test_loader = self._load_global_test_set()
        
        self.aggregation_strategy = self._create_aggregation_strategy()
        if not self.aggregation_strategy:
            raise ValueError(f"不支持的隐私模式或初始化策略失败: {self.privacy_mode}")

        logger.info(f"[Server] 初始化完成 (模式: {self.privacy_mode.upper()})，等待 {self.expected_clients} 个客户端注册")
    
    def _load_global_test_set(self):
        """加载全局测试集用于评估全局模型"""
        import numpy as np
        global_test_path = f"/app/data/global/{self.dataset_name}_test.npz"
        
        try:
            test_data = np.load(global_test_path)
            X_test = test_data["X_test"]
            y_test = test_data["y_test"]
            
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            eval_batch_size = config['training'].get('eval_batch_size', 512)
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
            
            logger.info(f"[Server] 加载全局测试集: {global_test_path}, 样本数: {len(X_test)}")
            return test_loader
        except FileNotFoundError:
            logger.warning(f"[Server] 未找到全局测试集: {global_test_path}，使用客户端聚合指标评估")
            return None
        except Exception as e:
            logger.error(f"[Server] 加载全局测试集出错: {e}")
            return None
    
    def evaluate_global_model(self, round_num):
        """使用全局测试集评估全局模型"""
        if self.global_test_loader is None:
            return None, None, None
        
        self.global_model.eval()
        test_acc = 0
        test_num = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.global_test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                
                # 收集用于计算 AUC
                probs = torch.softmax(output, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.cpu().numpy())
        
        accuracy = test_acc / test_num if test_num > 0 else 0
        
        # 计算 AUC
        try:
            from sklearn import metrics
            from sklearn.preprocessing import label_binarize
            import numpy as np
            
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            num_classes = all_probs.shape[1]
            labels_binarized = label_binarize(all_labels, classes=np.arange(num_classes))
            auc = metrics.roc_auc_score(labels_binarized, all_probs, average='micro')
        except Exception as e:
            logger.warning(f"[Server] 计算 AUC 出错: {e}")
            auc = 0
        
        logger.info(f"[Round {round_num+1}] 全局评估: Acc={accuracy:.4f}, AUC={auc:.4f}")
        return accuracy, auc, test_num

    def _create_aggregation_strategy(self):
        """根据配置创建并返回相应的聚合策略实例。"""
        if self.privacy_mode == 'none':
            return NoneAggregationStrategy(self)
        elif self.privacy_mode == 'he':
            return HeAggregationStrategy(self)
        elif self.privacy_mode == 'tee':
            return TeeAggregationStrategy(self)
        elif self.privacy_mode == 'mpc':
            return MpcAggregationStrategy(self)
        elif self.privacy_mode == 'sgx':
            return SgxAggregationStrategy(self)
        else:
            return None

    def RegisterAndSetup(self, request, context):
        client_id = request.client_id
        logger.info(f"[Server] 收到客户端 {client_id} 注册请求 (模型: {request.model_type}, 数据量: {request.data_size})")

        with self.status_condition:  # 使用 Condition 替代 Lock
            if client_id not in self.clients:
                self.clients[client_id] = ClientState(
                    client_id=client_id,
                    model_type=request.model_type,
                    data_size=request.data_size
                )
                logger.info(f"[Server] 客户端 {client_id} 注册成功 ({len(self.clients)}/{self.expected_clients})")
 
            response = self.aggregation_strategy.prepare_setup_response(request)

            if len(self.clients) >= self.expected_clients:
                self.next_step = True
                logger.info(f"[Server] 所有客户端已注册，准备开始训练")
                if self.start_time is None:
                    self.start_time = time.time()
                    logger.info("[Server] 联邦学习计时开始")
                # 唤醒所有等待状态更新的客户端
                self.status_condition.notify_all()

            return response

    def SubscribeTrainingStatus(self, request, context):
        """
        新增：流式RPC - 客户端订阅训练状态，服务器在状态变化时推送。
        使用 Condition 变量实现高效的等待/通知机制，消除轮询导致的锁竞争。
        """
        client_id = request.client_id
        logger.debug(f"[Server] 客户端 {client_id} 订阅训练状态")
        
        while context.is_active():
            with self.status_condition:
                # 使用 wait_for 模式：在循环中等待直到条件满足
                # 短超时(0.5秒)确保即使错过notify也能快速响应
                while not (self.converged or self.next_step):
                    # wait() 会自动释放锁，被唤醒后重新获取锁
                    self.status_condition.wait(timeout=0.5)
                    # 检查连接是否仍然活跃
                    if not context.is_active():
                        logger.debug(f"客户端 {client_id} 连接已断开")
                        return
                
                # 到这里说明 converged 或 next_step 为 True
                converged = self.converged
                next_step = self.next_step
                current_round = self.current_round
                expected = self.expected_clients
                submitted = len(self.client_parameters.get(current_round, {}))
                
                # 检查是否达到最大轮次（训练结束）
                # 注意：current_round 是 0-indexed，当 current_round == max_rounds 时表示已完成所有轮次
                # 使用 >= 而不是 +1 >=，避免在进入最后一轮之前就提前返回结束信号
                if current_round >= self.max_rounds:
                    code = 300
                    message = f"达到最大轮次 ({self.max_rounds})，训练结束"
                elif converged:
                    code = 300
                    message = "训练已收敛"
                elif next_step:
                    code = 200
                    message = "可以开始训练"
                    self.count += 1
                    if self.count >= expected:
                        self.next_step = False
                        self.count = 0
            
            # 构建并推送状态响应（在锁外进行 yield）
            response = federation_pb2.TrainingStatusResponse(
                code=code,
                message=message,
                registered_clients=len(self.clients),  # 返回真实注册数
                total_clients=expected,
                submitted_clients=submitted
            )
            
            try:
                yield response
            except Exception as e:
                logger.warning(f"[Server] 客户端 {client_id} 订阅中断: {e}")
                break
            
            # 推送完成后退出（每次订阅只等待一个状态变化）
            logger.debug(f"[Server] 客户端 {client_id} 状态订阅结束 (code={code})")
            break
        
        logger.debug(f"[Server] 客户端 {client_id} 订阅流关闭")

    def SubmitUpdate(self, request, context):
        """
        统一的更新提交入口。
        将请求直接转发给当前加载的聚合策略进行处理。
        """
        receive_start = time.time()
        
        client_id = request.client_id
        current_round = request.round
        request_size = request.ByteSize()

        if current_round not in self.communication_costs:
            self.communication_costs[current_round] = {}
        self.communication_costs[current_round][client_id] = request_size
        
        # 记录客户端延迟指标（统一格式，用于画图）
        if request.HasField('latency_metrics'):
            lm = request.latency_metrics
            logger.info(f"[Round {current_round+1}][Client {client_id}][LATENCY] training={lm.training_time:.4f}s, encryption={lm.encryption_time:.4f}s")
            logger.info(f"[Round {current_round+1}][Client {client_id}][PAYLOAD] upload={lm.payload_size_bytes/1024/1024:.2f} MB")
            logger.info(f"[Round {current_round+1}][Client {client_id}][RESOURCE] peak_memory={lm.peak_memory_mb:.2f} MB, cpu={lm.cpu_percent:.1f}%")
        
        logger.info(f"[Round {current_round+1}] 收到客户端 {client_id} 更新，大小: {request_size / 1024:.2f} KB")

        return self.aggregation_strategy.aggregate(request, context)

    def SubmitUpdateHeStream(self, request_iterator, context):
        """
        HE模式专用的流式更新入口。
        将请求流直接转发给当前加载的聚合策略进行处理。
        """
        # --- 通信开销统计 Start ---
        total_size = 0
        client_id = None
        current_round = None

        # 使用一个生成器表达式来包装原始迭代器，以便在迭代时计算大小
        def size_tracking_iterator(iterator):
            nonlocal total_size, client_id, current_round
            first_chunk = True
            for chunk in iterator:
                chunk_size = chunk.ByteSize()
                total_size += chunk_size
                if first_chunk:
                    client_id = chunk.client_id
                    current_round = chunk.round
                    first_chunk = False
                yield chunk

        tracked_iterator = size_tracking_iterator(request_iterator)
        # --- 通信开销统计 End ---

        if self.privacy_mode != 'he':
            logger.error("[Server] 非 HE 模式调用了 SubmitUpdateHeStream")
            return federation_pb2.ServerUpdate(code=400, message="此接口仅在HE模式下可用")
        
        # 将 *新的、带追踪的* 请求流和上下文直接传递给策略进行处理
        response = self.aggregation_strategy.aggregate_stream(tracked_iterator, context)

        # 在聚合完成后，记录总大小
        if client_id and current_round is not None:
            if current_round not in self.communication_costs:
                self.communication_costs[current_round] = {}
            self.communication_costs[current_round][client_id] = total_size
            logger.info(f"[Round {current_round+1}] 收到客户端 {client_id} HE 流式更新，大小: {total_size / 1024:.2f} KB")

        return response

    def GetGlobalModel(self, request, context):
        """提供当前全局模型参数"""
        client_id = request.client_id
        round_num = request.round
        model_parameters = self.global_model.get_parameters()
        logger.debug(f"[Round {round_num+1}] 向客户端 {client_id} 提供全局模型")
        
        model_params = federation_pb2.ModelParameters(
            parameters=serialize_parameters(model_parameters)
        )

        return model_params

    def process_round_completion(self, round_num):
        """处理轮次完成，聚合参数并更新全局模型"""
        try:
            with self.status_condition:  # 使用 Condition 替代 Lock
                logger.info(f"[Round {round_num+1}] 所有客户端更新已收集，开始聚合")
                
                # --- 系统资源监控 Start ---
                process = psutil.Process()
                start_cpu_time = process.cpu_times().user + process.cpu_times().system
                start_memory = process.memory_info().rss
                cpu_percent_start = process.cpu_percent()
                aggregation_start = time.time()
                # --- 系统资源监控 End ---

                aggregated_params = self.aggregation_strategy.aggregate_parameters(round_num)
                
                # --- 系统资源监控 End & Log ---
                aggregation_time = time.time() - aggregation_start
                end_cpu_time = process.cpu_times().user + process.cpu_times().system
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                cpu_percent = process.cpu_percent()
                cpu_time_used = end_cpu_time - start_cpu_time
                
                # 统一格式的资源日志
                logger.info(f"[Round {round_num+1}][Server][RESOURCE] peak_memory={peak_memory:.2f} MB, cpu={cpu_percent:.1f}%, cpu_time={cpu_time_used:.4f}s")

                self.global_model.set_parameters(aggregated_params)
                logger.info(f"[Round {round_num+1}] 全局模型更新完成")
                
                self.evaluate(round_num) 

                if self.converged or self.current_round + 1 == self.max_rounds:
                    # 标记为已收敛，确保客户端能正确收到结束信号
                    self.converged = True
                    self.next_step = True
                    # 唤醒所有等待状态更新的客户端（训练结束或收敛）
                    self.status_condition.notify_all()
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    logger.info(f"[Server] 训练结束，总耗时: {elapsed:.2f}s")

                    # --- 保存通信开销 ---
                    try:
                        costs_df = pd.DataFrame(self.communication_costs).sort_index()
                        costs_df.index.name = "Round"
                        costs_df = costs_df.reindex(sorted(costs_df.columns), axis=1) 
                        # 计算每轮总和与平均值
                        costs_df['Total_Bytes'] = costs_df.sum(axis=1)
                        costs_df['Total_KB'] = costs_df['Total_Bytes'] / 1024
                        # 计算所有轮次的总和
                        total_row = costs_df.sum().to_frame().T
                        total_row.index = ['Total']
                        costs_df = pd.concat([costs_df, total_row])
                        logger.info("[Server] 通信开销汇总 (KB):\n" + costs_df[['Total_KB']].to_string())
                    except Exception as e:
                        logger.error(f"[Server] 计算通信开销失败: {e}")
                    # --- 保存通信开销 End ---

                    # 创建并打印评估指标表格
                    eval_results = {
                        "Round": [i + 1 for i in range(len(self.rs_test_acc))],
                        "Accuracy": [f"{acc:.4f}" for acc in self.rs_test_acc],
                        "AUC": [f"{auc:.4f}" for auc in self.rs_auc],
                        "Loss": [f"{loss:.4f}" for loss in self.rs_train_loss]
                    }
                    df = pd.DataFrame(eval_results).set_index("Round")
                    logger.info("[Server] 评估指标汇总:\n" + df.to_string())

                    prefix = f"{self.privacy_mode}_"
                                        # plot_global_convergence_curve(self.rs_test_acc, self.rs_train_loss, self.rs_auc, prefix=prefix)
                else:
                    self.current_round += 1
                    self.next_step = True
                    # 唤醒所有等待状态更新的客户端（进入下一轮）
                    self.status_condition.notify_all()
                    logger.info(f"[Round {round_num+1}] 聚合完成，进入 Round {self.current_round+1}")
        except Exception as e:
            logger.error(f"[Server] 处理 Round {round_num} 出错: {e}", exc_info=True)

    def evaluate(self, round_num):
        """
        评估全局模型性能。
        
        优先使用全局测试集进行评估（更准确），
        如果全局测试集不可用，则使用客户端聚合指标。
        """
        # 优先使用全局测试集评估
        if self.global_test_loader is not None:
            accuracy, auc, test_num = self.evaluate_global_model(round_num)
            if accuracy is not None:
                self.rs_test_acc.append(accuracy)
                self.rs_auc.append(auc)
                # 对于全局测试集评估，loss 需要单独计算或使用客户端聚合
                self.aggregation_strategy.evaluate_metrics(round_num, skip_acc_auc=True)
        else:
            # 回退到使用客户端聚合指标
            self.aggregation_strategy.evaluate_metrics(round_num)

        # 输出每轮准确率日志（用于收敛曲线）
        if len(self.rs_test_acc) > 0:
            current_acc = self.rs_test_acc[-1]
            current_auc = self.rs_auc[-1] if len(self.rs_auc) > 0 else 0
            current_loss = self.rs_train_loss[-1] if len(self.rs_train_loss) > 0 else 0
            logger.info(f"[CONVERGENCE] round={round_num+1} accuracy={current_acc:.4f} auc={current_auc:.4f} loss={current_loss:.4f}")

        # [已禁用] 早停检测 - 统一使用 max_rounds=40 轮固定训练
        # if len(self.rs_test_acc) >= self.converge_window:
        #     recent_accs = self.rs_test_acc[-(self.converge_window):]
        #     acc_delta = max(recent_accs) - min(recent_accs)
        #     if acc_delta < self.acc_delta_threshold:
        #         self.converged = True
        #         logger.info(f"[Round {round_num+1}] 训练收敛: Acc 变化 ({acc_delta:.6f}) < 阈值 ({self.acc_delta_threshold})")


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config['grpc']['max_workers']),
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
            # keepalive 配置，防止长时间传输时连接超时
            ('grpc.keepalive_time_ms', 30000),                    # 每30秒发送一次keepalive ping
            ('grpc.keepalive_timeout_ms', 60000),                 # 等待60秒响应
            ('grpc.keepalive_permit_without_calls', True),        # 即使没有活跃调用也发送keepalive
            ('grpc.http2.max_pings_without_data', 0),             # 允许无数据时发送ping
            ('grpc.http2.min_ping_interval_without_data_ms', 10000),  # 最小ping间隔10秒
        ]
    )
    federation_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    
    port = config['grpc']['server_port']
    # TLS证书加载逻辑保持不变，因为所有模式都受益于传输层安全
    try:
        with open('/app/certs/server.key', 'rb') as f:
            private_key = f.read()
        with open('/app/certs/server.crt', 'rb') as f:
            certificate_chain = f.read()
        server_credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
        server.add_secure_port(f"0.0.0.0:{port}", server_credentials)
        logger.info(f"[Server] 启动 (SSL/TLS)，监听端口: {port}")
    except FileNotFoundError:
        server.add_insecure_port(f"[::]:{port}")
        logger.warning(f"[Server] 未找到证书，使用不安全模式启动，端口: {port}")

    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 