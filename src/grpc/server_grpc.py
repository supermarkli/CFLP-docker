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

from phe import paillier
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
from src.utils.draw import plot_global_convergence_curve
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
        logger.info(f"服务器配置: Dataset={self.dataset_name}, Model={self.model_name}")

        self.global_model = get_model(self.model_name, self.dataset_name).to(self.device)
        self.clients = {}
        self.current_round = 0
        self.lock = threading.Lock()
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
        
        self.aggregation_strategy = self._create_aggregation_strategy()
        if not self.aggregation_strategy:
            raise ValueError(f"不支持的隐私模式或初始化策略失败: {self.privacy_mode}")

        logger.info(f"服务器初始化完成 (模式: {self.privacy_mode})，等待 {self.expected_clients} 个客户端注册...")

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
        logger.info(f"接收到客户端 {client_id} 的注册请求 (模型: {request.model_type}, 数据量: {request.data_size})")

        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = ClientState(
                    client_id=client_id,
                    model_type=request.model_type,
                    data_size=request.data_size
                )
                logger.info(f"客户端 {client_id} 注册成功。当前 {len(self.clients)}/{self.expected_clients} 个客户端。")
 
            response = self.aggregation_strategy.prepare_setup_response(request)

            if len(self.clients) >= self.expected_clients:
                self.next_step = True
                logger.info(f"所有客户端已注册，设置 next_step=True，准备开始训练。")
                if self.start_time is None:
                    self.start_time = time.time()
                    logger.info("联邦学习流程计时开始")

            return response

    def CheckTrainingStatus(self, request, context):
        client_id = request.client_id
        
        # 最小化锁持有时间，快速读取和更新状态
        with self.lock:
            converged = self.converged
            next_step = self.next_step
            current_round = self.current_round
            count = self.count
            expected = self.expected_clients
            submitted = len(self.client_parameters.get(current_round, {}))
            
            if converged:
                code = 300
                message = "训练已收敛，提前终止"
            elif next_step:
                code = 200
                message = "可以开始训练"
                self.count += 1
                count = self.count  # 更新本地变量用于日志
                if self.count >= expected:
                    self.next_step = False
                    self.count = 0
            else:
                code = 100
                message = f"[Round {current_round+1}] 等待其他客户端"
        
        # 在锁外进行日志记录，避免长时间持有锁
        if code == 200:
            logger.info(f"[Round {current_round+1}] 客户端 {client_id} 获得训练许可 ({count}/{expected})")
        elif code == 100:
            logger.debug(f"[Round {current_round+1}] 客户端 {client_id} 等待中 (next_step={next_step}, submitted={submitted}/{expected})")

        return federation_pb2.TrainingStatusResponse(
            code=code,
            message=message,
            registered_clients=count,
            total_clients=expected,
            submitted_clients=submitted
        )

    def SubmitUpdate(self, request, context):
        """
        统一的更新提交入口。
        将请求直接转发给当前加载的聚合策略进行处理。
        """
        # 计算并记录通信开销
        client_id = request.client_id
        current_round = request.round
        request_size = request.ByteSize()

        if current_round not in self.communication_costs:
            self.communication_costs[current_round] = {}
        self.communication_costs[current_round][client_id] = request_size
        
        logger.info(f"[Round {current_round+1}] 收到来自客户端 {client_id} 的更新，数据大小: {request_size / 1024:.2f} KB")

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
            logger.error("非HE模式下调用了SubmitUpdateHeStream")
            return federation_pb2.ServerUpdate(code=400, message="此接口仅在HE模式下可用。")
        
        # 将 *新的、带追踪的* 请求流和上下文直接传递给策略进行处理
        response = self.aggregation_strategy.aggregate_stream(tracked_iterator, context)

        # 在聚合完成后，记录总大小
        if client_id and current_round is not None:
            if current_round not in self.communication_costs:
                self.communication_costs[current_round] = {}
            self.communication_costs[current_round][client_id] = total_size
            logger.info(f"[Round {current_round+1}] 收到来自客户端 {client_id} 的流式更新，数据总大小: {total_size / 1024:.2f} KB")

        return response

    def GetGlobalModel(self, request, context):
        """提供当前全局模型参数"""
        client_id = request.client_id
        round_num = request.round
        model_parameters = self.global_model.get_parameters()
        logger.info(f"向客户端 {client_id} 提供第{round_num+1}轮全局模型")
        
        model_params = federation_pb2.ModelParameters(
            parameters=serialize_parameters(model_parameters)
        )

        return model_params

    def process_round_completion(self, round_num):
        """处理轮次完成，聚合参数并更新全局模型"""
        try:
            with self.lock:
                logger.info(f"[Round {round_num+1}] 所有客户端参数已收集完毕，开始聚合。")
                
                # --- 通用资源监控 Start ---
                process = psutil.Process()
                start_cpu_time = process.cpu_times().user + process.cpu_times().system
                start_memory = process.memory_info().rss
                # --- 通用资源监控 End ---

                aggregated_params = self.aggregation_strategy.aggregate_parameters(round_num)
                
                # --- 通用资源监控 End & Log ---
                end_cpu_time = process.cpu_times().user + process.cpu_times().system
                current_memory = process.memory_info().rss
                
                cpu_time_used = end_cpu_time - start_cpu_time
                memory_usage = current_memory
                
                # 对于非 SGX 模式，记录主进程的资源消耗
                if self.privacy_mode != 'sgx':
                    logger.info(f"[Round {round_num+1}] Server Aggregation Resources - CPU Time: {cpu_time_used:.4f}s, Memory Usage: {memory_usage / 1024 / 1024:.2f} MB")
                else:
                    # SGX 模式下，主要计算在 Enclave 中，主进程开销较小，但记录下来也无妨，作为对比
                    logger.info(f"[Round {round_num+1}] Server Process (Host) Resources - CPU Time: {cpu_time_used:.4f}s, Memory Usage: {memory_usage / 1024 / 1024:.2f} MB (See previous log for Enclave resources)")
                # --- 通用资源监控 End ---

                self.global_model.set_parameters(aggregated_params)
                logger.info(f"[Round {round_num+1}] 全局模型参数更新完成。")
                
                self.evaluate(round_num) 

                if self.converged or self.current_round + 1 == self.max_rounds:
                    self.next_step = True
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    logger.info(f"训练结束。总耗时: {elapsed:.2f} 秒")

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
                        logger.info("通信开销 (KB) 汇总:\n" + costs_df[['Total_KB']].to_string())
                    except Exception as e:
                        logger.error(f"计算通信开销失败: {e}")
                    # --- 保存通信开销 End ---

                    # 创建并打印评估指标表格
                    eval_results = {
                        "Round": [i + 1 for i in range(len(self.rs_test_acc))],
                        "Accuracy": [f"{acc:.4f}" for acc in self.rs_test_acc],
                        "AUC": [f"{auc:.4f}" for auc in self.rs_auc],
                        "Loss": [f"{loss:.4f}" for loss in self.rs_train_loss]
                    }
                    df = pd.DataFrame(eval_results).set_index("Round")
                    logger.info("全局模型评估指标汇总:\n" + df.to_string())

                    prefix = f"{self.privacy_mode}_"
                    plot_global_convergence_curve(self.rs_test_acc, self.rs_train_loss, self.rs_auc, prefix=prefix)
                else:
                    self.current_round += 1
                    self.next_step = True
                    logger.info(f"第 {round_num+1} 轮聚合完成，进入第 {self.current_round+1} 轮。")
        except Exception as e:
            logger.error(f"处理轮次 {round_num} 完成时出错: {e}", exc_info=True)

    def evaluate(self, round_num):
        """评估所有客户端的平均指标，并检查收敛"""
        self.aggregation_strategy.evaluate_metrics(round_num)

        # if len(self.rs_test_acc) >= self.converge_window:
        #     recent_accs = self.rs_test_acc[-(self.converge_window):]
        #     acc_delta = max(recent_accs) - min(recent_accs)
        #     if acc_delta < self.acc_delta_threshold:
        #         self.converged = True
        #         logger.info(f"[Round {round_num}] 训练已收敛，准确率变化 ({acc_delta:.6f}) 小于阈值 ({self.acc_delta_threshold})。")


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config['grpc']['max_workers']),
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
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
        logger.info(f"联邦学习安全服务器正在启动，监听端口: {port}")
    except FileNotFoundError:
        server.add_insecure_port(f"[::]:{port}")
        logger.warning(f"未找到证书文件，使用不安全模式启动服务器于端口: {port}")

    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 