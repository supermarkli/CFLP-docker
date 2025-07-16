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
from src.models.models import FedAvgCNN
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
        self.global_model = FedAvgCNN().to(self.device)
        self.clients = {}
        self.current_round = 0
        self.lock = threading.Lock()
        self.client_parameters = defaultdict(dict)
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
        with self.lock:
            if self.converged:
                code = 300
                message = "训练已收敛，提前终止"
                return federation_pb2.TrainingStatusResponse(
                    code=code,
                    message=message,
                    registered_clients=self.count,
                    total_clients=self.expected_clients,
                    submitted_clients=len(self.client_parameters.get(self.current_round, {}))
                )
            logger.info(f"[Round {self.current_round+1}] 客户端 {client_id} 检查训练状态，当前next_step={self.next_step}, count={self.count}")
            if self.next_step:
                code = 200
                message = "可以开始训练"
                self.count += 1
                logger.info(f"[Round {self.current_round+1}] 客户端 {client_id} 获得训练许可，count增加到 {self.count}/{self.expected_clients}")
                if self.count >= self.expected_clients:
                    self.next_step = False
                    self.count = 0
                    logger.info(f"[Round {self.current_round+1}] 所有客户端已获得训练许可，重置next_step={self.next_step}, count={self.count}")
            else:
                code = 100
                message = f"[Round {self.current_round+1}] 等待其他客户端, 当前{self.count}/{self.expected_clients}个客户端"

            return federation_pb2.TrainingStatusResponse(
                code=code,
                message=message,
                registered_clients=self.count,
                total_clients=self.expected_clients,
                submitted_clients=len(self.client_parameters.get(self.current_round, {}))
            )

    def SubmitUpdate(self, request, context):
        """
        统一的更新提交入口。
        将请求直接转发给当前加载的聚合策略进行处理。
        """
        return self.aggregation_strategy.aggregate(request, context)

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
                
                aggregated_params = self.aggregation_strategy.aggregate_parameters(round_num)
                
                self.global_model.set_parameters(aggregated_params)
                logger.info(f"[Round {round_num+1}] 全局模型参数更新完成。")
                
                self.evaluate(round_num) 

                if self.converged or self.current_round + 1 == self.max_rounds:
                    self.next_step = True
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    logger.info(f"训练结束。总耗时: {elapsed:.2f} 秒")

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
        server.add_secure_port(f"[::]:{port}", server_credentials)
        logger.info(f"联邦学习安全服务器正在启动，监听端口: {port}")
    except FileNotFoundError:
        server.add_insecure_port(f"[::]:{port}")
        logger.warning(f"未找到证书文件，使用不安全模式启动服务器于端口: {port}")

    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 