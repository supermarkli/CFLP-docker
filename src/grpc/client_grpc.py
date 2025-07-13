import grpc
import os
import sys
import uuid
import pandas as pd
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from tqdm import tqdm
import random
import json
import hashlib
import gc

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging_config import get_logger
from src.grpc.generated import federation_pb2
from src.grpc.generated import federation_pb2_grpc
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.models.models import FedAvgCNN
from src.utils.config_utils import config
from src.strategies.client.none_strategy import NoneClientStrategy
from src.strategies.client.he_strategy import HeClientStrategy
from src.strategies.client.tee_strategy import TeeClientStrategy
from src.strategies.client.mpc_strategy import MpcClientStrategy

logger = get_logger()
random.seed(config['base']['random_seed'])
np.random.seed(config['base']['random_seed'])
torch.manual_seed(config['base']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['base']['random_seed'])

class FederatedLearningClient:
    def __init__(self, data=None):
        self.client_id = os.environ.get('CLIENT_ID') or str(uuid.uuid4())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FedAvgCNN().to(self.device)
        self.num_classes = 10
        self.current_round = 0
        self.batch_size = config['training']['batch_size']
        self.server_host = config['grpc']['server_host']
        self.server_port = config['grpc']['server_port']
        self._init_data(data)

        self.stub = None
        self.channel = None
        self.privacy_mode = None
        self.strategy = None  
        self.continue_training = True

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        

    def setup_connection_and_register(self):
        """建立gRPC连接，并与服务器协商运行模式和安全材料。"""
        try:
            with open('/app/certs/ca.crt', 'rb') as f:
                ca_cert = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
            channel = grpc.secure_channel(
                f"{self.server_host}:{self.server_port}", credentials,
                options=[('grpc.max_send_message_length', 500 * 1024 * 1024),
                         ('grpc.max_receive_message_length', 500 * 1024 * 1024)]
            )
            logger.info(f"客户端 {self.client_id} 初始化完成，数据集大小: {self.data_size}，使用安全通道(SSL/TLS)连接服务器。")
        except FileNotFoundError:
            logger.warning(f"未找到CA证书，使用不安全通道连接服务器。")
            channel = grpc.insecure_channel(
                f"{self.server_host}:{self.server_port}",
                options=[('grpc.max_send_message_length', 500 * 1024 * 1024),
                         ('grpc.max_receive_message_length', 500 * 1024 * 1024)]
            )
        self.channel = channel
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)

        register_request = federation_pb2.ClientInfo(
            client_id=self.client_id,
            model_type="CNN",
            data_size=self.data_size
        )

        max_retries = 5
        retry_interval = 3
        for attempt in range(max_retries):
            try:
                setup_response = self.stub.RegisterAndSetup(register_request)
                logger.info(f"客户端{self.client_id}注册成功。")
                break 
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"无法连接到服务器，将在 {retry_interval} 秒后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(retry_interval)
                else:
                    logger.error("多次尝试后仍无法连接到服务器，放弃连接。")
                    raise e
        
        self.privacy_mode = setup_response.privacy_mode
        logger.info(f"服务器运行模式为: {self.privacy_mode.upper()}")

        self.strategy = self._create_strategy(setup_response)
        if not self.strategy:
            raise ValueError(f"不支持的隐私模式: {self.privacy_mode}")

        initial_params = deserialize_parameters(setup_response.initial_model.parameters)
        self.model.set_parameters(initial_params)
        logger.info(f"客户端{self.client_id}已设置初始模型参数。")

    def _create_strategy(self, setup_response):
        """根据服务器响应创建并返回相应的客户端策略实例。"""
        if self.privacy_mode == 'none':
            return NoneClientStrategy(self)
        elif self.privacy_mode == 'he':
            # 策略类自己处理反序列化
            return HeClientStrategy(self, setup_response.he_public_key)
        elif self.privacy_mode == 'tee':
            # 策略类自己处理验证和反序列化
            return TeeClientStrategy(self, setup_response.tee_attestation_report, setup_response.tee_public_key)
        elif self.privacy_mode == 'mpc':
            return MpcClientStrategy(self)
        else:
            logger.error(f"接收到未知的隐私模式: {self.privacy_mode}")
            return None

    def _init_data(self, data):
        """初始化训练和测试数据"""
        if data is not None:
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            X_test = data.get('X_test')
            y_test = data.get('y_test')
            if X_train is not None and y_train is not None:
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                self.train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
                self.data_size = len(X_train_tensor)
                logger.debug(f"数据集划分完成 - 训练集: {X_train_tensor.shape}")
            else:
                self.train_data = None
                self.data_size = 0
                logger.warning("未提供训练集数据")
            if X_test is not None and y_test is not None:
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                self.test_data = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
                logger.debug(f"数据集划分完成 - 测试集: {X_test_tensor.shape}")
            else:
                self.test_data = None
                logger.warning("未提供测试集数据")
        else:
            logger.warning("未提供数据，训练集和测试集将为空")
            self.train_data = None
            self.test_data = None
            self.data_size = 0

    def train(self, epochs=1):
        """本地训练模型"""
        if self.train_data is None:
            logger.warning(f"客户端 {self.client_id}: 没有可用的训练数据")
            return None
        try:
            self.model.train()
            for epoch in range(epochs):
                for x, y in self.train_data:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        except Exception as e:
            logger.error(f"本地训练失败: {str(e)}")
            raise

    def train_metrics(self):
        self.model.eval()
        train_num = 0  # 总训练样本数
        losses = 0     # 累计损失

        with torch.no_grad():
            for x, y in self.train_data:

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]  

        return losses, train_num
    
    def test_metrics(self):
        self.model.eval()
        test_acc = 0  # 正确预测的样本数
        test_num = 0  # 总测试样本数
        y_prob = []   # 存储所有样本的预测概率
        y_true = []   # 存储所有样本的真实标签（二值化后）
        
        with torch.no_grad():
            for x, y in self.test_data:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # 合并所有batch的预测概率和真实标签
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # 计算AUC（micro平均）
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def _submit_update_with_retry(self, update_request):
        """使用重试逻辑提交统一的更新请求。"""
        max_retries = config['grpc']['max_retries']
        retry_interval = config['grpc']['retry_interval']
        log_prefix = f"[{self.privacy_mode.upper()}] [轮次 {self.current_round + 1}]"
        
        for attempt in range(max_retries):
            try:
                server_response = self.stub.SubmitUpdate(update_request)
                return server_response  # 成功则返回响应
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"{log_prefix} 提交更新失败 (服务器不可达)，将在 {retry_interval} 秒后重试...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"{log_prefix} 多次尝试后仍无法提交更新，训练终止。")
                    raise e  # 将异常重新抛出，由上层处理
        
        raise RuntimeError(f"{log_prefix} 多次尝试后仍无法提交更新。")

    def participate_in_training(self):
        """参与联邦学习训练"""
        self.setup_connection_and_register()
        
        while True:
            status_request = federation_pb2.ClientInfo(client_id=self.client_id)
            status_response = self.stub.CheckTrainingStatus(status_request)

            if status_response.code == 100:
                logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待其他客户注册 (当前进度: {status_response.registered_clients}/{status_response.total_clients})")
                time.sleep(1)  
            else:
                break

        while self.continue_training:
            logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}开始训练...")
            self.train(epochs=config['training']['epochs'])
            
            metrics_data = self.get_metrics()
            
            update_request = self.strategy.prepare_update_request(self.current_round, self.model.get_parameters(), metrics_data)

            self._submit_update_with_retry(update_request)

            logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待全局模型更新...")

            while True:
                status_request = federation_pb2.ClientInfo(client_id=self.client_id)
                status_response = self.stub.CheckTrainingStatus(status_request)
                if status_response.code == 200:
                    break
                if status_response.code == 300:
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}检测到服务器收敛信号，终止训练。")
                    self.continue_training = False
                    break
                elif status_response.code == 100:
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待服务器聚合 (当前进度: {status_response.submitted_clients}/{status_response.total_clients})")
                    time.sleep(2)
                else:
                    logger.warning(f"[Round {self.current_round+1}] 客户端{self.client_id}收到未知状态码 {status_response.code}，等待中...")
                    time.sleep(2)
                
            global_model_request = federation_pb2.GetModelRequest(client_id=self.client_id, round=self.current_round)
            global_model_response = self.stub.GetGlobalModel(global_model_request)
            global_params = deserialize_parameters(global_model_response.parameters)
            self.model.set_parameters(global_params)
            logger.info(f"[Round {self.current_round+1}] 成功更新全局模型。")
            
            self.current_round += 1
            if self.current_round >= config['federation']['max_rounds']:
                logger.info("达到最大训练轮次，结束训练。")
                self.continue_training = False

        logger.info("客户端训练流程结束。")

    def get_metrics(self):
        """计算并返回所有相关指标的字典。"""
        self.model.eval()
        test_acc, test_num, auc = self.test_metrics()
        loss, train_num = self.train_metrics()

        logger.info(f"[Round {self.current_round+1}] 本地评估: Acc={test_acc/test_num if test_num>0 else 0:.4f}, AUC={auc:.4f}, Loss={loss:.4f}")

        return {
            'test_acc': test_acc,
            'test_num': test_num,
            'auc': auc,
            'loss': loss,
            'train_num': train_num
        }

    def __del__(self):
        if self.channel:
            self.channel.close()
            logger.info(f"客户端 {self.client_id}: gRPC 通道已关闭。")



def load_client_data():
    """加载客户端训练集和测试集数据"""
    train_path = "/app/data/mnist_train.npz"
    test_path = "/app/data/mnist_test.npz"
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    X_train = train_data["X_train"]
    y_train = train_data["y_train"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    logger.info(f"成功加载客户端训练集: {train_path}, 形状: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"成功加载客户端测试集: {test_path}, 形状: X_test={X_test.shape}, y_test={y_test.shape}")
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def main():
    client_data = load_client_data()
    if client_data:
        client = FederatedLearningClient(data=client_data)
        try:
            client.participate_in_training()
        except Exception as e:
            logger.error(f"训练过程中发生致命错误: {e}", exc_info=True)
    else:
        logger.error("无法加载数据，客户端无法启动。")

if __name__ == "__main__":
    main() 