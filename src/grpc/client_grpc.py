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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
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
from src.models.models import get_model
from src.utils.config_utils import config
from src.strategies.client.none_strategy import NoneClientStrategy
from src.strategies.client.he_strategy import HeClientStrategy
from src.strategies.client.tee_strategy import TeeClientStrategy
from src.strategies.client.mpc_strategy import MpcClientStrategy
from src.strategies.client.sgx_strategy import SgxStrategy

logger = get_logger()
random.seed(config['base']['random_seed'])
np.random.seed(config['base']['random_seed'])
torch.manual_seed(config['base']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['base']['random_seed'])


class AugmentedDataset(Dataset):
    """支持数据增强的自定义数据集"""
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_train_transform(dataset_name):
    """根据数据集类型返回训练时的数据增强 transform"""
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        # CIFAR10: 32x32 彩色图像，使用随机裁剪和水平翻转
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif dataset_name == 'mnist':
        # MNIST: 28x28 灰度图像，使用轻度随机裁剪（手写数字不适合水平翻转）
        return transforms.Compose([
            transforms.RandomCrop(28, padding=2),
        ])
    else:
        return None


class FederatedLearningClient:
    def __init__(self, data=None):
        self.client_id = os.environ.get('CLIENT_ID') or str(uuid.uuid4())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dataset_name = config['data']['dataset']
        self.model_name = config['model']['name']
        
        self.model = get_model(self.model_name, self.dataset_name).to(self.device)
        
        self.num_classes = 10
        self.current_round = 0
        self.batch_size = config['training']['batch_size']
        self.server_host = config['grpc']['server_host']
        self.server_port = config['grpc']['server_port']
        self._init_data(data)
        self.logger = logger 

        self.stub = None
        self.channel = None
        self.privacy_mode = None
        self.strategy = None  
        self.continue_training = True

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 5e-4)
        )
        

    def setup_connection_and_register(self):
        """建立gRPC连接，并与服务器协商运行模式和安全材料。"""
        try:
            with open('/app/certs/ca.crt', 'rb') as f:
                ca_cert = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
            channel = grpc.secure_channel(
                f"{self.server_host}:{self.server_port}", credentials,
                options=[
                    ('grpc.ssl_target_name_override', 'server'), # Override for certificate validation
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024),
                    ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
                    # keepalive 配置，防止长时间传输时连接超时
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 60000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                ]
            )
            logger.info(f"客户端 {self.client_id} 初始化完成，数据集大小: {self.data_size}，使用安全通道(SSL/TLS)连接服务器{self.server_host}:{self.server_port}。")
        except FileNotFoundError:
            logger.warning(f"未找到CA证书，使用不安全通道连接服务器。")
            channel = grpc.insecure_channel(
                f"{self.server_host}:{self.server_port}",
                options=[
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024),
                    ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
                    # keepalive 配置，防止长时间传输时连接超时
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 60000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                ]
            )
        self.channel = channel
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)

        register_request = federation_pb2.ClientInfo(
            client_id=self.client_id,
            model_type=self.model_name,
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
                    logger.warning(f"无法连接到服务器 (详情: {e.details()})，将在 {retry_interval} 秒后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"多次尝试后仍无法连接到服务器，放弃连接。详情: {e.details()}")
                    raise e
        
        self.privacy_mode = setup_response.privacy_mode
        logger.info(f"服务器运行模式为: {self.privacy_mode.upper()}")

        # 1. Create the strategy based on the mode
        self.strategy = self._create_strategy(setup_response)
        
        # 2. Call the setup method on the strategy with the full response
        # self.strategy.setup(setup_response) # This line is now handled in _create_strategy

        # 3. Deserialize the initial model
        initial_parameters = deserialize_parameters(setup_response.initial_model.parameters)
        self.model.set_parameters(initial_parameters)
        logger.info(f"客户端{self.client_id}已设置初始模型参数。")

    def _create_strategy(self, setup_response):
        """根据服务器响应创建并返回相应的客户端策略实例。"""
        if self.privacy_mode == 'none':
            return NoneClientStrategy(self)
        elif self.privacy_mode == 'he':
            # 保持现有HE策略的加载方式不变
            return HeClientStrategy(self, setup_response.he_public_key)
        elif self.privacy_mode == 'tee':
            # 保持现有TEE策略的加载方式不变
            return TeeClientStrategy(self, setup_response.tee_attestation_report, setup_response.tee_public_key)
        elif self.privacy_mode == 'mpc':
            return MpcClientStrategy(self)
        elif self.privacy_mode == 'sgx':
            # SGX 策略需要先实例化，再用服务器返回的数据进行 setup
            sgx_strategy = SgxStrategy(self)
            sgx_strategy.setup(setup_response)
            return sgx_strategy
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
                # 获取训练时的数据增强 transform
                train_transform = get_train_transform(self.dataset_name)
                train_dataset = AugmentedDataset(X_train, y_train, transform=train_transform)
                self.train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
                self.data_size = len(train_dataset)
                logger.debug(f"数据集划分完成 - 训练集: {X_train.shape}，数据增强: {train_transform is not None}")
            else:
                self.train_data = None
                self.data_size = 0
                logger.warning("未提供训练集数据")
            if X_test is not None and y_test is not None:
                # 测试集不使用数据增强
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
        
        # 记录数据大小
        payload_size = update_request.ByteSize()
        logger.info(f"{log_prefix} 客户端 {self.client_id} 开始传输更新数据，大小: {payload_size / 1024 / 1024:.2f} MB")
        
        import time as _time
        start_time = _time.time()
        
        for attempt in range(max_retries):
            try:
                server_response = self.stub.SubmitUpdate(update_request)
                elapsed = _time.time() - start_time
                speed = payload_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
                logger.info(f"{log_prefix} 客户端 {self.client_id} 传输完成，耗时 {elapsed:.2f}s，速度 {speed:.2f} MB/s")
                return server_response  # 成功则返回响应
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"{log_prefix} 提交更新失败 (服务器不可达)，将在 {retry_interval} 秒后重试...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"{log_prefix} 多次尝试后仍无法提交更新，训练终止。")
                    raise e  # 将异常重新抛出，由上层处理
        
        raise RuntimeError(f"{log_prefix} 多次尝试后仍无法提交更新。")

    def _submit_update_stream_with_retry(self, update_generator):
        """带重试逻辑的流式提交更新。"""
        log_prefix = f"[HE Stream] [轮次 {self.current_round + 1}]"
        try:
            server_response = self.stub.SubmitUpdateHeStream(update_generator())
            if server_response.code == 200:
                logger.info(f"{log_prefix} 客户端 {self.client_id} 已成功提交流式更新。")
                return True
            else:
                logger.error(f"{log_prefix} 提交流式更新失败，服务器返回错误: code={server_response.code}, message='{server_response.message}'")
                return False
        except grpc._channel._InactiveRpcError as e:
            logger.error(f"{log_prefix} 提交流式更新时发生gRPC连接错误: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix} 提交流式更新时发生未知错误: {str(e)}", exc_info=True)
            return False

    def _wait_for_training_status(self, wait_for_code=200):
        """
        使用服务端流订阅训练状态，替代轮询机制。
        
        Args:
            wait_for_code: 期望的状态码，200表示可以继续训练，默认等待200
        
        Returns:
            status_response: 最终收到的状态响应
        """
        status_request = federation_pb2.ClientInfo(client_id=self.client_id)
        
        try:
            # 使用流式 RPC 订阅状态
            for status_response in self.stub.SubscribeTrainingStatus(status_request):
                if status_response.code == 200:
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}收到训练许可")
                    return status_response
                elif status_response.code == 300:
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}检测到服务器收敛信号")
                    return status_response
                elif status_response.code == 100:
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待中 (进度: {status_response.submitted_clients}/{status_response.total_clients})")
                    # 继续等待下一个流式响应，无需 sleep
                else:
                    logger.warning(f"[Round {self.current_round+1}] 客户端{self.client_id}收到未知状态码 {status_response.code}")
        except grpc.RpcError as e:
            logger.error(f"订阅训练状态时发生 gRPC 错误: {e.code()} - {e.details()}")
            # 发生错误时返回一个表示需要重试的响应
            return federation_pb2.TrainingStatusResponse(code=500, message="gRPC连接错误")
        
        # 流正常结束但未收到预期响应
        return federation_pb2.TrainingStatusResponse(code=100, message="流结束")

    def participate_in_training(self):
        """参与联邦学习训练"""
        self.setup_connection_and_register()
        
        # 使用流式订阅等待所有客户端注册完成
        logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待所有客户端注册...")
        status_response = self._wait_for_training_status()
        if status_response.code == 300:
            logger.info("训练已收敛，客户端退出。")
            return
        elif status_response.code == 500:
            logger.error("注册阶段发生连接错误，客户端退出。")
            return

        while self.continue_training:
            logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}开始训练...")
            self.train(epochs=config['training']['epochs'])
            
            metrics_data = self.get_metrics()
            
            model_parameters = self.model.get_parameters()
            
            if self.privacy_mode == 'he':
                # HE模式使用新的流式接口
                update_generator = self.strategy.prepare_stream_update_request(self.current_round, model_parameters, metrics_data)
                success = self._submit_update_stream_with_retry(update_generator)
                if not success:
                    logger.error(f"[Round {self.current_round+1}] HE流式提交失败，终止训练。")
                    self.continue_training = False
                    # 这里可以添加更复杂的错误处理，例如重试整个轮次
                    continue # 直接进入下一轮的循环检查（实际上会因为continue_training=False而退出）
            else:
                # 其他模式使用原有的接口
                update_request = self.strategy.prepare_update_request(self.current_round, model_parameters, metrics_data)
                self._submit_update_with_retry(update_request)

            logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待全局模型更新...")

            # 使用流式订阅等待服务器聚合完成（替代原来的轮询循环）
            status_response = self._wait_for_training_status()
            
            if status_response.code == 300:
                logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}检测到服务器收敛信号，终止训练。")
                self.continue_training = False
            elif status_response.code == 500:
                logger.error(f"[Round {self.current_round+1}] 连接错误，终止训练。")
                self.continue_training = False
                
            global_model_request = federation_pb2.GetModelRequest(client_id=self.client_id, round=self.current_round)
            global_model_response = self.stub.GetGlobalModel(global_model_request)
            global_params = deserialize_parameters(global_model_response.parameters)
            self.model.set_parameters(global_params)
            logger.info(f"[Round {self.current_round+1}] 成功更新全局模型。")
            
            self.current_round += 1
            # 注意：不再需要检查 max_rounds，服务器会在达到时发送 code=300 通知客户端

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
    dataset_name = config['data']['dataset']
    train_path = f"/app/data/{dataset_name}_train.npz"
    test_path = f"/app/data/{dataset_name}_test.npz"
    
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        X_train = train_data["X_train"]
        y_train = train_data["y_train"]
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        logger.info(f"成功加载客户端训练集: {train_path}, 形状: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"成功加载客户端测试集: {test_path}, 形状: X_test={X_test.shape}, y_test={y_test.shape}")
        return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    except FileNotFoundError as e:
        logger.error(f"无法找到数据文件: {e}")
        return None


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