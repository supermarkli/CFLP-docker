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
import psutil
import tracemalloc

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
        
        # GPU 信息日志（初始化时还没有 client_id，稍后输出）
        self._gpu_info = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self._gpu_info = f"🚀 使用 GPU 训练: {gpu_name} ({gpu_memory:.1f} GB)"
        else:
            self._gpu_info = "⚠️ 未检测到 GPU，使用 CPU 训练"
        
        self.dataset_name = config['data']['dataset']
        self.model_name = config['model']['name']
        
        self.model = get_model(self.model_name, self.dataset_name).to(self.device)
        
        self.num_classes = 10
        self.current_round = 0
        self.batch_size = config['training']['batch_size']
        self.server_host = config['grpc']['server_host']
        self.server_port = config['grpc']['server_port']
        self.logger = logger 

        self.stub = None
        self.channel = None
        self.privacy_mode = None
        self.strategy = None  
        self.continue_training = True

        self.loss = nn.CrossEntropyLoss()
        # 使用 SGD 优化器（与实验配置一致）
        lr = config['training']['learning_rate']
        momentum = config['training'].get('momentum', 0.9)
        weight_decay = config['training'].get('weight_decay', 0.0005)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        # 输出 GPU 和优化器信息
        logger.info(f"[Client {self.client_id}] {self._gpu_info}")
        logger.info(f"[Client {self.client_id}] 优化器: SGD (lr={lr}, momentum={momentum}, weight_decay={weight_decay})")
        
        # 使用 estimated_rounds 计算 T_max（与 CFLP_Revision 一致）
        estimated_rounds = config['training'].get('estimated_rounds', 200)
        local_epochs = config['training'].get('epochs', 3)
        T_max = estimated_rounds * local_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=T_max,
            eta_min=0.0  # 最小学习率为 0
        )
        logger.info(f"[Client {self.client_id}] 学习率调度: CosineAnnealingLR (T_max={T_max})")
        logger.info(f"[Client {self.client_id}] 数据集: {self.dataset_name}, 模型: {self.model_name}")
        
        # 初始化数据（数据增强日志在 _init_data 中输出）
        self._init_data(data)
        

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
            logger.info(f"[Client {self.client_id}] 初始化完成，数据量: {self.data_size}，使用 SSL/TLS 连接 {self.server_host}:{self.server_port}")
        except FileNotFoundError:
            logger.warning(f"[Client {self.client_id}] 未找到 CA 证书，使用不安全通道连接")
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
                logger.info(f"[Client {self.client_id}] 注册成功")
                break 
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"[Client {self.client_id}] 连接失败，{retry_interval}s 后重试 ({attempt+1}/{max_retries})")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"[Client {self.client_id}] 多次重试后仍无法连接服务器")
                    raise e
        
        self.privacy_mode = setup_response.privacy_mode
        logger.info(f"[Client {self.client_id}] 服务器模式: {self.privacy_mode.upper()}")

        # 1. Create the strategy based on the mode
        self.strategy = self._create_strategy(setup_response)
        
        # 2. Call the setup method on the strategy with the full response
        # self.strategy.setup(setup_response) # This line is now handled in _create_strategy

        # 3. Deserialize the initial model
        initial_parameters = deserialize_parameters(setup_response.initial_model.parameters)
        self.model.set_parameters(initial_parameters)
        logger.info(f"[Client {self.client_id}] 已设置初始模型参数")

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
            logger.error(f"[Client {self.client_id}] 未知的隐私模式: {self.privacy_mode}")
            return None

    def _init_data(self, data):
        """初始化训练和测试数据"""
        if data is not None:
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            X_test = data.get('X_test')
            y_test = data.get('y_test')
            if X_train is not None and y_train is not None:
                # 根据配置决定是否启用数据增强（与 CFLP_Revision 一致）
                use_augmentation = config['training'].get('use_augmentation', False)
                train_transform = get_train_transform(self.dataset_name) if use_augmentation else None
                train_dataset = AugmentedDataset(X_train, y_train, transform=train_transform)
                # 使用配置中的 num_workers 和 prefetch_factor
                num_workers = config['training'].get('num_workers', 0)  # 默认 0（单进程）
                prefetch_factor = config['training'].get('prefetch_factor', 2)
                self.train_data = DataLoader(
                    train_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    drop_last=True,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None
                )
                self.data_size = len(train_dataset)
                # 数据增强日志
                if use_augmentation and train_transform is not None:
                    logger.info(f"[Client {self.client_id}] 数据增强: 已启用")
                else:
                    logger.info(f"[Client {self.client_id}] 数据增强: 已禁用")
                logger.debug(f"[Client {self.client_id}] 数据划分完成 - 训练集: {X_train.shape}")
            else:
                self.train_data = None
                self.data_size = 0
                logger.warning(f"[Client {self.client_id}] 未提供训练集数据")
            if X_test is not None and y_test is not None:
                # 测试集不使用数据增强，使用 eval_batch_size
                eval_batch_size = config['training'].get('eval_batch_size', self.batch_size)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                num_workers = config['training'].get('num_workers', 0)
                prefetch_factor = config['training'].get('prefetch_factor', 2)
                self.test_data = DataLoader(
                    test_dataset, 
                    batch_size=eval_batch_size, 
                    shuffle=False, 
                    drop_last=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None
                )
                logger.debug(f"[Client {self.client_id}] 数据划分完成 - 验证集: {X_test_tensor.shape}")
            else:
                self.test_data = None
                logger.warning(f"[Client {self.client_id}] 未提供验证集数据")
        else:
            logger.warning(f"[Client {self.client_id}] 未提供数据，训练集和验证集为空")
            self.train_data = None
            self.test_data = None
            self.data_size = 0

    def train(self, epochs=1):
        """本地训练模型"""
        if self.train_data is None:
            logger.warning(f"[Client {self.client_id}] 没有可用的训练数据")
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
                # 每个 epoch 后更新学习率（CosineAnnealingLR）
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
        except Exception as e:
            logger.error(f"[Client {self.client_id}] 本地训练失败: {str(e)}")
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
        log_prefix = f"[Client {self.client_id}][Round {self.current_round + 1}]"
        
        # 记录数据大小
        payload_size = update_request.ByteSize()
        logger.info(f"{log_prefix} 开始上传更新，大小: {payload_size / 1024 / 1024:.2f} MB")
        
        import time as _time
        start_time = _time.time()
        
        for attempt in range(max_retries):
            try:
                server_response = self.stub.SubmitUpdate(update_request)
                elapsed = _time.time() - start_time
                speed = payload_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
                logger.info(f"{log_prefix} 上传完成，耗时 {elapsed:.2f}s，速度 {speed:.2f} MB/s")
                return server_response  # 成功则返回响应
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"{log_prefix} 上传失败，{retry_interval}s 后重试...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"{log_prefix} 多次重试后仍无法上传，训练终止")
                    raise e  # 将异常重新抛出，由上层处理
        
        raise RuntimeError(f"{log_prefix} 多次重试后仍无法上传")

    def _submit_update_stream_with_retry(self, update_generator):
        """带重试逻辑的流式提交更新。"""
        log_prefix = f"[Client {self.client_id}][Round {self.current_round + 1}]"
        try:
            server_response = self.stub.SubmitUpdateHeStream(update_generator())
            if server_response.code == 200:
                logger.info(f"{log_prefix} HE 流式上传成功")
                return True
            else:
                logger.error(f"{log_prefix} HE 流式上传失败: code={server_response.code}")
                return False
        except grpc._channel._InactiveRpcError as e:
            logger.error(f"{log_prefix} HE 流式上传 gRPC 错误: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix} HE 流式上传未知错误: {str(e)}", exc_info=True)
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
        
        log_prefix = f"[Client {self.client_id}][Round {self.current_round+1}]"
        try:
            # 使用流式 RPC 订阅状态
            for status_response in self.stub.SubscribeTrainingStatus(status_request):
                if status_response.code == 200:
                    logger.info(f"{log_prefix} 收到训练许可")
                    return status_response
                elif status_response.code == 300:
                    logger.info(f"{log_prefix} 检测到收敛信号")
                    return status_response
                elif status_response.code == 100:
                    logger.debug(f"{log_prefix} 等待中 ({status_response.submitted_clients}/{status_response.total_clients})")
                    # 继续等待下一个流式响应，无需 sleep
                else:
                    logger.warning(f"{log_prefix} 未知状态码: {status_response.code}")
        except grpc.RpcError as e:
            logger.error(f"{log_prefix} gRPC 错误: {e.code()} - {e.details()}")
            # 发生错误时返回一个表示需要重试的响应
            return federation_pb2.TrainingStatusResponse(code=500, message="gRPC连接错误")
        
        # 流正常结束但未收到预期响应
        return federation_pb2.TrainingStatusResponse(code=100, message="流结束")

    def participate_in_training(self):
        """参与联邦学习训练"""
        self.setup_connection_and_register()
        
        # 使用流式订阅等待所有客户端注册完成
        logger.info(f"[Client {self.client_id}] 等待所有客户端注册...")
        status_response = self._wait_for_training_status()
        if status_response.code == 300:
            logger.info(f"[Client {self.client_id}] 训练已收敛，退出")
            return
        elif status_response.code == 500:
            logger.error(f"[Client {self.client_id}] 注册阶段连接错误，退出")
            return

        # 获取进程对象用于资源监控
        process = psutil.Process()
        
        while self.continue_training:
            round_start_time = time.time()
            
            # 开始内存追踪
            tracemalloc.start()
            peak_memory_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent_start = process.cpu_percent()
            
            log_prefix = f"[Client {self.client_id}][Round {self.current_round+1}]"
            
            # === 阶段1: 本地训练 ===
            logger.info(f"{log_prefix} 开始训练...")
            train_start = time.time()
            self.train(epochs=config['training']['epochs'])
            train_time = time.time() - train_start
            
            metrics_data = self.get_metrics()
            local_acc = metrics_data['test_acc'] / metrics_data['test_num'] if metrics_data['test_num'] > 0 else 0
            
            model_parameters = self.model.get_parameters()
            payload_size_bytes = 0
            encrypt_time = 0
            upload_time = 0
            
            # 收集资源统计（在加密前获取，因为加密会消耗大量资源）
            peak_memory_mid = process.memory_info().rss / 1024 / 1024
            cpu_percent_mid = process.cpu_percent()
            
            if self.privacy_mode == 'he':
                # === 阶段2: 加密 (HE模式 - 流式) ===
                encrypt_start = time.time()
                # 收集延迟信息用于发送给服务端
                latency_info = {
                    'training_time': train_time,
                    'encryption_time': 0,  # 流式模式下加密时间在生成器中
                    'peak_memory_mb': max(peak_memory_before, peak_memory_mid),
                    'cpu_percent': max(cpu_percent_start, cpu_percent_mid)
                }
                update_generator = self.strategy.prepare_stream_update_request(
                    self.current_round, model_parameters, metrics_data, latency_info
                )
                encrypt_time = time.time() - encrypt_start
                
                # === 阶段3: 上传 (HE模式 - 流式) ===
                upload_start = time.time()
                success = self._submit_update_stream_with_retry(update_generator)
                upload_time = time.time() - upload_start
                
                if not success:
                    logger.error(f"{log_prefix} HE 流式上传失败，终止训练")
                    self.continue_training = False
                    tracemalloc.stop()
                    continue
            else:
                # === 阶段2: 加密/处理 (非HE模式) ===
                encrypt_start = time.time()
                update_request = self.strategy.prepare_update_request(self.current_round, model_parameters, metrics_data)
                encrypt_time = time.time() - encrypt_start
                payload_size_bytes = update_request.ByteSize()
                
                # 收集最终资源统计
                peak_memory_after_encrypt = process.memory_info().rss / 1024 / 1024
                cpu_percent_after = process.cpu_percent()
                
                # 添加延迟指标到请求中
                update_request.latency_metrics.training_time = train_time
                update_request.latency_metrics.encryption_time = encrypt_time
                update_request.latency_metrics.payload_size_bytes = payload_size_bytes
                update_request.latency_metrics.peak_memory_mb = max(peak_memory_before, peak_memory_mid, peak_memory_after_encrypt)
                update_request.latency_metrics.cpu_percent = max(cpu_percent_start, cpu_percent_mid, cpu_percent_after)
                
                # === 阶段3: 上传 ===
                upload_start = time.time()
                self._submit_update_with_retry(update_request)
                upload_time = time.time() - upload_start

            logger.info(f"{log_prefix} 等待全局模型更新...")

            # 使用流式订阅等待服务器聚合完成
            status_response = self._wait_for_training_status()
            
            if status_response.code == 300:
                logger.info(f"{log_prefix} 检测到收敛信号，终止训练")
                self.continue_training = False
            elif status_response.code == 500:
                logger.error(f"{log_prefix} 连接错误，终止训练")
                self.continue_training = False
            
            # === 阶段5: 下载 ===
            download_start = time.time()
            global_model_request = federation_pb2.GetModelRequest(client_id=self.client_id, round=self.current_round)
            global_model_response = self.stub.GetGlobalModel(global_model_request)
            download_size_bytes = global_model_response.ByteSize()
            global_params = deserialize_parameters(global_model_response.parameters)
            self.model.set_parameters(global_params)
            download_time = time.time() - download_start
            
            # 收集最终系统资源统计
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_final = process.memory_info().rss / 1024 / 1024  # MB
            
            round_time = time.time() - round_start_time
            
            # 本地日志（客户端控制台）
            logger.info(f"{log_prefix} 本轮完成: 训练={train_time:.2f}s, 加密={encrypt_time:.2f}s, "
                       f"上传={upload_time:.2f}s, 下载={download_time:.2f}s, 总耗时={round_time:.2f}s")
            
            self.current_round += 1

        logger.info(f"[Client {self.client_id}] 训练流程结束")

    def get_metrics(self):
        """计算并返回所有相关指标的字典。"""
        self.model.eval()
        test_acc, test_num, auc = self.test_metrics()
        loss, train_num = self.train_metrics()

        logger.info(f"[Client {self.client_id}][Round {self.current_round+1}] 本地评估: Acc={test_acc/test_num if test_num>0 else 0:.4f}, AUC={auc:.4f}, Loss={loss:.4f}")

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
            logger.info(f"[Client {self.client_id}] gRPC 通道已关闭")



def load_client_data():
    """
    加载客户端训练集和本地验证集数据
    
    数据结构：
    - /app/data/{dataset}_train.npz: 本地训练集 (X_train, y_train)
    - /app/data/{dataset}_val.npz: 本地验证集 (X_val, y_val) - 用于客户端本地评估
    
    注意：全局测试集保存在 /app/data/global/ 目录，由服务端加载进行全局评估
    """
    dataset_name = config['data']['dataset']
    train_path = f"/app/data/{dataset_name}_train.npz"
    val_path = f"/app/data/{dataset_name}_val.npz"
    
    client_id = os.environ.get('CLIENT_ID', 'unknown')
    try:
        # 加载本地训练集
        train_data = np.load(train_path)
        X_train = train_data["X_train"]
        y_train = train_data["y_train"]
        logger.info(f"[Client {client_id}] 加载训练集: {train_path}, 形状: {X_train.shape}")
        
        # 加载本地验证集
        val_data = np.load(val_path)
        X_val = val_data["X_val"]
        y_val = val_data["y_val"]
        logger.info(f"[Client {client_id}] 加载验证集: {val_path}, 形状: {X_val.shape}")
        
        # 返回数据，使用 X_test/y_test 作为键名以保持与现有代码的兼容性
        return {"X_train": X_train, "y_train": y_train, "X_test": X_val, "y_test": y_val}
    except FileNotFoundError as e:
        logger.error(f"[Client {client_id}] 无法找到数据文件: {e}")
        return None


def main():
    client_id = os.environ.get('CLIENT_ID', 'unknown')
    client_data = load_client_data()
    if client_data:
        client = FederatedLearningClient(data=client_data)
        try:
            client.participate_in_training()
        except Exception as e:
            logger.error(f"[Client {client_id}] 训练过程中发生致命错误: {e}", exc_info=True)
    else:
        logger.error(f"[Client {client_id}] 无法加载数据，客户端无法启动")

if __name__ == "__main__":
    main() 