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

        # -- 将在 setup_connection 中初始化 --
        self.stub = None
        self.channel = None
        self.privacy_mode = None
        self.he_public_key = None
        self.tee_public_key = None
        self.continue_training = True

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        logger.info(f"客户端 {self.client_id} 初始化完成，数据集大小: {self.data_size}")

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
            logger.info("使用安全通道(SSL/TLS)连接服务器。")
        except FileNotFoundError:
            logger.warning("未找到CA证书，使用不安全通道。")
            channel = grpc.insecure_channel(
                f"{self.server_host}:{self.server_port}",
                options=[('grpc.max_send_message_length', 500 * 1024 * 1024),
                         ('grpc.max_receive_message_length', 500 * 1024 * 1024)]
            )
        self.channel = channel
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)

        # -- 注册并协商运行模式 --
        logger.info(f"客户端{self.client_id}正在向服务器注册并获取联邦设置...")
        register_request = federation_pb2.ClientInfo(
            client_id=self.client_id,
            model_type="CNN",
            data_size=self.data_size
        )

        # 添加重试逻辑以应对服务器启动延迟
        max_retries = 5
        retry_interval = 3
        for attempt in range(max_retries):
            try:
                setup_response = self.stub.RegisterAndSetup(register_request)
                logger.info(f"客户端{self.client_id}注册成功。")
                break # 成功则跳出循环
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries - 1:
                    logger.warning(f"无法连接到服务器，将在 {retry_interval} 秒后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(retry_interval)
                else:
                    logger.error("多次尝试后仍无法连接到服务器，放弃连接。")
                    raise e
        
        self.privacy_mode = setup_response.privacy_mode
        logger.info(f"服务器运行模式为: {self.privacy_mode.upper()}")

        # 根据模式处理安全材料
        if self.privacy_mode == 'he':
            self.he_public_key = pickle.loads(setup_response.he_public_key)
            logger.info(f"客户端{self.client_id}成功接收并加载HE公钥。")
        
        elif self.privacy_mode == 'tee':
            # 1. 验证TEE身份
            report = json.loads(setup_response.tee_attestation_report.decode('utf-8'))
            actual_mrenclave = report.get("mrenclave")
            expected_mrenclave = config['tee']['expected_mrenclave']
            if actual_mrenclave == expected_mrenclave:
                logger.info(f"客户端{self.client_id}TEE身份验证成功！服务器可信。")
            else:
                raise Exception(f"TEE身份验证失败！预期MRENCLAVE为{expected_mrenclave}，实际为{actual_mrenclave}。")
            
            # 2. 加载公钥
            self.tee_public_key = serialization.load_pem_public_key(
                setup_response.tee_public_key,
            )
            logger.info("成功接收并加载TEE公钥。")

        # 设置初始模型参数
        initial_params = deserialize_parameters(setup_response.initial_model.parameters)
        self.model.set_parameters(initial_params)
        logger.info(f"客户端{self.client_id}已设置初始模型参数。")

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
    
    def _create_parameter_update_message(self, metrics_data):
        """创建参数更新消息（明文）"""
        parameters = self.model.get_parameters()
        serialized_params = serialize_parameters(parameters)
        training_metrics = federation_pb2.TrainingMetrics(
            test_acc=metrics_data.get('test_acc', 0.0),
            test_num=metrics_data.get('test_num', 0.0),
            auc=metrics_data.get('auc', 0.0),
            loss=metrics_data.get('loss', 0.0),
            train_num=metrics_data.get('train_num', 0)
        )
        model_parameters = federation_pb2.ModelParameters(
            parameters=serialized_params
        )
        params_and_metrics = federation_pb2.ParametersAndMetrics(
            parameters=model_parameters,
            metrics=training_metrics
        )
        return federation_pb2.ClientUpdate(
            client_id=self.client_id,
            round=self.current_round,  
            parameters_and_metrics=params_and_metrics
        )

    def _create_tee_parameter_update_message(self, metrics_data):
        """创建在TEE模式下加密的参数更新消息"""
        # 1. 创建包含明文参数和指标的载荷
        parameters = self.model.get_parameters()
        serialized_params = serialize_parameters(parameters)
        training_metrics = federation_pb2.TrainingMetrics(
            test_acc=metrics_data.get('test_acc', 0.0),
            test_num=metrics_data.get('test_num', 0.0),
            auc=metrics_data.get('auc', 0.0),
            loss=metrics_data.get('loss', 0.0),
            train_num=metrics_data.get('train_num', 0)
        )
        model_parameters = federation_pb2.ModelParameters(parameters=serialized_params)
        payload = federation_pb2.ParametersAndMetrics(
            parameters=model_parameters,
            metrics=training_metrics
        )
        serialized_payload = payload.SerializeToString()

        # 2. 生成一次性对称密钥(AES)并用RSA公钥加密它
        symmetric_key = AESGCM.generate_key(bit_length=256)
        encrypted_symmetric_key = self.tee_public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 3. 使用AES密钥加密主载荷
        aesgcm = AESGCM(symmetric_key)
        nonce = os.urandom(12)  # 96-bit nonce is recommended
        encrypted_payload = aesgcm.encrypt(nonce, serialized_payload, None)

        # 4. 创建最终的TEE更新消息
        return federation_pb2.TeeClientUpdate(
            client_id=self.client_id,
            round=self.current_round,
            encrypted_symmetric_key=encrypted_symmetric_key,
            nonce=nonce,
            encrypted_payload=encrypted_payload
        )

    def _create_encrypted_parameter_update_message(self, metrics_data):
        """创建参数更新消息（HE密文）"""
        parameters = self.model.get_parameters()
        encrypted_parameters = {}
        chunk_size = config['encryption']['chunk_size']
        # A ciphertext is a number modulo n^2. We need enough bytes to represent any such number.
        ciphertext_bytes_len = (self.he_public_key.nsquare.bit_length() + 7) // 8

        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                flat = value.flatten()
                encrypted_chunks = []
                total = len(flat)
                logger.info(f"开始加密参数 {key}, 形状: {value.shape}, 总参数量: {total}, 分块大小: {chunk_size}")

                for i in range(0, total, chunk_size):
                    chunk = flat[i:i+chunk_size]
                    encrypted_part = [self.he_public_key.encrypt(float(v)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') for v in chunk]
                    encrypted_chunks.extend(encrypted_part)
                    
                    progress = min(i + chunk_size, total)
                    logger.info(f"参数 {key} 加密进度: {progress}/{total} ({(progress/total)*100:.1f}%)")

                    del chunk, encrypted_part
                    gc.collect()
                
                encrypted_parameters[key] = {
                    'data': encrypted_chunks, 'shape': list(value.shape)
                }
            else: # scalar
                encrypted_value = self.he_public_key.encrypt(float(value))
                encrypted_parameters[key] = {
                    'data': [encrypted_value.ciphertext().to_bytes(ciphertext_bytes_len, 'big')], 'shape': [1]
                }
        
        proto_params = {k: federation_pb2.EncryptedNumpyArray(data=v['data'], shape=v['shape']) for k, v in encrypted_parameters.items()}
        model_parameters = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        encrypted_metrics = {k: self.he_public_key.encrypt(float(v)).ciphertext().to_bytes(ciphertext_bytes_len, 'big') for k, v in metrics_data.items()}
        encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(**encrypted_metrics)

        return federation_pb2.EncryptedClientUpdate(
            client_id=self.client_id, round=self.current_round,
            parameters_and_metrics=federation_pb2.EncryptedParametersAndMetrics(
                parameters=model_parameters, metrics=encrypted_metrics_proto
            )
        )

    def _submit_with_retry(self, submit_func, parameter_update, log_prefix):
        max_retries = config['grpc']['max_retries']
        retry_interval = config['grpc']['retry_interval']
        for attempt in range(max_retries):
            try:
                submit_func(parameter_update)
                logger.info(f"{log_prefix} 调用成功")
                return
            except grpc._channel._InactiveRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.warning(f"{log_prefix} 连接断开，重试 {attempt+1}/{max_retries}，原因: {e.details()}")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"{log_prefix} 调用异常: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                logger.error(f"{log_prefix} 调用异常: {str(e)}", exc_info=True)
                raise
        logger.error(f"{log_prefix} 多次重试后仍失败")
        raise RuntimeError(f"{log_prefix} 多次重试后仍失败")

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
            
            if self.privacy_mode == 'he':
                logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}在HE模式下: 正在加密并提交更新...")
                update_request = self._create_encrypted_parameter_update_message(metrics_data)
                self.stub.SubmitEncryptedUpdate(update_request)
            elif self.privacy_mode == 'tee':
                logger.info(f"[Round {self.current_round+1}] TEE模式: 正在加密并提交更新...")
                update_request = self._create_tee_parameter_update_message(metrics_data)
                self.stub.SubmitTeeUpdate(update_request)
            else: # 'none'
                logger.info(f"[Round {self.current_round+1}] NONE模式: 正在提交明文更新...")
                update_request = self._create_parameter_update_message(metrics_data)
                self.stub.SubmitUpdate(update_request)

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
                    logger.info(f"[Round {self.current_round+1}] 客户端{self.client_id}等待服务器聚合: {status_response.message} (当前进度: {status_response.registered_clients}/{status_response.total_clients})")
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

        logger.info(f"[Round {self.current_round+1}] 本地评估: Acc={test_acc/test_num if test_num>0 else 0:.4f}, AUC={auc:.4f}, Loss={loss/train_num if train_num>0 else 0:.4f}")

        return {
            'test_acc': test_acc,
            'test_num': test_num,
            'auc': auc,
            'loss': loss,
            'train_num': train_num
        }

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'channel') and self.channel:
                self.channel.close()
                logger.info("已关闭gRPC channel")
        except Exception as e:
            logger.error(f"关闭gRPC channel时出错: {str(e)}")



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