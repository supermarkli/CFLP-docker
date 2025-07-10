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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from tqdm import tqdm 
import random  # 新增

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 在模块加载时设置随机种子

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging_config import get_logger
from src.grpc.generated import federation_pb2
from src.grpc.generated import federation_pb2_grpc
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.models.models import FedAvgCNN
from src.utils.config_utils import config # 导入全局配置

logger = get_logger()

class FederatedLearningClient:
    def __init__(self, data=None):
        self.client_id = config['client_id'] or str(uuid.uuid4())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FedAvgCNN().to(self.device)
        self.num_classes = 10
        self.current_round = 0
        self.batch_size = config['training']['batch_size']
        self.server_host = config['grpc']['server_host']
        self.server_port = config['grpc']['server_port']
        self._init_data(data)
        self.continue_training = True

        self.use_homomorphic_encryption = config['encryption']['enabled']
        logger.info(f"同态加密状态: {'启用' if self.use_homomorphic_encryption else '未启用'}")
        if self.use_homomorphic_encryption:
            try:
                import pickle
                with open('/app/certs/public_key.pkl', 'rb') as f:
                    self.public_key = pickle.load(f)
                logger.info("成功加载同态加密公钥")
            except FileNotFoundError:
                logger.error("未找到同态加密公钥文件: /app/certs/public_key.pkl")
                raise
            except Exception as e:
                logger.error(f"加载同态加密公钥失败: {str(e)}")
                raise
        try:
            # 尝试读取CA证书
            with open('/app/certs/ca.crt', 'rb') as f:
                ca_cert = f.read()
            
            # 创建安全凭证
            credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
            
            # 创建安全通道
            self.channel = grpc.secure_channel(
                f"{self.server_host}:{self.server_port}",
                credentials,
                options=[
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024)
                ]
            )
            logger.info("使用安全通道(SSL/TLS)连接服务器")
            
        except FileNotFoundError:
            logger.warning("未找到CA证书文件: /app/certs/ca.crt，将使用不安全通道")
            self.channel = grpc.insecure_channel(
                f"{self.server_host}:{self.server_port}",
                options=[
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024)
                ]
            )
            logger.info("使用不安全通道连接服务器")
            
        except Exception as e:
            logger.error(f"gRPC连接初始化失败: {str(e)}")
            raise
        
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['training']['learning_rate'])

        logger.info(f"客户端 {self.client_id} 初始化完成，数据集大小: {self.data_size}")

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
    
    def _create_parameter_update_message(self, metrics):
        """创建参数更新消息（明文）"""
        parameters = self.model.get_parameters()
        serialized_params = serialize_parameters(parameters)
        training_metrics = federation_pb2.TrainingMetrics(
            test_acc=metrics.get('test_acc', 0.0),
            test_num=metrics.get('test_num', 0.0),
            auc=metrics.get('auc', 0.0),
            loss=metrics.get('loss', 0.0),
            train_num=metrics.get('train_num', 0)
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

    def _create_encrypted_parameter_update_message(self, metrics):
        """创建参数更新消息（密文，分块整体pickle优化）"""
        parameters = self.model.get_parameters()
        encrypted_parameters = {}
        import gc
        chunk_size = config['encryption']['chunk_size']
        for key, value in parameters.items():
            try:
                if isinstance(value, np.ndarray):
                    flat = value.flatten()
                    encrypted_chunks = []
                    total = len(flat)
                    logger.info(f"开始加密参数 {key}, 形状: {value.shape}, 总参数量: {total}")
                    for i in range(0, total, chunk_size):
                        chunk = flat[i:i+chunk_size]
                        encrypted_chunk = [self.public_key.encrypt(float(v)) for v in chunk]
                        encrypted_bytes = pickle.dumps(encrypted_chunk)
                        encrypted_chunks.append(encrypted_bytes)
                        del encrypted_chunk, encrypted_bytes
                        gc.collect()
                        logger.info(f"参数 {key} 加密进度: {min(i+chunk_size, total)}/{total} ({(min(i+chunk_size, total)/total)*100:.1f}%)")
                    encrypted_parameters[key] = {
                        'data': encrypted_chunks,  
                        'shape': list(value.shape)
                    }
                    del flat, encrypted_chunks
                    gc.collect()
                else:
                    encrypted_value = self.public_key.encrypt(float(value))
                    encrypted_bytes = pickle.dumps([encrypted_value])
                    encrypted_parameters[key] = {
                        'data': [encrypted_bytes],
                        'shape': [1]
                    }
            except Exception as e:
                logger.error(f"加密参数 {key} 时出错: {e}", exc_info=True)
                raise
        proto_params = {}
        for k, v in encrypted_parameters.items():
            proto_params[k] = federation_pb2.EncryptedNumpyArray(
                data=v['data'],
                shape=v['shape']
            )
        model_parameters = federation_pb2.EncryptedModelParameters(parameters=proto_params)

        encrypted_metrics = {}
        metrics_to_encrypt = {
            'test_acc': float(metrics.get('test_acc', 0.0)),
            'test_num': float(metrics.get('test_num', 0.0)),
            'auc': float(metrics.get('auc', 0.0)),
            'loss': float(metrics.get('loss', 0.0)),
            'train_num': float(metrics.get('train_num', 0.0))
        }
        
        for key, value in metrics_to_encrypt.items():
            encrypted_metrics[key] = pickle.dumps(self.public_key.encrypt(float(value)))

        encrypted_metrics_proto = federation_pb2.EncryptedTrainingMetrics(
            test_acc=encrypted_metrics['test_acc'],
            test_num=encrypted_metrics['test_num'],
            auc=encrypted_metrics['auc'],
            loss=encrypted_metrics['loss'],
            train_num=encrypted_metrics['train_num']
        )

        params_and_metrics = federation_pb2.EncryptedParametersAndMetrics(
            parameters=model_parameters,
            metrics=encrypted_metrics_proto
        )

        return federation_pb2.EncryptedClientUpdate(
            client_id=self.client_id,
            round=self.current_round,
            parameters_and_metrics=params_and_metrics
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
        register_request = federation_pb2.ClientInfo(
            client_id=self.client_id,
            model_type="CNN",
            data_size=self.data_size
        )
        register_response = self.stub.Register(register_request)

        if register_response.parameters and register_response.parameters.parameters:
            initial_params = deserialize_parameters(register_response.parameters.parameters)
            self.model.set_parameters(initial_params)
            logger.info("已设置初始模型参数")
        
        while True:
            status_request = federation_pb2.ClientInfo(
                client_id=self.client_id
            )
            status_response = self.stub.CheckTrainingStatus(status_request)
            if status_response.code == 100:
                logger.info(f"等待其他客户端注册 ({status_response.registered_clients}/{status_response.total_clients})")
                time.sleep(1)  
                continue
            else :
                logger.info(f"所有客户端已就绪，开始训练")
                break
        
        while self.continue_training:
            logger.info(f"[Round {self.current_round+1}] 开始训练")
            self.train(epochs=config['training']['epochs'])
            logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 完成本地训练")
            test_acc, test_num, auc = self.test_metrics()
            logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 测试集: acc={test_acc}, num={test_num}, auc={auc}")
            loss, train_num = self.train_metrics()
            logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 训练集: loss={loss}, num={train_num}")
            metrics = {
                'test_acc': test_acc,
                'test_num': test_num,
                'auc': auc,
                'loss': loss,
                'train_num': train_num
                }
            if self.use_homomorphic_encryption:
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 正在加密模型参数...")
                parameter_update = self._create_encrypted_parameter_update_message(metrics)
                log_prefix = f"[Round {self.current_round}] SubmitEncryptedUpdate"
                self._submit_with_retry(self.stub.SubmitEncryptedUpdate, parameter_update, log_prefix)
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 提交密文参数更新成功")
            else:
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 正在创建明文参数更新消息...")
                parameter_update = self._create_parameter_update_message(metrics)
                log_prefix = f"[Round {self.current_round+1}] SubmitUpdate"
                self._submit_with_retry(self.stub.SubmitUpdate, parameter_update, log_prefix)
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 提交明文参数更新成功")
            
            while True:
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 等待其他客户端提交参数")
                status_request = federation_pb2.ClientInfo(
                    client_id=self.client_id
                )
                time.sleep(1)
                try:
                    status_response = self.stub.CheckTrainingStatus(status_request)
                except Exception as e:
                    logger.error(f"[Round {self.current_round}] CheckTrainingStatus 调用异常: {str(e)}", exc_info=True)
                    raise
                if status_response.code == 100:
                    logger.info(f"[Round {self.current_round+1}] 等待其他客户端提交参数 ({status_response.registered_clients}/{status_response.total_clients})")
                    time.sleep(1)  
                    continue
                elif status_response.code == 200:
                    logger.info(f"[Round {self.current_round+1}] 所有客户端已就绪，开始请求全局模型")
                    break
                elif status_response.code == 300:
                    self.continue_training = False
                    logger.info(f"[Round {self.current_round+1}] 检测到服务器收敛信号，客户端提前终止训练")
                    break
                else:
                    logger.error(f"检查训练状态失败，状态码: {status_response.code}, 消息: {status_response.message}")
                    return
            
            logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 请求全局模型")
            global_model_request = federation_pb2.GetModelRequest(
                client_id=self.client_id,
                round=self.current_round
            )
            try:
                global_model_response = self.stub.GetGlobalModel(global_model_request)
            except Exception as e:
                logger.error(f"[Round {self.current_round}] GetGlobalModel 调用异常: {str(e)}", exc_info=True)
                raise
            
            try:
                global_params = deserialize_parameters(global_model_response.parameters)
                self.model.set_parameters(global_params)
                logger.info(f"[Round {self.current_round+1}] 客户端 {self.client_id} 更新模型参数")
            except Exception as e:
                logger.error(f"[Round {self.current_round}] 反序列化或设置模型参数异常: {str(e)}", exc_info=True)
                raise
            self.current_round += 1

        logger.info(f"[Round {self.current_round}] 客户端 {self.client_id} 完成训练")
        
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'channel'):
                self.channel.close()
                logger.info("已关闭gRPC通道")
        except Exception as e:
            logger.error(f"关闭gRPC通道时出错: {str(e)}")



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
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
    else:
        logger.error("无法加载数据，客户端无法启动。")

if __name__ == "__main__":
    main() 