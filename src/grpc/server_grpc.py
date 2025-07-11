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

logger = get_logger()
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

        # -- 从配置加载参数 --
        self.privacy_mode = config['federation']['privacy_mode']
        self.expected_clients = config['federation']['expected_clients']
        self.max_rounds = config['federation']['max_rounds']
        self.acc_delta_threshold = config['federation']['convergence']['acc_delta_threshold']
        self.converge_window = config['federation']['convergence']['window']
        
        # -- 动态密钥/身份管理 --
        self.he_public_key = None
        self.he_private_key = None
        self.mrenclave = None

        if self.privacy_mode == 'he':
            logger.info("启动HE模式：正在生成Paillier密钥对...")
            key_size = config['encryption']['key_size']
            self.he_public_key, self.he_private_key = paillier.generate_paillier_keypair(n_length=key_size)
            logger.info(f"Paillier密钥对生成完毕 (密钥长度: {key_size} bits)。")
        
        elif self.privacy_mode == 'tee':
            logger.info("启动TEE模式：正在模拟生成身份指纹(MRENCLAVE)...")
            # 在真实场景中，MRENCLAVE由TEE构建过程生成。这里我们模拟它。
            # 我们对模型结构进行哈希来模拟一个稳定的、与代码相关的指纹。
            model_str = str(self.global_model.state_dict())
            self.mrenclave = hashlib.sha256(model_str.encode()).hexdigest()
            logger.info(f"模拟的MRENCLAVE为: {self.mrenclave}")

        logger.info(f"服务器初始化完成 (模式: {self.privacy_mode})，等待 {self.expected_clients} 个客户端注册...")

    def RegisterAndSetup(self, request, context):
        client_id = request.client_id
        logger.info(f"接收到客户端 {client_id} 的注册请求 (模型: {request.model_type}, 数据量: {request.data_size})")

        with self.lock:
            if client_id not in self.clients:
                if self.start_time is None:
                    self.start_time = time.time()
                    logger.info("联邦学习流程计时开始")
                self.clients[client_id] = ClientState(
                    client_id=client_id,
                    model_type=request.model_type,
                    data_size=request.data_size
                )
                logger.info(f"客户端 {client_id} 注册成功。当前 {len(self.clients)}/{self.expected_clients} 个客户端。")

            response = federation_pb2.SetupResponse(
                privacy_mode=self.privacy_mode,
                initial_model=federation_pb2.ModelParameters(
                    parameters=serialize_parameters(self.global_model.get_parameters())
                )
            )

            if self.privacy_mode == 'he':
                logger.info(f"向客户端 {client_id} 提供HE公钥。")
                response.he_public_key = pickle.dumps(self.he_public_key)

            elif self.privacy_mode == 'tee':
                logger.info(f"向客户端 {client_id} 提供模拟的证明报告。")
                # 模拟证明报告，真实场景下由TEE硬件和SDK生成
                report_data = {
                    "mrenclave": self.mrenclave,
                    "report_data": hashlib.sha256(request.client_id.encode()).hexdigest() # 绑定请求
                }
                response.attestation_report = json.dumps(report_data).encode('utf-8')

            return response

    def CheckTrainingStatus(self, request, context):
        client_id = request.client_id
        with self.lock:
            if self.converged:
                return federation_pb2.TrainingStatusResponse(code=300, message="训练已收敛，提前终止")
            
            ready_clients = len(self.clients)
            if ready_clients >= self.expected_clients:
                submitted_this_round = len(self.client_parameters.get(self.current_round, {}))
                if submitted_this_round < self.expected_clients:
                     return federation_pb2.TrainingStatusResponse(
                        code=200, 
                        message=f"可以开始训练/等待其他客户端提交参数 ({submitted_this_round}/{self.expected_clients})",
                        registered_clients=submitted_this_round, 
                        total_clients=self.expected_clients)
                else: # 所有人都提交了，等待下一轮
                    return federation_pb2.TrainingStatusResponse(
                        code=100, 
                        message=f"等待第 {self.current_round + 1} 轮开始",
                        registered_clients=submitted_this_round, 
                        total_clients=self.expected_clients)
            else:
                return federation_pb2.TrainingStatusResponse(
                    code=100, 
                    message=f"等待其他客户端注册 ({ready_clients}/{self.expected_clients})",
                    registered_clients=ready_clients, 
                    total_clients=self.expected_clients)

    def SubmitUpdate(self, request, context):
        """接收客户端明文模型更新"""
        if self.privacy_mode not in ['none', 'tee']:
            return federation_pb2.ServerUpdate(code=400, message="服务器当前不接受明文更新。")
        return self._submit_update_handler(request, context, is_encrypted=False)

    def SubmitEncryptedUpdate(self, request, context):
        """接收客户端密文模型更新"""
        if self.privacy_mode != 'he':
            return federation_pb2.ServerUpdate(code=400, message="服务器当前不接受密文更新。")
        return self._submit_update_handler(request, context, is_encrypted=True)

    def _submit_update_handler(self, request, context, is_encrypted):
        """统一处理更新请求的内部逻辑"""
        try:
            client_id = request.client_id
            round_num = request.round
            
            with self.lock:
                if round_num != self.current_round:
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.current_round}")

                if is_encrypted:
                    params, metrics_data = self._process_encrypted_update(request)
                    self.clients[client_id].encrypted_metrics = metrics_data
                else:
                    params, metrics_data = self._process_plaintext_update(request)
                    self.clients[client_id].metrics = metrics_data
                
                self.client_parameters[round_num][client_id] = params
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} 的 {'密文' if is_encrypted else '明文'} 更新。")

                submitted_clients = len(self.client_parameters[round_num])
                if submitted_clients >= self.expected_clients:
                    threading.Thread(target=self._process_round_completion, args=(round_num,), daemon=True).start()

                return federation_pb2.ServerUpdate(code=200, message="更新已收到")

        except Exception as e:
            logger.error(f"处理客户端更新时出错: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message=f"服务器错误: {str(e)}")

    def _process_plaintext_update(self, request):
        parameters = deserialize_parameters(request.parameters_and_metrics.parameters.parameters)
        metrics = request.parameters_and_metrics.metrics
        metrics_data = {
            'test_acc': metrics.test_acc, 'test_num': metrics.test_num, 'auc': metrics.auc,
            'loss': metrics.loss, 'train_num': metrics.train_num
        }
        return parameters, metrics_data

    def _process_encrypted_update(self, request):
        params = {}
        for key, enc_array in request.parameters_and_metrics.parameters.parameters.items():
            flat = [paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(b, 'big')) for b in enc_array.data]
            arr = np.array(flat, dtype=object).reshape(enc_array.shape)
            params[key] = arr
        
        metrics = request.parameters_and_metrics.metrics
        metrics_data = {
            'test_acc': paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(metrics.test_acc, 'big')),
            'test_num': paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(metrics.test_num, 'big')),
            'auc': paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(metrics.auc, 'big')),
            'loss': paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(metrics.loss, 'big')),
            'train_num': paillier.EncryptedNumber(self.he_private_key.public_key, int.from_bytes(metrics.train_num, 'big'))
        }
        return params, metrics_data

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
            
    def aggregate_parameters(self, round_num):
        """聚合明文客户端参数"""
        with self.lock:
            active_clients = [self.clients[cid] for cid in self.client_parameters[round_num].keys()]
            parameters_list = list(self.client_parameters[round_num].values())
        
        total_data_size = sum(client.data_size for client in active_clients)
        client_weights = [client.data_size / total_data_size for client in active_clients]
        
        aggregated = {}
        param_structure = parameters_list[0]
        for param_name in param_structure.keys():
            aggregated[param_name] = sum(weight * params[param_name] for params, weight in zip(parameters_list, client_weights))
        
        logger.info(f"[Round {round_num+1}] 明文参数聚合完成。")
        return aggregated

    def aggregate_encrypted_parameters(self, round_num):
        """聚合加密的客户端参数"""
        with self.lock:
            active_clients = [self.clients[cid] for cid in self.client_parameters[round_num].keys()]
            parameters_list = list(self.client_parameters[round_num].values())
        
        total_data_size = sum(client.data_size for client in active_clients)
        client_weights = [client.data_size / total_data_size for client in active_clients]

        aggregated = {}
        param_structure = parameters_list[0]
        for param_name in param_structure.keys():
            # 加权求和
            weighted_sum = sum(params[param_name] * weight for params, weight in zip(parameters_list, client_weights))
            
            # 解密
            logger.debug(f"开始解密参数 {param_name}...")
            flat_sum = weighted_sum.flatten()
            decrypted_flat = np.array([self.he_private_key.decrypt(val) for val in flat_sum])
            aggregated[param_name] = decrypted_flat.reshape(param_structure[param_name].shape)

        logger.info(f"[Round {round_num+1}] 密文参数聚合及解密完成。")
        return aggregated
        
    def _process_round_completion(self, round_num):
        """处理轮次完成，聚合参数并更新全局模型"""
        try:
            logger.info(f"[Round {round_num+1}] 所有客户端参数已收集完毕，开始聚合。")
            
            aggregator = self.aggregate_encrypted_parameters if self.privacy_mode == 'he' else self.aggregate_parameters
            aggregated_params = aggregator(round_num)
            
            self.global_model.set_parameters(aggregated_params)
            logger.info(f"[Round {round_num+1}] 全局模型参数更新完成。")
            
            self.evaluate(round_num) # 评估并检查收敛

            with self.lock:
                if self.converged or self.current_round >= self.max_rounds:
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    logger.info(f"训练结束。总耗时: {elapsed:.2f} 秒")
                    prefix = f"{self.privacy_mode}_"
                    plot_global_convergence_curve(self.rs_test_acc, self.rs_train_loss, self.rs_auc, prefix=prefix)
                else:
                    self.current_round += 1

        except Exception as e:
            logger.error(f"处理轮次 {round_num+1} 完成时出错: {e}", exc_info=True)
            
    def evaluate(self, round_num):
        """评估所有客户端的平均指标，并检查收敛"""
        if self.privacy_mode == 'he':
            self._evaluate_encrypted_metrics(round_num)
        else:
            self._evaluate_plaintext_metrics(round_num)

        # 检查收敛
        if len(self.rs_test_acc) >= self.converge_window:
            recent_accs = self.rs_test_acc[-(self.converge_window):]
            acc_delta = max(recent_accs) - min(recent_accs)
            if acc_delta < self.acc_delta_threshold:
                self.converged = True
                logger.info(f"[Round {round_num+1}] 训练已收敛，准确率变化 ({acc_delta:.6f}) 小于阈值 ({self.acc_delta_threshold})。")

    def _evaluate_plaintext_metrics(self, round_num):
        """评估明文指标"""
        total_test_acc, total_test_num = 0, 0
        total_auc, total_loss, total_train_num = 0, 0, 0
        
        with self.lock:
            clients_in_round = [self.clients[cid] for cid in self.client_parameters[round_num].keys()]

        for c in clients_in_round:
            m = c.metrics
            if m:
                total_test_acc += m['test_acc']
                total_test_num += m['test_num']
                total_auc += m['auc'] * m['test_num'] # AUC需要加权
                total_loss += m['loss'] # loss已经是加权过的
                total_train_num += m['train_num']

        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0
        
        self.rs_test_acc.append(avg_acc)
        self.rs_train_loss.append(avg_loss)
        self.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估: Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")

    def _evaluate_encrypted_metrics(self, round_num):
        """解密并评估加密指标"""
        agg_metrics = defaultdict(lambda: None)
        
        with self.lock:
            clients_in_round = [self.clients[cid] for cid in self.client_parameters[round_num].keys()]

        for c in clients_in_round:
            em = c.encrypted_metrics
            if em:
                for key, value in em.items():
                    agg_metrics[key] = value if agg_metrics[key] is None else agg_metrics[key] + value
        
        if not agg_metrics:
            return

        decrypted_metrics = {k: self.he_private_key.decrypt(v) for k, v in agg_metrics.items()}
        
        total_test_acc = decrypted_metrics.get('test_acc', 0)
        total_test_num = decrypted_metrics.get('test_num', 0)
        total_auc = decrypted_metrics.get('auc', 0)
        total_loss = decrypted_metrics.get('loss', 0)
        total_train_num = decrypted_metrics.get('train_num', 0)

        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0

        self.rs_test_acc.append(avg_acc)
        self.rs_train_loss.append(avg_loss)
        self.rs_auc.append(avg_auc)
        logger.info(f"[Round {round_num+1}] 全局评估 (HE): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")


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