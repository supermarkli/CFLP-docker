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
from src.models.models import FedAvgCNN

from src.grpc.generated import federation_pb2
from src.grpc.generated import federation_pb2_grpc
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.utils.draw import plot_global_convergence_curve  # 新增：导入绘图函数

logger = get_logger()

class ClientState:
    def __init__(self, client_id, model_type, data_size):
        self.client_id = client_id
        self.model_type = model_type
        self.data_size = data_size
        self.current_round = 0

class FederatedLearningServicer(federation_pb2_grpc.FederatedLearningServicer):
    def __init__(self, use_homomorphic_encryption=False):        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = FedAvgCNN().to(self.device)
        self.clients = {}  
        self.current_round = 0
        self.acc_delta_threshold = 0.001
        self.converge_window = 3
        self.count = 0
        self.next_step = False
        self.need_aggregate = False
        self.client_parameters = defaultdict(dict)
        self.aggregated_parameters = None
        self.lock = threading.Lock()  
        self.expected_clients = 3
        self.max_rounds = 100
        self.use_homomorphic_encryption = use_homomorphic_encryption or os.environ.get("USE_HOMOMORPHIC_ENCRYPTION", "False").lower() == "true"
        logger.info(f"同态加密状态: {'启用' if self.use_homomorphic_encryption else '未启用'}")
        self.start_time = None  
        self.end_time = None   
        self.rs_test_acc = [] 
        self.rs_train_loss = [] 
        self.rs_auc = []  # 新增：记录每轮AUC
        self.converged = False
        if self.use_homomorphic_encryption:
            try:
                import pickle
                with open('/app/certs/private_key.pkl', 'rb') as f:
                    self.private_key = pickle.load(f)
                logger.info("成功加载同态加密私钥")
            except FileNotFoundError:
                logger.error("未找到同态加密私钥文件: /app/certs/private_key.pkl")
                raise
            except Exception as e:
                logger.error(f"加载同态加密私钥失败: {str(e)}")
                raise
        logger.info(f"服务器初始化完成，等待 {self.expected_clients} 个客户端注册，计划训练 {self.max_rounds} 轮")

    def aggregate_parameters(self, round_num):
        """聚合客户端参数，保证参数和权重顺序严格一致"""
        try:
            self.evaluate()
            active_client_ids = list(self.client_parameters[round_num].keys())
            parameters_list = [self.client_parameters[round_num][cid] for cid in active_client_ids]
            active_clients = [self.clients[cid] for cid in active_client_ids]
            total_data_size = sum(client.data_size for client in active_clients)
            client_weights = [client.data_size / total_data_size for client in active_clients]
            logger.info(f"开始聚合参数，参数列表长度: {len(parameters_list)}")
            if not parameters_list:
                raise ValueError("参数列表为空")
            param_structure = parameters_list[0]
            aggregated = {}
            logger.info(f"当前轮次活跃客户端IDs: {active_client_ids}")
            logger.info(f"计算客户端权重完成: {client_weights}")
            for i, param_name in enumerate(param_structure.keys()):
                if not all(param_name in params for params in parameters_list):
                    raise ValueError(f"参数 {param_name} 在某些客户端中缺失")
                param_shape = parameters_list[0][param_name].shape
                param_dtype = parameters_list[0][param_name].dtype
                aggregated[param_name] = np.zeros(param_shape, dtype=param_dtype)
                for j, (params, weight) in enumerate(zip(parameters_list, client_weights)):
                    logger.debug(f"聚合参数 {param_name}: 客户端 {j+1}/{len(parameters_list)}, 权重 {weight}")
                    param_value = params[param_name]
                    if param_value.shape != param_shape:
                        raise ValueError(f"参数 {param_name} 的形状不一致")
                    aggregated[param_name] += weight * param_value
            return aggregated
        except Exception as e:
            logger.error(f"参数聚合过程中发生错误: {str(e)}")
            logger.exception(e)
            raise
        
    def Register(self, request, context):
        """处理客户端注册请求"""
        client_id = request.client_id
        logger.info(f"接收到客户端 {client_id} 的注册请求")
        
        with self.lock:
            if self.start_time is None:
                self.start_time = time.time()
                logger.info("联邦学习流程计时开始")
            self.clients[client_id] = ClientState(
                client_id=client_id,
                model_type=request.model_type,
                data_size=request.data_size
            )
            logger.info(f"客户端 {client_id} 注册成功，当前 {len(self.clients)}/{self.expected_clients} 个客户端")
            if len(self.clients) >= self.expected_clients:
                self.next_step = True
            return federation_pb2.RegisterResponse(
                code=200,
                parameters=federation_pb2.ModelParameters(
                    parameters=serialize_parameters(self.global_model.get_parameters())
                ),
                message="注册成功"
            )
            
    def CheckTrainingStatus(self, request, context):
        client_id = request.client_id
        with self.lock:
            # 新增：收敛判断
            if self.converged:
                code = 300
                message = "训练已收敛，提前终止"
                return federation_pb2.TrainingStatusResponse(
                    code=code,
                    message=message,
                    registered_clients=self.count,
                    total_clients=self.expected_clients
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
                total_clients=self.expected_clients
            )
            
    def SubmitUpdate(self, request, context):
        """接收客户端模型更新（优化锁粒度）"""
        try:
            client_id = request.client_id
            round_num = request.round
            parameters = deserialize_parameters(request.parameters_and_metrics.parameters.parameters)
            metrics = {
                'test_acc': request.parameters_and_metrics.metrics.test_acc,
                'test_num': request.parameters_and_metrics.metrics.test_num,
                'auc': request.parameters_and_metrics.metrics.auc,
                'loss': request.parameters_and_metrics.metrics.loss,
                'train_num': request.parameters_and_metrics.metrics.train_num
            }
            logger.info(f"[Round {self.current_round+1}] 接收到客户端 {client_id} 的参数更新")
            response_kwargs = {}

            with self.lock:
                if round_num != self.current_round:
                    logger.warning(f"客户端 {client_id} 的轮次 {round_num+1} 与服务器当前轮次 {self.current_round+1} 不匹配")
                    response_kwargs = dict(
                        code=400,
                        current_round=self.current_round,
                        message=f"轮次不匹配，当前服务器轮次为 {self.current_round+1}",
                        total_clients=self.expected_clients
                    )
                else:
                    self.client_parameters[round_num][client_id] = parameters
                    if client_id in self.clients:
                        self.clients[client_id].current_round = round_num
                        self.clients[client_id].metrics = metrics
                    logger.info(f"[Round {self.current_round+1}] 存储客户端 {client_id} 参数更新,已提交 {len(self.client_parameters[round_num])}/{self.expected_clients}个客户端")
                    submitted_clients = len(self.client_parameters[round_num])
                    if submitted_clients >= self.expected_clients:
                        logger.info(f"[Round {self.current_round+1}]所有客户端参数已收集完毕，开始聚合")
                        self.need_aggregate = True
                        threading.Thread(target=self._process_round_completion, args=(round_num,), daemon=True).start()
                        self.current_round += 1
                        time.sleep(1)
                        self.next_step = True
                        
                    response_kwargs = dict(
                        code=200,
                        current_round=self.current_round,
                        message="",
                        total_clients=self.expected_clients
                    )

            return federation_pb2.ServerUpdate(**response_kwargs)

        except Exception as e:
            logger.error(f"处理客户端更新时出错: {str(e)}")
            logger.exception(e)
            return federation_pb2.ServerUpdate(
                code=500,
                current_round=self.current_round,
                message=f"服务器错误: {str(e)}",
                total_clients=self.expected_clients
            )

    def SubmitEncryptedUpdate(self, request, context):
        """接收客户端密文模型更新（适配分块整体pickle解包）"""
        try:
            import pickle
            client_id = request.client_id
            round_num = request.round
            # 保持参数和metrics都处于加密状态
            encrypted_params = request.parameters_and_metrics.parameters.parameters
            params = {}
            for key, enc_array in encrypted_params.items():
                flat = []
                for b in enc_array.data:
                    flat.extend(pickle.loads(b))
                arr = np.array(flat, dtype=object).reshape(enc_array.shape)
                params[key] = arr

            # 存储加密的metrics
            encrypted_metrics = {
                'test_acc': pickle.loads(request.parameters_and_metrics.metrics.test_acc),
                'test_num': pickle.loads(request.parameters_and_metrics.metrics.test_num),
                'auc': pickle.loads(request.parameters_and_metrics.metrics.auc),
                'loss': pickle.loads(request.parameters_and_metrics.metrics.loss),
                'train_num': pickle.loads(request.parameters_and_metrics.metrics.train_num)
            }
            
            logger.info(f"[Round {self.current_round+1}] 接收到客户端 {client_id} 的密文参数更新")
            response_kwargs = {}

            with self.lock:
                if round_num != self.current_round:
                    logger.warning(f"客户端 {client_id} 的轮次 {round_num+1} 与服务器当前轮次 {self.current_round+1} 不匹配")
                    response_kwargs = dict(
                        code=400,
                        current_round=self.current_round,
                        message=f"轮次不匹配，当前服务器轮次为 {self.current_round+1}",
                        total_clients=self.expected_clients
                    )
                else:
                    self.client_parameters[round_num][client_id] = params
                    if client_id in self.clients:
                        self.clients[client_id].current_round = round_num
                        self.clients[client_id].encrypted_metrics = encrypted_metrics  # 存储加密的metrics
                    logger.info(f"[Round {self.current_round+1}] 存储客户端 {client_id} 密文参数更新,已提交 {len(self.client_parameters[round_num])}/{self.expected_clients}个客户端")
                    submitted_clients = len(self.client_parameters[round_num])
                    if submitted_clients >= self.expected_clients:
                        logger.info(f"[Round {self.current_round+1}]所有客户端参数已收集完毕，开始聚合")
                        threading.Thread(target=self._process_round_completion, args=(round_num,), daemon=True).start()
                        self.current_round += 1
                        time.sleep(1)
                        self.next_step = True
                        
                    response_kwargs = dict(
                        code=200,
                        current_round=self.current_round,
                        message="",
                        total_clients=self.expected_clients
                    )

            return federation_pb2.ServerUpdate(**response_kwargs)

        except Exception as e:
            logger.error(f"处理客户端密文更新时出错: {str(e)}")
            logger.exception(e)
            return federation_pb2.ServerUpdate(
                code=500,
                current_round=self.current_round,
                message=f"服务器错误: {str(e)}",
                total_clients=self.expected_clients
            )

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
            
    def aggregate_encrypted_parameters(self, round_num):
        """聚合加密的客户端参数，保证参数和权重顺序严格一致"""
        try:
            self.evaluate()
            active_client_ids = list(self.client_parameters[round_num].keys())
            parameters_list = [self.client_parameters[round_num][cid] for cid in active_client_ids]
            active_clients = [self.clients[cid] for cid in active_client_ids]
            total_data_size = sum(client.data_size for client in active_clients)
            client_weights = [client.data_size / total_data_size for client in active_clients]
            logger.info(f"开始聚合密文参数，参数列表长度: {len(parameters_list)}")
            if not parameters_list:
                raise ValueError("参数列表为空")
            param_structure = parameters_list[0]
            aggregated = {}
            logger.info(f"当前轮次活跃客户端IDs: {active_client_ids}")
            logger.info(f"计算客户端权重完成: {client_weights}")
            
            for param_name in param_structure.keys():
                if not all(param_name in params for params in parameters_list):
                    raise ValueError(f"参数 {param_name} 在某些客户端中缺失")
                param_shape = parameters_list[0][param_name].shape
                agg_array = None
                for j, (params, weight) in enumerate(zip(parameters_list, client_weights)):
                    logger.debug(f"聚合密文参数 {param_name}: 客户端 {j+1}/{len(parameters_list)}, 权重 {weight}")
                    param_value = params[param_name]
                    if param_value.shape != param_shape:
                        raise ValueError(f"参数 {param_name} 的形状不一致")
                    # 对每个加密值进行加权
                    weighted_array = np.vectorize(lambda x: x * weight, otypes=[object])(param_value)
                    if agg_array is None:
                        agg_array = weighted_array
                    else:
                        # 同态加密支持加法，直接相加
                        agg_array = np.vectorize(lambda a, b: a + b, otypes=[object])(agg_array, weighted_array)
                
                # 解密聚合后的参数
                decrypted = np.vectorize(self.private_key.decrypt)(agg_array)
                aggregated[param_name] = decrypted.reshape(param_shape)
            
            return aggregated
        except Exception as e:
            logger.error(f"密文参数聚合过程中发生错误: {str(e)}")
            logger.exception(e)
            raise

    def _process_round_completion(self, round_num):
        """处理轮次完成，聚合参数并更新全局模型（支持同态加密）"""
        try:
            if not self.use_homomorphic_encryption:
                try:
                    aggregated_params = self.aggregate_parameters(round_num)
                    self.global_model.set_parameters(aggregated_params)
                    logger.info(f"[Round {self.current_round}] 全局模型参数更新完成")
                except Exception as e:
                    logger.error(f"参数聚合或模型更新时出错: {str(e)}")
                    logger.exception(e)
                    raise
            else:
                try:
                    logger.info("开始同态加密参数聚合")
                    aggregated_params = self.aggregate_encrypted_parameters(round_num)
                    self.global_model.set_parameters(aggregated_params)
                    logger.info(f"[Round {self.current_round}] 全局模型参数更新完成（同态加密聚合）")
                except Exception as e:
                    logger.error(f"密文参数聚合或模型更新时出错: {str(e)}")
                    logger.exception(e)
                    raise

            if self.converged or self.current_round  >= self.max_rounds:
                prefix = 'he_' if self.use_homomorphic_encryption else ''
                plot_global_convergence_curve(self.rs_test_acc, self.rs_train_loss, self.rs_auc, prefix=prefix)
                logger.info("训练结束，已绘制收敛曲线图 out/convergence_curve.png")
                logger.info(f"达到最大轮次 {self.current_round }，结束训练")
                self.end_time = time.time()
                elapsed = self.end_time - self.start_time if self.start_time else None
                self.converged = True
                if elapsed is not None:
                    logger.info(f"联邦学习流程总耗时: {elapsed:.2f} 秒")
            logger.info(f"轮次 {round_num+1} 处理完成")
        except Exception as e:
            logger.error(f"处理轮次完成时出错: {str(e)}")
            logger.exception(e)

    def evaluate(self, acc=None, loss=None):
        """评估所有客户端的平均准确率、AUC和训练损失，并打印和保存历史"""
        if not self.use_homomorphic_encryption:
            # 明文metrics的处理逻辑保持不变
            total_test_acc = 0
            total_test_num = 0
            total_auc = 0
            total_loss = 0
            total_train_num = 0
            accs = []
            aucs = []
            for c in self.clients.values():
                m = getattr(c, 'metrics', None)
                if m:
                    ta = m.get('test_acc', 0)
                    tn = m.get('test_num', 0)
                    auc = m.get('auc', 0)
                    l = m.get('loss', 0)
                    trn = m.get('train_num', 0)
                    total_test_acc += ta
                    total_test_num += tn
                    total_auc += auc * tn
                    total_loss += l * trn
                    total_train_num += trn
                    accs.append(ta / tn if tn > 0 else 0)
                    aucs.append(auc)
        else:
            agg_test_acc = None
            agg_test_num = None
            agg_auc = None
            agg_loss = None
            agg_train_num = None
            # 新增：累加所有客户端的加密指标
            for c in self.clients.values():
                encrypted_metrics = getattr(c, 'encrypted_metrics', None)
                if encrypted_metrics:
                    if agg_test_acc is None:
                        agg_test_acc = encrypted_metrics['test_acc']
                        agg_test_num = encrypted_metrics['test_num']
                        agg_auc = encrypted_metrics['auc']
                        agg_loss = encrypted_metrics['loss']
                        agg_train_num = encrypted_metrics['train_num']
                    else:
                        agg_test_acc += encrypted_metrics['test_acc']
                        agg_test_num += encrypted_metrics['test_num']
                        agg_auc += encrypted_metrics['auc']
                        agg_loss += encrypted_metrics['loss']
                        agg_train_num += encrypted_metrics['train_num']
            if agg_test_acc is not None:
                total_test_acc = self.private_key.decrypt(agg_test_acc)
                total_test_num = int(self.private_key.decrypt(agg_test_num))
                total_auc = self.private_key.decrypt(agg_auc)
                total_loss = self.private_key.decrypt(agg_loss)
                total_train_num = int(self.private_key.decrypt(agg_train_num))
            else:
                total_test_acc = 0
                total_test_num = 0
                total_auc = 0
                total_loss = 0
                total_train_num = 0

        # 计算平均值
        avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
        avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0

        if acc is None:
            self.rs_test_acc.append(avg_acc)
        else:
            acc.append(avg_acc)
        if loss is None:
            self.rs_train_loss.append(avg_loss)
        else:
            loss.append(avg_loss)
        self.rs_auc.append(avg_auc)

        logger.info("Averaged Train Loss: {:.4f}".format(avg_loss))
        logger.info("Averaged Test Accuracy: {:.4f}".format(avg_acc))
        logger.info("Averaged Test AUC: {:.4f}".format(avg_auc))
        if not self.use_homomorphic_encryption:
            logger.info("Std Test Accuracy: {:.4f}".format(np.std(accs)))
            logger.info("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if len(self.rs_test_acc) >= self.converge_window + 1:
            window = self.converge_window
            recent_accs = self.rs_test_acc[-(window+1):]
            acc_delta = max(recent_accs) - min(recent_accs)
            threshold = self.acc_delta_threshold
            if acc_delta < threshold:
                self.converged = True
                logger.info(f"[自动收敛] 最近{window+1}轮准确率变化({acc_delta:.6f})小于阈值({threshold})，判定收敛，训练将提前终止。")

def serve():
    """启动gRPC服务器"""
    # 创建服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )
    
    # 添加服务
    federation_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(use_homomorphic_encryption=True), server
    )
    
    port = os.environ.get("GRPC_SERVER_PORT", "50051")
    
    try:
        # 尝试读取服务器证书和私钥
        with open('/app/certs/server.key', 'rb') as f:
            private_key = f.read()
        with open('/app/certs/server.crt', 'rb') as f:
            certificate_chain = f.read()
            
        # 创建服务器安全凭证
        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)]
        )
        
        # 使用安全端口
        server.add_secure_port(f"[::]:{port}", server_credentials)
        logger.info(f"联邦学习安全服务器正在启动，监听端口: {port}")
        
    except FileNotFoundError as e:
        # 如果找不到证书文件，使用不安全端口
        logger.warning(f"未找到证书文件: {str(e)}，将使用不安全端口")
        server.add_insecure_port(f"[::]:{port}")
        logger.info(f"联邦学习服务器（不安全模式）正在启动，监听端口: {port}")
        
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise
    
    # 启动服务器
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 