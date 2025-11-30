import os
import socket
import pickle
import time
import numpy as np
import threading

from src.strategies.server.base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import deserialize_parameters, serialize_parameters

# --- 配置 ---
ENCLAVE_HOST = "aggregator"
ENCLAVE_PORT = 8000
RETRY_INTERVAL = 2
MAX_RETRIES = 20

class SgxAggregationStrategy(AggregationStrategy):
    """
    SGX模式的聚合策略。
    此策略通过TCP套接字与一个独立的、受信任的SGX enclave通信，
    以执行安全聚合。使用流式处理协议，逐个发送客户端数据以减少内存峰值。
    """
    def __init__(self, server):
        super().__init__(server)
        self.public_key, self.quote = self._get_initial_attestation()
        self.last_aggregated_metrics = None

    def _connect_to_enclave(self):
        """建立到聚合器enclave的TCP套接字连接。"""
        self.server.logger.info(f"正在尝试连接到SGX聚合器enclave at {ENCLAVE_HOST}:{ENCLAVE_PORT}...")
        for i in range(MAX_RETRIES):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ENCLAVE_HOST, ENCLAVE_PORT))
                self.server.logger.info("✅ 成功连接到SGX聚合器enclave。")
                return client_socket
            except (socket.error, ConnectionRefusedError) as e:
                self.server.logger.warning(f"连接enclave失败: {e}。等待聚合器启动... ({i+1}/{MAX_RETRIES})")
                time.sleep(RETRY_INTERVAL)
        raise ConnectionError("❌ 多次重试后未能连接到SGX聚合器enclave。")

    def _recv_exact(self, sock, length):
        """精确接收指定长度的数据"""
        data = b""
        while len(data) < length:
            packet = sock.recv(min(65536, length - len(data)))
            if not packet:
                raise ConnectionError("连接意外关闭")
            data += packet
        return data

    def _get_initial_attestation(self):
        """从enclave获取公钥和证明quote。"""
        self.server.logger.info("正在从enclave请求初始证明数据...")
        enclave_socket = self._connect_to_enclave()
        enclave_socket.sendall(b"GET_ATTESTATION")
        response_bytes = enclave_socket.recv(4096)
        pubkey_pem, quote = pickle.loads(response_bytes)
        self.server.logger.info("✅ 已从enclave收到公钥和quote。")
        enclave_socket.close()
        return pubkey_pem, quote

    def prepare_setup_response(self, request):
        """为客户端准备设置响应，包括enclave的公钥和quote。"""
        initial_model_params = self.server.global_model.get_parameters()
        
        return federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(parameters=serialize_parameters(initial_model_params)),
            tee_public_key=self.public_key,
            sgx_quote=self.quote
        )

    def aggregate(self, request, context):
        """
        处理来自客户端的TEE/SGX加密更新。
        仅存储加密的载荷，实际解密和聚合在 aggregate_parameters 中进行。
        """
        client_id = request.client_id
        round_num = request.round
        payload = request.tee

        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'sgx' 模式不匹配。")

        with self.server.lock:
            if round_num != self.server.current_round:
                return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前为 {self.server.current_round} 轮")

            self.server.client_parameters[round_num][client_id] = payload
            self.server.logger.info(f"已收到并存储来自客户端 {client_id} 的第 {round_num+1} 轮SGX更新。")

            if len(self.server.client_parameters[round_num]) >= self.server.expected_clients:
                threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

        return federation_pb2.ServerUpdate(code=200, message="Update received", current_round=round_num)
        
    def aggregate_parameters(self, round_num):
        """
        使用流式协议将参数的聚合委托给SGX enclave。
        逐个发送客户端数据，避免内存峰值。
        """
        self.server.logger.info(f"[第 {round_num+1} 轮] 正在将聚合任务委托给SGX enclave（流式模式）。")
        
        client_updates = self.server.client_parameters[round_num]
        num_clients = len(client_updates)
        
        enclave_socket = self._connect_to_enclave()
        try:
            # 1. 发送流式聚合命令
            enclave_socket.sendall(b"AGGREGATE_STREAM")
            
            if enclave_socket.recv(1024) != b"READY":
                raise ConnectionAbortedError("Enclave没有发出数据就绪信号。")

            # 2. 发送客户端数量 (4字节)
            enclave_socket.sendall(num_clients.to_bytes(4, byteorder='big'))
            self.server.logger.info(f"📤 开始流式发送 {num_clients} 个客户端的数据...")

            # 3. 逐个发送客户端数据
            for i, (client_id, update_payload) in enumerate(client_updates.items()):
                if not isinstance(update_payload, federation_pb2.TeePayload):
                    raise TypeError(f"SGX模式下期望TeePayload，但从客户端 {client_id} 收到了 {type(update_payload)}")
                
                num_samples = self.server.clients[client_id].data_size
                
                encrypted_key = update_payload.encrypted_symmetric_key
                nonce = update_payload.nonce
                encrypted_data = update_payload.encrypted_payload
                
                # 序列化单个客户端的数据
                client_data = pickle.dumps(((encrypted_key, nonce, encrypted_data), num_samples))
                data_length = len(client_data)
                
                # 发送长度 + 数据
                enclave_socket.sendall(data_length.to_bytes(8, byteorder='big'))
                enclave_socket.sendall(client_data)
                
                self.server.logger.info(f"📤 已发送客户端 {i+1}/{num_clients} 的数据 ({data_length / 1024 / 1024:.2f} MB)")

            self.server.logger.info("📤 所有客户端数据已发送，等待Enclave处理结果...")

            # 4. 接收结果长度 (8字节)
            length_bytes = self._recv_exact(enclave_socket, 8)
            result_length = int.from_bytes(length_bytes, byteorder='big')

            # 5. 接收结果数据
            response_data = self._recv_exact(enclave_socket, result_length)
            
            result = pickle.loads(response_data)
            if "error" in result:
                raise RuntimeError(f"Enclave返回错误: {result['error']}")

            self.server.logger.info("✅ 已从SGX enclave收到聚合后的参数和指标。")
            self.last_aggregated_metrics = result.get('metrics', {})
            aggregated_params = result.get('params', {})
            return {k: v for k, v in aggregated_params.items()}
        finally:
            enclave_socket.close()

    def evaluate_metrics(self, round_num):
        """
        使用由enclave计算并返回的聚合指标。
        """
        if self.last_aggregated_metrics is None:
            self.server.logger.warning(f"[第 {round_num+1} 轮] 没有可用的聚合指标。")
            return

        total_samples = self.last_aggregated_metrics.get('total_samples', 0)
        
        if total_samples > 0:
            avg_acc = self.last_aggregated_metrics.get('test_acc', 0)
            avg_auc = self.last_aggregated_metrics.get('auc', 0)
            avg_loss = self.last_aggregated_metrics.get('loss', 0)
            
            server_cpu = self.last_aggregated_metrics.get('server_cpu_time', 0)
            server_mem = self.last_aggregated_metrics.get('server_memory_usage', 0)
            
            self.server.rs_test_acc.append(avg_acc)
            self.server.rs_auc.append(avg_auc)
            self.server.rs_train_loss.append(avg_loss)
            self.server.logger.info(f"[第 {round_num+1} 轮] 全局指标 - 准确率: {avg_acc:.4f}, AUC: {avg_auc:.4f}, 损失: {avg_loss:.4f}")
            self.server.logger.info(f"[第 {round_num+1} 轮] Enclave资源 - CPU耗时: {server_cpu:.4f}s, 内存使用: {server_mem/1024/1024:.2f} MB")
        else:
            self.server.logger.warning(f"[第 {round_num+1} 轮] 聚合指标中总样本数为0。")
        
        self.last_aggregated_metrics = None
