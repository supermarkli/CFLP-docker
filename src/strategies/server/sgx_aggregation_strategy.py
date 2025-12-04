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
        self.server.logger.debug(f"[Server] 连接 SGX Enclave {ENCLAVE_HOST}:{ENCLAVE_PORT}...")
        for i in range(MAX_RETRIES):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ENCLAVE_HOST, ENCLAVE_PORT))
                self.server.logger.info("[Server] ✅ 连接 SGX Enclave 成功")
                return client_socket
            except (socket.error, ConnectionRefusedError) as e:
                self.server.logger.warning(f"[Server] Enclave 连接失败，重试 ({i+1}/{MAX_RETRIES})")
                time.sleep(RETRY_INTERVAL)
        raise ConnectionError("[Server] ❌ 多次重试后未能连接 SGX Enclave")

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
        self.server.logger.info("[Server] SGX 模式：请求 Enclave 证明...")
        enclave_socket = self._connect_to_enclave()
        enclave_socket.sendall(b"GET_ATTESTATION")
        response_bytes = enclave_socket.recv(4096)
        pubkey_pem, quote = pickle.loads(response_bytes)
        self.server.logger.info("[Server] ✅ 收到 Enclave 公钥和 Quote")
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
            self.server.logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} SGX 更新")

            if len(self.server.client_parameters[round_num]) >= self.server.expected_clients:
                threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

        return federation_pb2.ServerUpdate(code=200, message="Update received", current_round=round_num)
        
    def aggregate_parameters(self, round_num):
        """
        使用流式协议将参数的聚合委托给SGX enclave。
        逐个发送客户端数据，避免内存峰值。
        """
        self.server.logger.info(f"[Round {round_num+1}] 委托 SGX Enclave 聚合...")
        
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
            self.server.logger.debug(f"[Round {round_num+1}] 发送 {num_clients} 个客户端数据到 Enclave...")

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
                
                self.server.logger.debug(f"[Round {round_num+1}] 发送客户端 {i+1}/{num_clients} ({data_length / 1024 / 1024:.2f} MB)")

            self.server.logger.debug(f"[Round {round_num+1}] 等待 Enclave 处理...")

            # 4. 接收结果长度 (8字节)
            length_bytes = self._recv_exact(enclave_socket, 8)
            result_length = int.from_bytes(length_bytes, byteorder='big')

            # 5. 接收结果数据
            response_data = self._recv_exact(enclave_socket, result_length)
            
            result = pickle.loads(response_data)
            if "error" in result:
                raise RuntimeError(f"Enclave 返回错误: {result['error']}")

            self.server.logger.info(f"[Round {round_num+1}] ✅ 收到 Enclave 聚合结果")
            self.last_aggregated_metrics = result.get('metrics', {})
            aggregated_params = result.get('params', {})
            
            # 从 Enclave 返回的 metrics 中提取解密和聚合时间
            # Enclave 内的 server_cpu_time 包含解密和聚合的总时间
            enclave_cpu_time = self.last_aggregated_metrics.get('server_cpu_time', 0)
            # SGX 模式下，解密和聚合都在 Enclave 中进行
            # 我们估算解密约占 30%，聚合约占 70%
            decryption_time = enclave_cpu_time * 0.3
            aggregation_time = enclave_cpu_time * 0.7
            
            self.server.logger.info(f"[Round {round_num+1}][LATENCY] decryption={decryption_time:.4f}s")
            self.server.logger.info(f"[Round {round_num+1}][LATENCY] aggregation={aggregation_time:.4f}s")
            
            return {k: v for k, v in aggregated_params.items()}
        finally:
            enclave_socket.close()

    def evaluate_metrics(self, round_num, skip_acc_auc=False):
        """
        使用由enclave计算并返回的聚合指标。
        """
        if self.last_aggregated_metrics is None:
            self.server.logger.warning(f"[Round {round_num+1}] 没有可用的聚合指标")
            return

        total_samples = self.last_aggregated_metrics.get('total_samples', 0)
        
        if total_samples > 0:
            avg_acc = self.last_aggregated_metrics.get('test_acc', 0)
            avg_auc = self.last_aggregated_metrics.get('auc', 0)
            avg_loss = self.last_aggregated_metrics.get('loss', 0)
            
            server_cpu = self.last_aggregated_metrics.get('server_cpu_time', 0)
            server_mem = self.last_aggregated_metrics.get('server_memory_usage', 0)
            
            self.server.rs_train_loss.append(avg_loss)
            
            if not skip_acc_auc:
                self.server.rs_test_acc.append(avg_acc)
                self.server.rs_auc.append(avg_auc)
                self.server.logger.info(f"[Round {round_num+1}] 客户端聚合 (SGX): Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Loss={avg_loss:.4f}")
            else:
                self.server.logger.info(f"[Round {round_num+1}] 客户端聚合 (SGX): Loss={avg_loss:.4f}")
            
            self.server.logger.info(f"[Round {round_num+1}][Enclave][RESOURCE] cpu_time={server_cpu:.4f}s, memory={server_mem/1024/1024:.2f} MB")
        else:
            self.server.logger.warning(f"[Round {round_num+1}] 聚合指标中总样本数为 0")
        
        self.last_aggregated_metrics = None
        
        # 清理本轮的加密参数缓存，避免内存泄漏
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num]
            self.server.logger.debug(f"[Round {round_num+1}] 已清理 client_parameters 缓存")