import threading
import json
import hashlib
import struct
from .base_aggregation_strategy import AggregationStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters, deserialize_parameters
from src.utils.logging_config import get_logger

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = get_logger()


class TeeAggregationStrategy(AggregationStrategy):
    """
    TDX (Trust Domain Extensions) 聚合策略。
    
    安全模型：
    - 整个聚合器运行在 Intel TDX Trust Domain (TD) 虚拟机中
    - 所有 TD 内存由硬件自动加密 (MKTME - Multi-Key Total Memory Encryption)
    - 客户端数据使用 RSA+AES 混合加密传输，在 TD 内解密
    - 解密后的数据在 TD 内存中受 TDX 硬件保护，主机/VMM 无法访问
    - 通过 TD Quote 提供远程证明，让客户端验证聚合器运行在真实 TDX 环境中
    """
    
    # TDX 设备路径
    TDX_GUEST_DEVICE = "/dev/tdx_guest"
    TDX_REPORT_DATA_SIZE = 64  # TDX report_data 最大 64 字节
    
    def __init__(self, server_instance):
        super().__init__(server_instance)
        
        # 1. 生成 RSA 密钥对（用于客户端加密通信）
        logger.info("[Server] TDX 模式：生成 RSA 密钥对...")
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        
        # 2. 检测 TDX 环境并生成证明
        self.is_tdx_environment = self._detect_tdx_environment()
        self.td_quote = None
        
        if self.is_tdx_environment:
            logger.info("[Server] TDX 模式：检测到 TDX 硬件环境，生成 TD Quote...")
            self.td_quote = self._generate_td_quote()
        else:
            logger.warning("[Server] TDX 模式：未检测到 TDX 硬件，将使用模拟证明（仅用于开发）")
            # 模拟 MRENCLAVE 用于非 TDX 环境的开发测试
            model_str = str(self.server.global_model.state_dict())
            self.mrenclave = hashlib.sha256(model_str.encode()).hexdigest()
        
        # 3. 用于记录每轮解密时间
        self.round_decrypt_times = {}
    
    def _detect_tdx_environment(self):
        """检测是否运行在 TDX Trust Domain 中"""
        import os
        
        # 检查 TDX guest 设备是否存在
        if os.path.exists(self.TDX_GUEST_DEVICE):
            logger.info(f"[Server] 检测到 TDX 设备: {self.TDX_GUEST_DEVICE}")
            return True
        
        # 备用检测：检查 dmesg 或 cpuid
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "tdx_guest" in cpuinfo.lower():
                    return True
        except:
            pass
        
        return False
    
    def _generate_td_quote(self):
        """
        使用 /dev/tdx_guest 生成 TD Quote（远程证明）。
        Quote 包含 TD 的测量值，可用于验证代码运行在真实 TDX 环境中。
        """
        import os
        import fcntl
        
        try:
            # 计算公钥的哈希作为 report_data，将公钥与 TD 身份绑定
            pubkey_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            report_data = hashlib.sha256(pubkey_pem).digest()
            # 确保 report_data 是 64 字节（TDX 要求）
            report_data = report_data.ljust(self.TDX_REPORT_DATA_SIZE, b'\x00')
            
            # TDX_CMD_GET_REPORT0 ioctl 命令（Linux 6.x 内核）
            # 定义: _IOWR('T', 1, struct tdx_report_req)
            # struct tdx_report_req { __u8 reportdata[64]; __u8 tdreport[1024]; }
            # 计算: _IOC(3, ord('T'), 1, 64+1024) = 0xc4405401
            TDX_CMD_GET_REPORT0 = 0xc4405401
            
            # 构建请求结构
            # struct tdx_report_req {
            #     __u8 reportdata[TDX_REPORTDATA_LEN];  // 64 字节
            #     __u8 tdreport[TDX_REPORT_LEN];        // 1024 字节
            # }
            req_buffer = bytearray(64 + 1024)
            req_buffer[:64] = report_data
            
            with open(self.TDX_GUEST_DEVICE, "rb+", buffering=0) as f:
                fcntl.ioctl(f.fileno(), TDX_CMD_GET_REPORT0, req_buffer)
            
            td_report = bytes(req_buffer[64:])
            
            # 对于完整的远程证明，还需要将 TD Report 发送给 QGS/PCCS 获取 Quote
            # 这里简化处理，直接返回 TD Report
            logger.info(f"[Server] ✅ TD Report 已生成 ({len(td_report)} 字节)")
            
            return {
                "type": "TDX_QUOTE",
                "report_data_hash": report_data[:32].hex(),
                "td_report": td_report.hex()[:128] + "...",  # 截断以便日志显示
                "td_report_full": td_report
            }
            
        except FileNotFoundError:
            logger.error(f"[Server] TDX 设备 {self.TDX_GUEST_DEVICE} 不存在")
            return None
        except PermissionError:
            logger.error(f"[Server] 无权限访问 {self.TDX_GUEST_DEVICE}，请使用 root 或添加权限")
            return None
        except OSError as e:
            # ioctl 可能失败，回退到简化模式
            logger.warning(f"[Server] TD Report 生成失败 ({e})，使用简化证明")
            pubkey_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return {
                "type": "TDX_SIMPLIFIED",
                "pubkey_hash": hashlib.sha256(pubkey_pem).hexdigest(),
                "note": "TD Quote generation failed, using simplified attestation"
            }
        except Exception as e:
            logger.error(f"[Server] 生成 TD Quote 时出错: {e}")
            return None

    def prepare_setup_response(self, request):
        """准备设置响应，包含 TDX 证明和公钥"""
        logger.debug(f"[Server] 向客户端 {request.client_id} 提供 TDX 证明报告和公钥")
        
        response = federation_pb2.SetupResponse(
            privacy_mode=self.server.privacy_mode,
            initial_model=federation_pb2.ModelParameters(
                parameters=serialize_parameters(self.server.global_model.get_parameters())
            )
        )
        
        # 构建证明报告
        if self.is_tdx_environment and self.td_quote:
            report_data = {
                "type": "TDX",
                "is_hardware_tdx": True,
                "quote": self.td_quote.get("type"),
                "pubkey_hash": self.td_quote.get("report_data_hash") or self.td_quote.get("pubkey_hash"),
            }
            logger.info(f"[Server] 提供真实 TDX 证明给客户端 {request.client_id}")
        else:
            report_data = {
                "type": "TDX_SIMULATED", 
                "is_hardware_tdx": False,
                "mrenclave": getattr(self, 'mrenclave', 'N/A'),
                "warning": "Running in development mode without TDX hardware"
            }
            logger.warning(f"[Server] 提供模拟 TDX 证明给客户端 {request.client_id}（非 TDX 硬件环境）")
        
        response.tee_attestation_report = json.dumps(report_data).encode('utf-8')
        response.tee_public_key = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return response

    def aggregate(self, request, context):
        """
        处理来自客户端的 TDX 加密更新。
        
        安全说明：
        - 解密在 TD 内存中进行，受 TDX 硬件保护
        - 解密后的明文数据存储在 TD 内存中，主机/VMM 无法访问
        - 整个聚合过程都在 Trust Domain 内完成
        """
        import time
        payload = request.tee
        round_num = request.round
        client_id = request.client_id
        
        if not payload:
            return federation_pb2.ServerUpdate(code=400, message="请求载荷与 'tee' 模式不匹配。")

        try:
            # 1. 在 TD 内存中解密载荷（受 TDX MKTME 保护）
            decrypt_start = time.time()
            
            # RSA 解密 AES 密钥
            symmetric_key = self.private_key.decrypt(
                payload.encrypted_symmetric_key,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
            
            # AES-GCM 解密主载荷
            aesgcm = AESGCM(symmetric_key)
            decrypted_payload_bytes = aesgcm.decrypt(payload.nonce, payload.encrypted_payload, None)
            
            # 反序列化
            params_and_metrics = federation_pb2.ParametersAndMetrics()
            params_and_metrics.ParseFromString(decrypted_payload_bytes)
            
            decrypt_time = time.time() - decrypt_start

            # 2. 处理解密后的数据（仍在 TD 内存中）
            with self.server.lock:
                if round_num != self.server.current_round:
                    return federation_pb2.ServerUpdate(code=400, message=f"轮次不匹配，服务器当前轮次为 {self.server.current_round}")
                
                # 累计本轮解密时间
                if round_num not in self.round_decrypt_times:
                    self.round_decrypt_times[round_num] = 0.0
                self.round_decrypt_times[round_num] += decrypt_time
                
                params, metrics_data = self._process_plaintext_update(params_and_metrics)
                self.server.clients[client_id].metrics = metrics_data
                self.server.client_parameters[round_num][client_id] = params
                
                logger.info(f"[Round {round_num+1}] 收到客户端 {client_id} TDX 更新 (解密: {decrypt_time:.4f}s)")

                submitted_clients = len(self.server.client_parameters[round_num])
                if submitted_clients >= self.server.expected_clients:
                    threading.Thread(target=self.server.process_round_completion, args=(round_num,)).start()

                return federation_pb2.ServerUpdate(
                    code=200, 
                    current_round=self.server.current_round, 
                    message="更新已收到"
                )

        except Exception as e:
            logger.error(f"[Server] 处理 TDX 更新失败: {e}", exc_info=True)
            return federation_pb2.ServerUpdate(code=500, message="解密或处理 TDX 载荷时发生错误")

    def _process_plaintext_update(self, params_and_metrics):
        """复用与 'none' 模式相同的明文处理逻辑。"""
        parameters = deserialize_parameters(params_and_metrics.parameters.parameters)
        metrics = params_and_metrics.metrics
        metrics_data = {
            'test_acc': metrics.test_acc, 'test_num': metrics.test_num, 'auc': metrics.auc,
            'loss': metrics.loss, 'train_num': metrics.train_num
        }
        return parameters, metrics_data 

    def aggregate_parameters(self, round_num):
        """
        在 TDX Trust Domain 内聚合客户端参数 (FedAvg)。
        
        安全说明：
        - 所有参数数据都在 TD 内存中，受 MKTME 硬件加密保护
        - 聚合计算在 TD 内完成，主机/VMM 无法观察计算过程
        - TDX 提供的是 VM 级别的机密计算，而非 enclave 级别
        """
        import time
        
        # 获取本轮累计的解密时间
        total_decrypt_time = self.round_decrypt_times.pop(round_num, 0.0)
        logger.info(f"[Round {round_num+1}][LATENCY] decryption={total_decrypt_time:.4f}s (TDX 保护的内存中解密)")
        
        aggregation_start = time.time()
        active_clients = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]
        parameters_list = list(self.server.client_parameters[round_num].values())
        
        total_data_size = sum(client.data_size for client in active_clients)
        if total_data_size == 0: 
            return self.server.global_model.get_parameters()
        
        client_weights = [client.data_size / total_data_size for client in active_clients]
        
        # FedAvg 加权聚合（在 TD 内存中进行）
        aggregated = {}
        param_structure = parameters_list[0]
        for param_name in param_structure.keys():
            aggregated[param_name] = sum(weight * params[param_name] for params, weight in zip(parameters_list, client_weights))
        
        aggregation_time = time.time() - aggregation_start
        logger.info(f"[Round {round_num+1}][LATENCY] aggregation={aggregation_time:.4f}s")
        
        # 记录 TDX 环境状态
        env_status = "TDX 硬件保护" if self.is_tdx_environment else "模拟模式"
        logger.info(f"[Round {round_num+1}] TDX 聚合完成 ({env_status})")
        
        return aggregated

    def evaluate_metrics(self, round_num, skip_acc_auc=False):
        """评估聚合指标（在 TDX Trust Domain 内完成）"""
        total_test_acc, total_test_num = 0, 0
        total_auc, total_loss, total_train_num = 0, 0, 0
        
        clients_in_round = [self.server.clients[cid] for cid in self.server.client_parameters[round_num].keys()]

        for c in clients_in_round:
            m = c.metrics
            if m:
                total_test_acc += m['test_acc']
                total_test_num += m['test_num']
                total_auc += m['auc'] * m['test_num']  # AUC 需要加权
                total_loss += m['loss']
                total_train_num += m['train_num']
        
        # 清理本轮存储的指标
        for c in clients_in_round: 
            c.metrics = None
        
        avg_loss = total_loss / total_train_num if total_train_num > 0 else 0
        self.server.rs_train_loss.append(avg_loss)
        
        if not skip_acc_auc:
            avg_acc = total_test_acc / total_test_num if total_test_num > 0 else 0
            avg_auc = total_auc / total_test_num if total_test_num > 0 else 0
            self.server.rs_test_acc.append(avg_acc)
            self.server.rs_auc.append(avg_auc)
            logger.info(f"[Round {round_num+1}] 客户端聚合 (TDX): Loss={avg_loss:.4f}")
        else:
            logger.info(f"[Round {round_num+1}] 客户端聚合 (TDX): Loss={avg_loss:.4f}")

        # 清理本轮的参数
        if round_num in self.server.client_parameters:
            del self.server.client_parameters[round_num] 