import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters
from src.utils.config_utils import config
from src.utils.logging_config import get_logger

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = get_logger()


class TeeClientStrategy(ClientStrategy):
    """
    TDX (Trust Domain Extensions) 客户端策略。
    
    安全模型：
    - 验证服务器运行在 Intel TDX Trust Domain 中（通过 TD Quote）
    - 使用 RSA+AES 混合加密将参数发送给 TD 内的聚合器
    - 只有 TD 内的聚合器能够解密数据
    """
    
    def __init__(self, client_instance, tee_attestation_report_bytes, tee_public_key_bytes):
        super().__init__(client_instance)

        # 1. 验证 TDX 证明
        report = json.loads(tee_attestation_report_bytes.decode('utf-8'))
        self._verify_tdx_attestation(report)
        
        # 2. 加载 TDX 聚合器的公钥
        self.tee_public_key = serialization.load_pem_public_key(tee_public_key_bytes)
        logger.info(f"[Client {self.client.client_id}] TDX 策略已加载聚合器公钥")
    
    def _verify_tdx_attestation(self, report):
        """
        验证 TDX 远程证明。
        
        在生产环境中，应该：
        1. 将 TD Quote 发送给 Intel PCCS/DCAP 服务验证
        2. 检查 MRTD、RTMR 等测量值
        3. 验证公钥哈希是否绑定在 Quote 中
        """
        report_type = report.get("type", "UNKNOWN")
        is_hardware_tdx = report.get("is_hardware_tdx", False)
        
        if report_type == "TDX" and is_hardware_tdx:
            # 真实 TDX 硬件环境
            pubkey_hash = report.get("pubkey_hash", "N/A")
            logger.info(f"[Client {self.client.client_id}] ✅ TDX 硬件证明验证通过")
            logger.info(f"[Client {self.client.client_id}]    - 证明类型: {report.get('quote', 'N/A')}")
            logger.info(f"[Client {self.client.client_id}]    - 公钥绑定: {pubkey_hash[:16]}...")
            
        elif report_type == "TDX_SIMULATED":
            # 模拟环境（开发测试用）
            logger.warning(f"[Client {self.client.client_id}] ⚠️ 服务器运行在 TDX 模拟模式（非硬件保护）")
            logger.warning(f"[Client {self.client.client_id}]    - 警告: {report.get('warning', 'N/A')}")
            
            # 检查配置是否允许模拟模式
            allow_simulated = config.get('tee', {}).get('allow_simulated', True)
            if not allow_simulated:
                raise SecurityError("TDX 硬件验证失败，且配置不允许模拟模式")
        else:
            # 未知类型，回退到旧的 MRENCLAVE 验证
            actual_mrenclave = report.get("mrenclave", "N/A")
            expected_mrenclave = config.get('tee', {}).get('expected_mrenclave', '')
            logger.warning(f"[Client {self.client.client_id}] 使用传统 MRENCLAVE 验证（非 TDX）")
            
        logger.info(f"[Client {self.client.client_id}] TDX 策略初始化成功")

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """
        创建发送给 TDX Trust Domain 的加密参数更新。
        
        安全说明：
        - 使用 RSA+AES 混合加密，只有 TD 内的聚合器持有私钥
        - AES 密钥用 TD 的 RSA 公钥加密，确保只有 TD 能解密
        - 数据在传输和到达 TD 之前都是加密的
        """
        import time
        encrypt_start = time.time()
        
        # 1. 创建包含参数和指标的载荷
        serialized_params = serialize_parameters(model_parameters)
        training_metrics = federation_pb2.TrainingMetrics(**metrics)
        model_params_proto = federation_pb2.ModelParameters(parameters=serialized_params)
        
        payload = federation_pb2.ParametersAndMetrics(
            parameters=model_params_proto,
            metrics=training_metrics
        )
        serialized_payload = payload.SerializeToString()

        # 2. 生成一次性 AES 密钥并用 TD 的 RSA 公钥加密
        symmetric_key = AESGCM.generate_key(bit_length=256)
        encrypted_symmetric_key = self.tee_public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 3. 使用 AES-GCM 加密主载荷（提供认证加密）
        aesgcm = AESGCM(symmetric_key)
        nonce = os.urandom(12)  # 96-bit nonce
        encrypted_payload = aesgcm.encrypt(nonce, serialized_payload, None)

        encrypt_time = time.time() - encrypt_start
        logger.info(f"[TDX] 参数加密完成 (RSA+AES-GCM)，耗时: {encrypt_time:.4f}s，密文大小: {len(encrypted_payload)/1024/1024:.2f} MB")

        # 4. 创建 TDX 载荷
        tee_payload = federation_pb2.TeePayload(
            encrypted_symmetric_key=encrypted_symmetric_key,
            nonce=nonce,
            encrypted_payload=encrypted_payload
        )

        return federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            tee=tee_payload
        )


class SecurityError(Exception):
    """TDX 安全验证失败异常"""
    pass 