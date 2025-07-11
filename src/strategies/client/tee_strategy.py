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
    def __init__(self, client_instance, tee_attestation_report_bytes, tee_public_key_bytes):
        super().__init__(client_instance)

        # 1. 验证TEE身份
        report = json.loads(tee_attestation_report_bytes.decode('utf-8'))
        actual_mrenclave = report.get("mrenclave")
        expected_mrenclave = config['tee']['expected_mrenclave']
        if actual_mrenclave == expected_mrenclave:
            logger.info(f"客户端 {self.client.client_id} 的 TEE 策略初始化成功：身份验证通过！")
        else:
            raise Exception(f"TEE身份验证失败！预期MRENCLAVE为{expected_mrenclave}，实际为{actual_mrenclave}。")
        
        # 2. 加载TEE公钥
        self.tee_public_key = serialization.load_pem_public_key(tee_public_key_bytes)
        logger.info("TEE 策略已加载公钥。")

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建在TEE模式下加密的参数更新消息"""
        # 1. 创建包含明文参数和指标的载荷
        serialized_params = serialize_parameters(model_parameters)
        training_metrics = federation_pb2.TrainingMetrics(**metrics)
        model_params_proto = federation_pb2.ModelParameters(parameters=serialized_params)
        
        payload = federation_pb2.ParametersAndMetrics(
            parameters=model_params_proto,
            metrics=training_metrics
        )
        serialized_payload = payload.SerializeToString()

        # 2. 生成一次性对称密钥(AES)并用TEE的RSA公钥加密它
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

        # 4. 创建 TEE 载荷
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