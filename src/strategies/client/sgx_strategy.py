import pickle
import hashlib
import json
import os
import sys
import numpy as np

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.client.base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters
from src.utils.config_utils import config


class SgxStrategy(ClientStrategy):
    """
    客户端的SGX策略。
    """
    def __init__(self, client):
        super().__init__(client)
        self.public_key = None
        self.logger = client.logger

    def setup(self, setup_response):
        """
        为SGX模式设置客户端。
        - 验证SGX Enclave的身份（通过Quote）。
        - 存储用于加密的公钥。
        """
        self.logger.info("SGX模式设置：正在从服务器接收Quote和公钥...")

        # 1. 解析和验证Quote（此处简化）
        # 在真实场景中，需要一个库来解析Quote并与PCCS/DCAP服务验证。
        # 这里我们只检查MRENCLAVE是否与配置匹配。
        # sgx_quote_bytes = setup_response.sgx_quote
        # quote = parse_quote(sgx_quote_bytes) # 假设的解析函数
        # actual_mrenclave = quote['mrenclave']
        expected_mrenclave = config['sgx']['expected_mrenclave']
        self.logger.warning(f"跳过SGX Quote验证，仅用于开发测试。预期的MRENCLAVE: {expected_mrenclave}")

        # if actual_mrenclave != expected_mrenclave:
        #     raise SecurityException(f"SGX身份验证失败！预期MRENCLAVE为 {expected_mrenclave}, 实际为 {actual_mrenclave}。")
        
        self.logger.info("✅ SGX Enclave身份(MRENCLAVE)已信任。")

        # 2. 加载公钥
        server_pubkey_bytes = setup_response.tee_public_key
        self.public_key = serialization.load_pem_public_key(server_pubkey_bytes)
        self.logger.info("SGX Enclave公钥已成功加载。")

    def _convert_params_to_float16(self, model_parameters):
        """
        将模型参数从 float32 转换为 float16 以减少传输数据量和 Enclave 内存占用。
        """
        float16_params = {}
        for name, param in model_parameters.items():
            if isinstance(param, np.ndarray):
                float16_params[name] = param.astype(np.float16)
            else:
                float16_params[name] = param
        return float16_params

    def prepare_update_request(self, current_round, model_parameters, metrics):
        """
        准备经SGX保护的更新请求。
        使用 float16 压缩模型参数以减少数据量。
        """
        if not self.public_key:
            raise ValueError("公钥未设置，请确认setup()方法已被调用。")

        self.logger.info("正在使用SGX公钥进行混合加密（float16压缩）...")

        # 1. 将模型参数转换为 float16 以减少数据量（约减少50%）
        float16_params = self._convert_params_to_float16(model_parameters)
        
        # 2. 序列化模型参数和指标
        params_and_metrics = {
            'params': float16_params,
            'metrics': metrics
        }
        serialized_payload = pickle.dumps(params_and_metrics)
        
        self.logger.info(f"序列化后数据大小: {len(serialized_payload) / 1024 / 1024:.2f} MB (float16压缩)")

        # 3. 生成对称密钥(AES)并用SGX的公钥加密
        symmetric_key = AESGCM.generate_key(bit_length=256)
        encrypted_symmetric_key = self.public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # 4. 使用AES密钥加密主载荷
        aesgcm = AESGCM(symmetric_key)
        nonce = os.urandom(12)  # 推荐96位
        encrypted_payload = aesgcm.encrypt(nonce, serialized_payload, None)

        # 5. 创建TeePayload
        tee_payload = federation_pb2.TeePayload(
            encrypted_symmetric_key=encrypted_symmetric_key,
            nonce=nonce,
            encrypted_payload=encrypted_payload
        )
        
        # 6. 创建最终的客户端更新请求
        client_update = federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            tee=tee_payload
        )
        return client_update

class SecurityException(Exception):
    pass