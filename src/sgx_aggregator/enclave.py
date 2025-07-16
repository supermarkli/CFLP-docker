import socket
import os
import pickle
import numpy as np
import json
import hashlib
import sys

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 引入我们项目中的 protobuf 定义，以便能够反序列化
# 注意：这需要在 enclave 的环境中安装 protobuf 和 grpcio-tools
# from src.grpc.generated import federation_pb2 # 不再需要
# from src.utils.parameter_utils import deserialize_parameters # 不再需要

# --- 配置 ---
SOCKET_PATH = "/tmp/ipc/aggregator.sock"
ENCLAVE_KEY_BITS = 2048

# --- Enclave 状态 ---
# 1. 生成RSA密钥对，而不是Paillier
private_key = rsa.generate_private_key(public_exponent=65537, key_size=ENCLAVE_KEY_BITS)
public_key_pem = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
print("✅ Enclave内部已成功生成RSA密钥对。")

def get_attestation_report():
    """为客户端生成真实的证明报告 (quote)，并将RSA公钥与之绑定。"""
    print("Enclave: 正在生成真实的证明报告 (quote)...")
    
    # 将PEM格式的公钥哈希作为报告数据
    report_data = hashlib.sha256(public_key_pem).digest()
    assert len(report_data) <= 64, "用户报告数据过长"

    try:
        with open("/dev/attestation/user_report_data", "wb") as f:
            f.write(report_data)
        
        with open("/dev/attestation/quote", "rb") as f:
            quote = f.read()
            
        print(f"✅ 证明报告已成功生成 ({len(quote)} 字节)。")
        # 返回PEM格式的公钥和quote
        return public_key_pem, quote

    except FileNotFoundError:
        error_msg = "警告：无法找到 /dev/attestation/ 文件。此代码未在 Gramine-SGX 环境中运行。将使用临时密钥进行开发测试。"
        print(f"🟡 {error_msg}")
        # 在非SGX环境中，我们仍然返回公钥和一个特殊的“quote”来表示这是开发模式
        # 这允许服务器和客户端继续运行，但会跳过真实的身份验证
        return public_key_pem, pickle.dumps({"error": "DEV_MODE_NO_QUOTE"})
    except Exception as e:
        error_msg = f"生成证明报告时发生未知错误: {e}"
        print(f"❌ {error_msg}")
        return None, pickle.dumps({"error": str(e)})


def handle_aggregation_request(data):
    """处理经混合加密的聚合请求。"""
    print("Enclave: 已收到聚合请求。")
    try:
        updates = pickle.loads(data)
        
        aggregated_params = None
        # 这个字典现在只包含从客户端指标中累加的键
        aggregated_metrics = {
            'test_acc': 0, 'auc': 0, 'loss': 0, 
            'test_num': 0, 'train_num': 0
        }
        # 这个变量专门用于累加参数加权所需的样本数
        total_samples_for_params = 0
        
        for payload_tuple, num_samples in updates:
            encrypted_key, nonce, encrypted_data = payload_tuple

            # 2. 用RSA私钥解密AES密钥
            symmetric_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )

            # 3. 用AES密钥解密主载荷
            aesgcm = AESGCM(symmetric_key)
            decrypted_payload_bytes = aesgcm.decrypt(nonce, encrypted_data, None)
            
            # 4. 反序列化明文参数和指标
            params_and_metrics = pickle.loads(decrypted_payload_bytes)
            
            decrypted_params = params_and_metrics['params']
            metrics = params_and_metrics['metrics']

            # 5. 累加参数和指标
            total_samples_for_params += num_samples
            if aggregated_params is None:
                aggregated_params = {name: decrypted_params[name] * num_samples for name in decrypted_params}
            else:
                for name in aggregated_params:
                    aggregated_params[name] += decrypted_params[name] * num_samples
            
            # 使用从客户端传来的测试样本数进行加权
            test_num = metrics.get('test_num', 0)
            aggregated_metrics['test_acc'] += metrics.get('test_acc', 0) # 客户端传来的就是正确预测数
            aggregated_metrics['auc'] += metrics.get('auc', 0) * test_num
            aggregated_metrics['loss'] += metrics.get('loss', 0) # loss已经是sum, 不需要乘以样本数
            aggregated_metrics['test_num'] += test_num
            aggregated_metrics['train_num'] += metrics.get('train_num', 0)

        # 计算加权平均值
        if aggregated_params and total_samples_for_params > 0:
            final_params = {name: params / total_samples_for_params for name, params in aggregated_params.items()}
            
            total_test_num = aggregated_metrics['test_num']
            total_train_num = aggregated_metrics['train_num']
            # 构建最终返回的指标字典
            final_metrics = {
                'test_acc': aggregated_metrics['test_acc'] / total_test_num if total_test_num > 0 else 0,
                'auc': aggregated_metrics['auc'] / total_test_num if total_test_num > 0 else 0,
                'loss': aggregated_metrics['loss'] / total_train_num if total_train_num > 0 else 0,
                'total_samples': total_samples_for_params 
            }
            print("✅ 聚合成功。")
            return pickle.dumps({"params": final_params, "metrics": final_metrics})
        else:
            raise ValueError("没有可聚合的数据。")

    except Exception as e:
        print(f"❌ 聚合过程中出错: {e}")
        return pickle.dumps({"error": str(e)})

def main():
    """主函数，运行套接字服务器。"""
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    server.bind(SOCKET_PATH)
    server.listen(10)
    
    print(f"🚀 Enclave聚合器正在监听 {SOCKET_PATH}")

    while True:
        connection, _ = server.accept()
        print("🤝 已接受来自主服务器的连接。")
        try:
            command = connection.recv(1024).decode().strip()
            
            if command == "GET_ATTESTATION":
                pubkey_bytes, quote = get_attestation_report()
                if pubkey_bytes:
                    connection.sendall(pickle.dumps((pubkey_bytes, quote)))
                
            elif command == "AGGREGATE":
                connection.sendall(b"READY")
                data = b""
                while True:
                    part = connection.recv(4096)
                    if not part: break
                    data += part
                
                result = handle_aggregation_request(data)
                connection.sendall(result)

            else:
                connection.sendall(b"Unknown Command")

        except Exception as e:
            print(f"❌ 连接中出错: {e}")
        finally:
            connection.close()
            print("连接已关闭。")

if __name__ == "__main__":
    # 确保enclave可以找到项目模块
    # sys.path.append('/app') # 不再需要，因为依赖已解耦
    main()