import socket
import os
import pickle
import numpy as np
import hashlib
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logging.info("🚀 Enclave 脚本启动，正在导入库...")

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
logging.info("✅ 库导入完成。")

# --- 配置 ---
ENCLAVE_HOST = "0.0.0.0"
ENCLAVE_PORT = 8000
ENCLAVE_KEY_BITS = 2048

# --- Enclave 状态 ---
# 1. 生成RSA密钥对
logging.info("⏳ 正在生成RSA密钥对... 在enclave内部可能需要一些时间。")
private_key = rsa.generate_private_key(public_exponent=65537, key_size=ENCLAVE_KEY_BITS)
logging.info("✅ RSA密钥对已生成。")
public_key_pem = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
logging.info("✅ Enclave内部已成功生成RSA密钥对。")

def get_attestation_report():
    """为客户端生成真实的证明报告 (quote)，并将RSA公钥与之绑定。"""
    logging.info("Enclave: 正在生成真实的证明报告 (quote)...")
    
    report_data = hashlib.sha256(public_key_pem).digest()
    assert len(report_data) <= 64, "用户报告数据过长"

    try:
        with open("/dev/attestation/user_report_data", "wb") as f:
            f.write(report_data)
        
        with open("/dev/attestation/quote", "rb") as f:
            quote = f.read()
            
        logging.info(f"✅ 证明报告已成功生成 ({len(quote)} 字节)。")
        return public_key_pem, quote

    except FileNotFoundError:
        error_msg = "警告：无法找到 /dev/attestation/ 文件。此代码未在 Gramine-SGX 环境中运行。将使用临时密钥进行开发测试。"
        logging.warning(f"🟡 {error_msg}")
        return public_key_pem, pickle.dumps({"error": "DEV_MODE_NO_QUOTE"})
    except Exception as e:
        error_msg = f"生成证明报告时发生未知错误: {e}"
        logging.error(f"❌ {error_msg}")
        return None, pickle.dumps({"error": str(e)})


def handle_aggregation_request(data):
    """处理经混合加密的聚合请求。"""
    logging.info("Enclave: 已收到聚合请求。")
    try:
        updates = pickle.loads(data)
        
        aggregated_params = None
        aggregated_metrics = {
            'test_acc': 0, 'auc': 0, 'loss': 0, 
            'test_num': 0, 'train_num': 0
        }
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
            logging.info("✅ 聚合成功。")
            return pickle.dumps({"params": final_params, "metrics": final_metrics})
        else:
            raise ValueError("没有可聚合的数据。")

    except Exception as e:
        logging.error(f"❌ 聚合过程中出错: {e}")
        return pickle.dumps({"error": str(e)})

def main():
    """主函数，运行套接字服务器。"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许地址重用
    server.bind((ENCLAVE_HOST, ENCLAVE_PORT))
    server.listen(10)
    
    logging.info(f"🚀 Enclave聚合器正在监听 {ENCLAVE_HOST}:{ENCLAVE_PORT}")

    while True:
        connection, _ = server.accept()
        logging.info("🤝 已接受来自主服务器的连接。")
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
            logging.error(f"❌ 连接中出错: {e}")
        finally:
            connection.close()
            logging.info("连接已关闭。")

if __name__ == "__main__":
    main()