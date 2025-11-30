import socket
import os
import pickle
import numpy as np
import hashlib
import logging
import psutil
import time

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


def recv_exact(connection, length):
    """
    精确接收指定长度的数据。
    使用预分配的 bytearray 避免内存碎片和重复分配。
    """
    buffer = bytearray(length)
    view = memoryview(buffer)
    received = 0
    while received < length:
        chunk_size = connection.recv_into(view[received:], min(65536, length - received))
        if chunk_size == 0:
            raise ConnectionError("连接意外关闭")
        received += chunk_size
    return bytes(buffer)


def process_single_client(encrypted_key, nonce, encrypted_data, num_samples, 
                          aggregated_params, aggregated_metrics, total_samples):
    """
    处理单个客户端的加密数据，解密并累加到聚合结果中。
    输入数据是 float16 格式，聚合结果也用 float16 存储以节省内存。
    使用原地操作 (in-place) 尽可能减少内存分配。
    """
    # 1. 用RSA私钥解密AES密钥
    symmetric_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

    # 2. 用AES密钥解密主载荷
    aesgcm = AESGCM(symmetric_key)
    decrypted_payload_bytes = aesgcm.decrypt(nonce, encrypted_data, None)
    
    # 3. 反序列化明文参数和指标（参数已经是 float16）
    params_and_metrics = pickle.loads(decrypted_payload_bytes)
    # 立即释放解密后的字节数据
    del decrypted_payload_bytes
    
    decrypted_params = params_and_metrics['params']
    metrics = params_and_metrics['metrics']
    # 释放中间对象
    del params_and_metrics

    # 4. 累加参数（使用 float16 存储，原地操作减少内存）
    total_samples += num_samples
    if aggregated_params is None:
        # 第一个客户端：直接使用其参数作为初始聚合结果（避免复制）
        aggregated_params = decrypted_params  # 直接接管引用，不复制
    else:
        # 后续客户端：原地累加
        for name in aggregated_params:
            # 使用 numpy 的原地操作，避免创建新数组
            # 先转为 float32 视图进行计算（避免 float16 溢出）
            np.add(aggregated_params[name], decrypted_params[name], 
                   out=aggregated_params[name], casting='same_kind')
        # 释放当前客户端的参数
        del decrypted_params

    # 5. 累加指标
    test_num = metrics.get('test_num', 0)
    aggregated_metrics['test_acc'] += metrics.get('test_acc', 0)
    aggregated_metrics['auc'] += metrics.get('auc', 0) * test_num
    aggregated_metrics['loss'] += metrics.get('loss', 0)
    aggregated_metrics['test_num'] += test_num
    aggregated_metrics['train_num'] += metrics.get('train_num', 0)
    
    return aggregated_params, aggregated_metrics, total_samples


def handle_streaming_aggregation(connection):
    """
    流式处理聚合请求：逐个接收客户端数据，解密并累加，避免内存峰值。
    """
    logging.info("Enclave: 开始流式聚合处理。")
    
    # 资源监控起始点
    process = psutil.Process()
    start_cpu_time = process.cpu_times().user + process.cpu_times().system
    start_memory = process.memory_info().rss
    
    try:
        # 1. 读取客户端数量 (4字节)
        num_clients_bytes = recv_exact(connection, 4)
        num_clients = int.from_bytes(num_clients_bytes, byteorder='big')
        logging.info(f"📊 准备接收 {num_clients} 个客户端的数据...")

        # 初始化聚合状态
        aggregated_params = None
        aggregated_metrics = {
            'test_acc': 0, 'auc': 0, 'loss': 0, 
            'test_num': 0, 'train_num': 0
        }
        total_samples = 0

        # 2. 逐个接收并处理客户端数据
        for i in range(num_clients):
            # 读取该客户端数据长度 (8字节)
            length_bytes = recv_exact(connection, 8)
            data_length = int.from_bytes(length_bytes, byteorder='big')
            
            logging.info(f"📥 接收客户端 {i+1}/{num_clients} 的数据 ({data_length / 1024 / 1024:.2f} MB)...")
            
            # 读取该客户端的数据
            client_data = recv_exact(connection, data_length)
            
            # 反序列化
            payload_tuple, num_samples = pickle.loads(client_data)
            # 立即释放原始数据
            del client_data
            
            encrypted_key, nonce, encrypted_data = payload_tuple
            del payload_tuple
            
            # 处理该客户端数据（解密、累加、释放）
            aggregated_params, aggregated_metrics, total_samples = process_single_client(
                encrypted_key, nonce, encrypted_data, num_samples,
                aggregated_params, aggregated_metrics, total_samples
            )
            
            # 释放加密数据
            del encrypted_key, nonce, encrypted_data
            
            logging.info(f"✅ 客户端 {i+1}/{num_clients} 处理完成。")

        # 3. 计算最终结果
        if aggregated_params and total_samples > 0:
            # 将 float16 聚合结果转为 float32，并计算平均值
            # 注意：这里 aggregated_params 存储的是各客户端参数的累加和
            # 需要除以客户端数量（num_clients）来得到平均值
            final_params = {}
            for name, params in aggregated_params.items():
                # 转为 float32 并除以客户端数量
                final_params[name] = params.astype(np.float32) / num_clients
            
            # 释放聚合结果
            del aggregated_params
            
            total_test_num = aggregated_metrics['test_num']
            total_train_num = aggregated_metrics['train_num']
            
            # 资源监控结束点
            end_cpu_time = process.cpu_times().user + process.cpu_times().system
            current_memory = process.memory_info().rss
            
            cpu_time_used = end_cpu_time - start_cpu_time
            memory_usage = current_memory
            
            # 构建最终返回的指标字典
            final_metrics = {
                'test_acc': aggregated_metrics['test_acc'] / total_test_num if total_test_num > 0 else 0,
                'auc': aggregated_metrics['auc'] / total_test_num if total_test_num > 0 else 0,
                'loss': aggregated_metrics['loss'] / total_train_num if total_train_num > 0 else 0,
                'total_samples': total_samples,
                'server_cpu_time': cpu_time_used,
                'server_memory_usage': memory_usage
            }
            logging.info(f"✅ 流式聚合成功。CPU耗时: {cpu_time_used:.4f}s, 内存使用: {memory_usage / 1024 / 1024:.2f} MB")
            return pickle.dumps({"params": final_params, "metrics": final_metrics})
        else:
            raise ValueError("没有可聚合的数据。")

    except Exception as e:
        logging.error(f"❌ 流式聚合过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return pickle.dumps({"error": str(e)})


def main():
    """主函数，运行套接字服务器。"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                
            elif command == "AGGREGATE_STREAM":
                # 新的流式聚合命令
                connection.sendall(b"READY")
                
                result = handle_streaming_aggregation(connection)
                
                # 发送结果 (长度前缀 + 数据)
                result_length = len(result)
                connection.sendall(result_length.to_bytes(8, byteorder='big'))
                connection.sendall(result)

            else:
                connection.sendall(b"Unknown Command")

        except Exception as e:
            logging.error(f"❌ 连接中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            connection.close()
            logging.info("连接已关闭。")

if __name__ == "__main__":
    main()
