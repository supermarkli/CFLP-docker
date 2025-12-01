"""
高性能 Shamir 秘密共享实现 v2

优化策略：
1. 固定 8 字节编码（适用于 prime < 2^64）
2. NumPy 向量化聚合
3. 多进程并行处理
4. 循环展开优化
"""

import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
import struct

# 禁用 OpenMP 多线程，避免与多进程冲突
os.environ['OMP_NUM_THREADS'] = '1'

# 全局配置
_NUM_WORKERS = min(8, mp.cpu_count())

# 固定编码：每个份额 8 字节（uint64）
_SHARE_SIZE = 8


def _mod_inverse(a: int, p: int) -> int:
    """计算 a 在模 p 下的逆元（扩展欧几里得算法）"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a % p, p)
    return (x % p + p) % p


def _share_chunk(args):
    """
    多进程工作函数：为一个数据块生成秘密共享
    返回固定 8 字节编码的份额
    """
    chunk_values, k, n, scaling_factor, prime, seed = args
    
    # 每个进程使用独立的随机种子
    rng = np.random.default_rng(seed)
    
    num_secrets = len(chunk_values)
    
    # 预分配结果数组 (使用 uint64)
    shares = [np.zeros(num_secrets, dtype=np.uint64) for _ in range(n)]
    
    for i, val in enumerate(chunk_values):
        # 缩放并转换为整数
        secret = int(val * scaling_factor) % prime
        
        # 生成 k-1 个随机系数
        coeffs = [secret]
        for _ in range(k - 1):
            coeffs.append(int(rng.integers(0, min(prime, 2**62))))
        
        # 在 x = 1, 2, ..., n 处评估多项式 (Horner's method)
        for party_idx in range(n):
            x = party_idx + 1
            result = 0
            for c in reversed(coeffs[1:]):
                result = (result * x + c) % prime
            result = (result * x + coeffs[0]) % prime
            shares[party_idx][i] = result
    
    return shares


def _recover_chunk_vectorized(args):
    """
    向量化恢复秘密（使用 NumPy，适用于 prime < 2^62）
    """
    chunk_shares_by_party, k, prime, lagrange_coeffs, prime_half, scaling_factor, total_weight = args
    
    num_elements = len(chunk_shares_by_party[0])
    divisor = float(scaling_factor * total_weight)
    
    # 转换为 numpy 数组
    shares = [np.array(s, dtype=np.uint64) for s in chunk_shares_by_party]
    lc = np.array(lagrange_coeffs, dtype=np.uint64)
    
    # 向量化计算 - 使用 Python 对象类型以支持大整数
    # 但对于 2^61-1 素数，uint64 乘法可能溢出，需要用 Python int
    results = np.zeros(num_elements, dtype=np.float64)
    
    if k == 3:
        # k=3 循环展开优化
        s0, s1, s2 = shares[0], shares[1], shares[2]
        c0, c1, c2 = int(lc[0]), int(lc[1]), int(lc[2])
        
        for i in range(num_elements):
            result = (int(s0[i]) * c0 + int(s1[i]) * c1 + int(s2[i]) * c2) % prime
            if result > prime_half:
                result = result - prime
            results[i] = float(result) / divisor
    else:
        for i in range(num_elements):
            result = 0
            for j in range(k):
                result += int(shares[j][i]) * int(lc[j])
            result = result % prime
            if result > prime_half:
                result = result - prime
            results[i] = float(result) / divisor
    
    return results.tolist()


def fast_batch_share(
    values: np.ndarray,
    k: int,
    n: int,
    scaling_factor: int,
    prime: int,
    chunk_size: int = 50000
) -> List[bytes]:
    """
    高性能批量秘密共享（多进程并行 + 固定 8 字节编码）
    """
    flat = values.flatten().astype(np.float64)
    total = len(flat)
    
    # 分块
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append(flat[i:i + chunk_size])
    
    # 准备任务参数（每个任务使用不同的随机种子）
    base_seed = np.random.randint(0, 2**31)
    task_args = [
        (chunk, k, n, scaling_factor, prime, base_seed + i)
        for i, chunk in enumerate(chunks)
    ]
    
    # 并行执行
    all_shares = [[] for _ in range(n)]
    
    if len(chunks) > 1 and _NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as executor:
            results = list(executor.map(_share_chunk, task_args))
        
        for chunk_shares in results:
            for party_idx in range(n):
                all_shares[party_idx].append(chunk_shares[party_idx])
    else:
        for args in task_args:
            chunk_shares = _share_chunk(args)
            for party_idx in range(n):
                all_shares[party_idx].append(chunk_shares[party_idx])
    
    # 合并并编码为固定 8 字节格式
    encoded = []
    for party_chunks in all_shares:
        # 合并所有块
        all_values = np.concatenate(party_chunks)
        # 直接转换为字节（固定 8 字节 big-endian）
        encoded.append(all_values.astype('>u8').tobytes())
    
    return encoded


def fast_batch_share_variable(
    values: np.ndarray,
    k: int,
    n: int,
    scaling_factor: int,
    prime: int,
    chunk_size: int = 50000
) -> List[bytes]:
    """
    变长编码版本（兼容旧格式）
    """
    flat = values.flatten().astype(np.float64)
    total = len(flat)
    
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append(flat[i:i + chunk_size])
    
    base_seed = np.random.randint(0, 2**31)
    task_args = [
        (chunk, k, n, scaling_factor, prime, base_seed + i)
        for i, chunk in enumerate(chunks)
    ]
    
    all_shares = [[] for _ in range(n)]
    
    if len(chunks) > 1 and _NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as executor:
            results = list(executor.map(_share_chunk, task_args))
        
        for chunk_shares in results:
            for party_idx in range(n):
                all_shares[party_idx].extend(chunk_shares[party_idx])
    else:
        for args in task_args:
            chunk_shares = _share_chunk(args)
            for party_idx in range(n):
                all_shares[party_idx].extend(chunk_shares[party_idx])
    
    # 变长编码
    encoded = []
    for party_shares in all_shares:
        byte_parts = []
        for y_val in party_shares:
            y_int = int(y_val)
            if y_int < 0:
                y_int = y_int % prime
            byte_len = max(1, (y_int.bit_length() + 7) // 8)
            y_bytes = y_int.to_bytes(byte_len, byteorder='big', signed=False)
            byte_parts.append(len(y_bytes).to_bytes(2, 'big') + y_bytes)
        encoded.append(b''.join(byte_parts))
    
    return encoded


def decode_fixed_shares(data_bytes: bytes) -> np.ndarray:
    """解码固定 8 字节格式的份额"""
    return np.frombuffer(data_bytes, dtype='>u8')


def decode_variable_shares(data_bytes: bytes) -> List[int]:
    """解码变长格式的份额"""
    y_values = []
    offset = 0
    data_len = len(data_bytes)
    while offset < data_len:
        length = int.from_bytes(data_bytes[offset:offset + 2], 'big')
        offset += 2
        y_bytes = data_bytes[offset:offset + length]
        y_val = int.from_bytes(y_bytes, byteorder='big', signed=False)
        y_values.append(y_val)
        offset += length
    return y_values


def vectorized_secret_to_shares(
    secrets: np.ndarray,
    k: int,
    n: int,
    prime: int
) -> List[np.ndarray]:
    """
    向量化的秘密共享生成（用于小数据量）
    """
    num_secrets = len(secrets)
    rng = np.random.default_rng()
    
    shares = [np.empty(num_secrets, dtype=object) for _ in range(n)]
    
    for i, secret in enumerate(secrets):
        s = int(secret) % prime
        coeffs = [s]
        for _ in range(k - 1):
            coeffs.append(int(rng.integers(0, min(prime, 2**62))))
        
        for party_idx in range(n):
            x = party_idx + 1
            result = 0
            for c in reversed(coeffs[1:]):
                result = (result * x + c) % prime
            result = (result * x + coeffs[0]) % prime
            shares[party_idx][i] = result
    
    return shares


def vectorized_shares_to_secret(
    shares_by_party: List[np.ndarray],
    k: int,
    prime: int
) -> np.ndarray:
    """
    向量化的秘密恢复（拉格朗日插值）
    """
    shares = shares_by_party[:k]
    num_secrets = len(shares[0])
    x_coords = list(range(1, k + 1))
    
    lagrange_coeffs = []
    for i in range(k):
        xi = x_coords[i]
        numerator = 1
        denominator = 1
        for j in range(k):
            if i != j:
                xj = x_coords[j]
                numerator = (numerator * (-xj)) % prime
                denominator = (denominator * (xi - xj)) % prime
        coeff = (numerator * _mod_inverse(denominator, prime)) % prime
        lagrange_coeffs.append(coeff)
    
    secrets = np.empty(num_secrets, dtype=object)
    for secret_idx in range(num_secrets):
        result = 0
        for i in range(k):
            y_i = int(shares[i][secret_idx])
            result = (result + y_i * lagrange_coeffs[i]) % prime
        secrets[secret_idx] = result
    
    return secrets


def aggregate_shares_vectorized(
    all_client_shares: List[np.ndarray],
    prime: int
) -> np.ndarray:
    """
    向量化聚合份额（NumPy 优化）
    
    Args:
        all_client_shares: List of numpy arrays, each from one client
        prime: Prime modulus
    
    Returns:
        Summed shares as numpy array
    """
    # 堆叠所有客户端的份额
    stacked = np.stack(all_client_shares)
    # 使用 Python 对象类型进行精确模运算
    # 但对于 2^61-1 素数，直接用 uint64 求和再取模更快
    summed = np.sum(stacked.astype(np.uint64), axis=0)
    # 取模
    return summed % prime


def fast_batch_recover(
    encoded_shares_by_party: List[List[int]],
    k: int,
    prime: int,
    scaling_factor: int,
    total_weight: float,
    chunk_size: int = 50000
) -> np.ndarray:
    """
    高性能批量秘密恢复（使用线程池，兼容 gRPC 环境）
    
    Args:
        encoded_shares_by_party: 每个 party 的份额列表
        k: Shamir 阈值
        prime: 素数模
        scaling_factor: 缩放因子
        total_weight: 总权重（用于加权平均，可以是 num_clients 或 total_data_size）
        chunk_size: 分块大小
    """
    num_elements = len(encoded_shares_by_party[0])
    prime_half = prime // 2
    x_coords = list(range(1, k + 1))
    
    # 预计算拉格朗日系数
    lagrange_coeffs = []
    for i in range(k):
        xi = x_coords[i]
        numerator = 1
        denominator = 1
        for j in range(k):
            if i != j:
                xj = x_coords[j]
                numerator = (numerator * (-xj)) % prime
                denominator = (denominator * (xi - xj)) % prime
        coeff = (numerator * _mod_inverse(denominator, prime)) % prime
        lagrange_coeffs.append(coeff)
    
    # 分块
    chunks = []
    for i in range(0, num_elements, chunk_size):
        end = min(i + chunk_size, num_elements)
        chunk_shares = [party_shares[i:end] for party_shares in encoded_shares_by_party[:k]]
        chunks.append(chunk_shares)
    
    # 准备任务参数
    task_args = [
        (chunk, k, prime, lagrange_coeffs, prime_half, scaling_factor, total_weight)
        for chunk in chunks
    ]
    
    # 使用线程池（避免 gRPC fork 问题）
    all_results = []
    
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as executor:
            results = list(executor.map(_recover_chunk_vectorized, task_args))
        for chunk_result in results:
            all_results.extend(chunk_result)
    else:
        for args in task_args:
            all_results.extend(_recover_chunk_vectorized(args))
    
    return np.array(all_results, dtype=np.float64)
