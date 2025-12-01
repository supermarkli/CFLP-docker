"""
高性能 Shamir 秘密共享实现

优化策略：
1. 多进程并行处理
2. 批量处理减少函数调用开销
3. 优化的多项式求值（Horner's method）
"""

import numpy as np
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

# 禁用 OpenMP 多线程，避免与多进程冲突
os.environ['OMP_NUM_THREADS'] = '1'

# 全局配置
_NUM_WORKERS = min(8, mp.cpu_count())


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
    """
    chunk_values, k, n, scaling_factor, prime, seed = args
    
    # 每个进程使用独立的随机种子
    rng = np.random.default_rng(seed)
    
    num_secrets = len(chunk_values)
    
    # 预分配结果：shares[party_idx] = list of y values
    shares = [[] for _ in range(n)]
    
    for i, val in enumerate(chunk_values):
        # 缩放并转换为整数
        secret = int(val * scaling_factor) % prime
        
        # 生成 k-1 个随机系数
        coeffs = [secret]
        for _ in range(k - 1):
            coeffs.append(int(rng.integers(0, min(prime, 2**62))))
        
        # 在 x = 1, 2, ..., n 处评估多项式
        for party_idx in range(n):
            x = party_idx + 1
            # Horner's method
            result = 0
            for c in reversed(coeffs[1:]):
                result = (result * x + c) % prime
            result = (result * x + coeffs[0]) % prime
            shares[party_idx].append(result)
    
    return shares


def _recover_chunk(args):
    """
    工作函数：恢复一个数据块的秘密
    使用 Python 大整数保证精度，循环展开优化速度
    """
    chunk_shares_by_party, k, prime, lagrange_coeffs, prime_half, scaling_factor, num_clients = args
    
    num_elements = len(chunk_shares_by_party[0])
    divisor = float(scaling_factor * num_clients)
    
    # 预转换拉格朗日系数为 Python int
    lc = [int(c) for c in lagrange_coeffs]
    
    # 预分配结果
    results = [0.0] * num_elements
    
    # 针对 k=3 的常见情况进行循环展开优化
    if k == 3:
        s0, s1, s2 = chunk_shares_by_party[0], chunk_shares_by_party[1], chunk_shares_by_party[2]
        c0, c1, c2 = lc[0], lc[1], lc[2]
        
        for i in range(num_elements):
            # 直接展开计算，减少循环开销
            result = (int(s0[i]) * c0 + int(s1[i]) * c1 + int(s2[i]) * c2) % prime
            if result > prime_half:
                result = result - prime
            results[i] = float(result) / divisor
    else:
        # 通用情况
        for elem_idx in range(num_elements):
            result = 0
            for party_idx in range(k):
                result += int(chunk_shares_by_party[party_idx][elem_idx]) * lc[party_idx]
            result = result % prime
            
            if result > prime_half:
                result = result - prime
            
            results[elem_idx] = float(result) / divisor
    
    return results


def fast_batch_share(
    values: np.ndarray,
    k: int,
    n: int,
    scaling_factor: int,
    prime: int,
    chunk_size: int = 50000
) -> List[bytes]:
    """
    高性能批量秘密共享（多进程并行）
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
                all_shares[party_idx].extend(chunk_shares[party_idx])
    else:
        # 单进程
        for args in task_args:
            chunk_shares = _share_chunk(args)
            for party_idx in range(n):
                all_shares[party_idx].extend(chunk_shares[party_idx])
    
    # 编码为二进制
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
    
    # 恢复秘密
    secrets = np.empty(num_secrets, dtype=object)
    for secret_idx in range(num_secrets):
        result = 0
        for i in range(k):
            y_i = int(shares[i][secret_idx])
            result = (result + y_i * lagrange_coeffs[i]) % prime
        secrets[secret_idx] = result
    
    return secrets


def fast_batch_recover(
    encoded_shares_by_party: List[List[int]],
    k: int,
    prime: int,
    scaling_factor: int,
    num_clients: int,
    chunk_size: int = 50000
) -> np.ndarray:
    """
    高性能批量秘密恢复（使用线程池，兼容 gRPC 环境）
    """
    from concurrent.futures import ThreadPoolExecutor
    
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
        (chunk, k, prime, lagrange_coeffs, prime_half, scaling_factor, num_clients)
        for chunk in chunks
    ]
    
    # 使用线程池（避免 gRPC fork 问题）
    all_results = []
    
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as executor:
            results = list(executor.map(_recover_chunk, task_args))
        for chunk_result in results:
            all_results.extend(chunk_result)
    else:
        for args in task_args:
            all_results.extend(_recover_chunk(args))
    
    return np.array(all_results, dtype=np.float64)
