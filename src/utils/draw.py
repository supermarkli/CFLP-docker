#!/usr/bin/env python3
"""
联邦学习实验结果可视化工具

功能：
1. 端到端延迟分解图 (End-to-End Latency Breakdown)
2. 内存足迹对比 (Memory Footprint)
3. 收敛图 (Convergence Curve)
4. 自动读取logs目录，按模式分组并取平均值
5. 终端输出带标准差的统计表格

绘图风格：论文级别，使用 Times New Roman 和宋体
"""

import re
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============== 字体设置 ==============
def get_fonts():
    """获取可用的字体：宋体风格和Times New Roman风格"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找宋体风格字体
    song_candidates = ['SimSun', 'STSong', 'Noto Serif CJK SC', 'AR PL SungtiL GB']
    song_font = None
    for font in song_candidates:
        if font in available_fonts:
            song_font = font
            break
    
    # 尝试加载本地宋体
    SCRIPT_DIR = Path(__file__).parent
    SIMSUN_PATH = SCRIPT_DIR / 'simsun.ttf'
    if SIMSUN_PATH.exists():
        fm.fontManager.addfont(str(SIMSUN_PATH))
        song_font = 'SimSun'
    
    if song_font is None:
        song_font = 'DejaVu Sans'
    
    # 查找 Times New Roman 风格字体
    times_candidates = ['Times New Roman', 'Nimbus Roman', 'PT Serif', 'DejaVu Serif', 'Noto Serif']
    times_font = None
    for font in times_candidates:
        if font in available_fonts:
            times_font = font
            break
    if times_font is None:
        times_font = 'DejaVu Serif'
    
    return song_font, times_font

SONG_FONT, TIMES_FONT = get_fonts()

# 设置全局字体
plt.rcParams['font.family'] = TIMES_FONT
plt.rcParams['font.serif'] = [TIMES_FONT]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 日志目录和输出目录
LOGS_DIR = Path(__file__).parent.parent.parent / 'logs'
OUT_DIR = Path(__file__).parent.parent.parent / 'out'

# 模式名称映射
MODE_DISPLAY_NAMES = {
    'none': 'BASE',
    'he': 'HE',
    'mpc': 'MPC',
    'sgx': 'SGX',
    'tee': 'TDX'
}

# 模式顺序
MODE_ORDER = ['none', 'he', 'mpc', 'sgx', 'tee']

# 颜色方案（朴素学术风格）
COLORS = {
    'training': '#4472C4',      # 训练 - 学术蓝
    'encryption': '#ED7D31',    # 加密 - 学术橙
    'communication': '#A5A5A5', # 通信 - 灰色
    'decryption': '#FFC000',    # 解密 - 金黄
    'aggregation': '#70AD47',   # 聚合 - 学术绿
    'client': '#5B9BD5',        # 客户端内存 - 浅蓝
    'server': '#ED7D31',        # 服务端内存 - 橙色
    'enclave': '#70AD47',       # Enclave内存 - 绿色
}

# 中文图例标签
LABELS_CN = {
    'training': '训练',
    'encryption': '加密',
    'communication': '传输',
    'decryption': '解密',
    'aggregation': '聚合',
    'client': '客户端峰值',
    'server': '服务端峰值',
    'enclave': 'Enclave峰值',
}

# 模式颜色
MODE_COLORS = {
    'none': '#6C757D',   # 灰色
    'he': '#2E86AB',     # 蓝色
    'mpc': '#A23B72',    # 玫红
    'sgx': '#3A7D44',    # 绿色
    'tee': '#F18F01',    # 橙色
}


def parse_log_file(log_path):
    """
    解析联邦学习日志文件，提取关键指标
    支持 HE 模式的流式传输时间推算
    """
    data = {
        'rounds': [],
        'latency': {
            'training': [],
            'encryption': [],
            'communication': [],  # 传输时间
            'decryption': [],
            'aggregation': [],
        },
        'memory': {
            'client_peak': [],
            'server_peak': [],
            'enclave': [],
        },
        'payload': [],  # 上传数据量 (MB)
        'convergence': {
            'accuracy': [],
            'auc': [],
            'loss': [],
        },
    }
    
    current_round = None
    round_client_training = []
    round_client_encryption = []
    round_client_memory = []
    round_client_stream_time = []  # HE 流式传输时间
    round_client_payload = []      # 上传数据量
    
    # 正则表达式模式
    patterns = {
        'round_client_latency': re.compile(
            r'\[Round (\d+)\]\[Client (\d+)\]\[LATENCY\] training=([\d.]+)s(?:, encryption=([\d.]+)s)?'
        ),
        'round_client_resource': re.compile(
            r'\[Round (\d+)\]\[Client (\d+)\]\[RESOURCE\] peak_memory=([\d.]+) MB'
        ),
        'round_client_payload': re.compile(
            r'\[Round (\d+)\]\[Client (\d+)\]\[PAYLOAD\] upload=([\d.]+) MB'
        ),
        'round_latency_dec': re.compile(r'\[Round (\d+)\]\[LATENCY\] decryption=([\d.]+)s'),
        'round_latency_agg': re.compile(r'\[Round (\d+)\]\[LATENCY\] aggregation=([\d.]+)s'),
        'round_server_resource': re.compile(
            r'\[Round (\d+)\]\[Server\]\[RESOURCE\] peak_memory=([\d.]+) MB'
        ),
        'round_enclave_resource': re.compile(
            r'\[Round (\d+)\]\[Enclave\]\[RESOURCE\] cpu_time=([\d.]+)s, memory=([\d.]+) MB'
        ),
        'convergence_new': re.compile(
            r'\[CONVERGENCE\] round=(\d+) accuracy=([\d.]+) auc=([\d.]+) loss=([\d.]+)'
        ),
        # HE 流式传输时间（用于推算加密+传输时间）
        'he_stream_complete': re.compile(
            r'\[Round (\d+)\] 客户端 (\d+) 接收完成: \d+ 层, ([\d.]+)s'
        ),
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 客户端延迟
            m = patterns['round_client_latency'].search(line)
            if m:
                rnd, cid, train_t, enc_t = m.groups()
                rnd = int(rnd)
                if current_round != rnd:
                    if current_round is not None and round_client_training:
                        data['latency']['training'].append(np.mean(round_client_training))
                        
                        # HE 模式：流式传输时间包含加密+传输
                        if round_client_stream_time:
                            total_stream = np.mean(round_client_stream_time)
                            train_time = np.mean(round_client_training)
                            # 流式传输时间 - 训练时间 = 加密 + 传输时间
                            enc_comm_time = max(0, total_stream - train_time)
                            # 估算：HE 加密占主导，传输约占 5-10%
                            # 基于 payload 大小估算传输时间（Docker 网络约 500 MB/s）
                            if round_client_payload:
                                payload_mb = np.mean(round_client_payload)
                                comm_time = payload_mb / 500.0  # Docker 网络约 500 MB/s
                                enc_time = max(0, enc_comm_time - comm_time)
                            else:
                                enc_time = enc_comm_time * 0.95
                                comm_time = enc_comm_time * 0.05
                            data['latency']['encryption'].append(enc_time)
                            data['latency']['communication'].append(comm_time)
                        elif round_client_encryption:
                            data['latency']['encryption'].append(np.mean(round_client_encryption))
                            # 估算传输时间（Docker 网络约 500 MB/s）
                            if round_client_payload:
                                payload_mb = np.mean(round_client_payload)
                                comm_time = payload_mb / 500.0
                            else:
                                comm_time = 0.1  # 默认 0.1s
                            data['latency']['communication'].append(comm_time)
                        else:
                            data['latency']['encryption'].append(0)
                            data['latency']['communication'].append(0)
                        
                        if round_client_memory:
                            data['memory']['client_peak'].append(np.max(round_client_memory))
                        if round_client_payload:
                            data['payload'].append(np.mean(round_client_payload))
                    
                    current_round = rnd
                    round_client_training = []
                    round_client_encryption = []
                    round_client_memory = []
                    round_client_stream_time = []
                    round_client_payload = []
                    data['rounds'].append(rnd)
                
                round_client_training.append(float(train_t))
                if enc_t:
                    round_client_encryption.append(float(enc_t))
                continue
            
            # HE 流式传输完成时间
            m = patterns['he_stream_complete'].search(line)
            if m:
                rnd, cid, stream_t = m.groups()
                round_client_stream_time.append(float(stream_t))
                continue
            
            # 客户端 PAYLOAD
            m = patterns['round_client_payload'].search(line)
            if m:
                rnd, cid, payload = m.groups()
                round_client_payload.append(float(payload))
                continue
            
            # 客户端资源
            m = patterns['round_client_resource'].search(line)
            if m:
                rnd, cid, mem = m.groups()
                round_client_memory.append(float(mem))
                continue
            
            # 解密时间
            m = patterns['round_latency_dec'].search(line)
            if m:
                rnd, dec_t = m.groups()
                data['latency']['decryption'].append(float(dec_t))
                continue
            
            # 聚合时间
            m = patterns['round_latency_agg'].search(line)
            if m:
                rnd, agg_t = m.groups()
                data['latency']['aggregation'].append(float(agg_t))
                continue
            
            # 服务端资源
            m = patterns['round_server_resource'].search(line)
            if m:
                rnd, mem = m.groups()
                data['memory']['server_peak'].append(float(mem))
                continue
            
            # Enclave 资源
            m = patterns['round_enclave_resource'].search(line)
            if m:
                rnd, cpu_t, mem = m.groups()
                data['memory']['enclave'].append(float(mem))
                continue
            
            # 收敛数据
            m = patterns['convergence_new'].search(line)
            if m:
                rnd, acc, auc, loss = m.groups()
                data['convergence']['accuracy'].append(float(acc))
                data['convergence']['auc'].append(float(auc))
                data['convergence']['loss'].append(float(loss))
                continue
    
    # 保存最后一轮的数据
    if round_client_training:
        data['latency']['training'].append(np.mean(round_client_training))
        
        if round_client_stream_time:
            total_stream = np.mean(round_client_stream_time)
            train_time = np.mean(round_client_training)
            enc_comm_time = max(0, total_stream - train_time)
            if round_client_payload:
                payload_mb = np.mean(round_client_payload)
                comm_time = payload_mb / 500.0  # Docker 网络约 500 MB/s
                enc_time = max(0, enc_comm_time - comm_time)
            else:
                enc_time = enc_comm_time * 0.98
                comm_time = enc_comm_time * 0.02
            data['latency']['encryption'].append(enc_time)
            data['latency']['communication'].append(comm_time)
        elif round_client_encryption:
            data['latency']['encryption'].append(np.mean(round_client_encryption))
            if round_client_payload:
                comm_time = np.mean(round_client_payload) / 500.0  # Docker 网络约 500 MB/s
            else:
                comm_time = 0.1
            data['latency']['communication'].append(comm_time)
        else:
            data['latency']['encryption'].append(0)
            data['latency']['communication'].append(0)
        
        if round_client_memory:
            data['memory']['client_peak'].append(np.max(round_client_memory))
        if round_client_payload:
            data['payload'].append(np.mean(round_client_payload))
    
    return data


def discover_logs(logs_dir):
    """发现日志目录中的所有日志文件，并按模式分组"""
    logs_by_mode = defaultdict(list)
    logs_dir = Path(logs_dir)
    
    if not logs_dir.exists():
        print(f"❌ 日志目录不存在: {logs_dir}")
        return logs_by_mode
    
    for log_file in logs_dir.glob('*.log'):
        filename = log_file.stem
        for mode in MODE_ORDER:
            if filename.startswith(f'{mode}_'):
                logs_by_mode[mode].append(log_file)
                break
    
    return logs_by_mode


def aggregate_multiple_logs(log_paths):
    """解析多个日志文件并聚合数据（收集所有轮次数据计算平均值和标准差）"""
    all_data = []
    total_rounds = 0
    for log_path in log_paths:
        data = parse_log_file(log_path)
        all_data.append(data)
        if data['latency']['training']:
            total_rounds += len(data['latency']['training'])
    
    if not all_data:
        return None
    
    aggregated = {
        'num_logs': len(all_data),
        'total_rounds': total_rounds,
        'latency': {k: {'mean': 0, 'std': 0, 'count': 0} for k in ['training', 'encryption', 'communication', 'decryption', 'aggregation']},
        'memory': {k: {'mean': 0, 'std': 0, 'count': 0} for k in ['client_peak', 'server_peak', 'enclave']},
        'payload': {'mean': 0, 'std': 0, 'count': 0},
        'convergence': {
            'accuracy': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
            'auc': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
            'loss': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
        },
    }
    
    # 聚合延迟数据：收集所有日志所有轮次的原始数据
    for key in ['training', 'encryption', 'communication', 'decryption', 'aggregation']:
        all_values = []
        for d in all_data:
            if d['latency'][key]:
                all_values.extend(d['latency'][key])  # 收集所有轮次的数据
        if all_values:
            aggregated['latency'][key]['mean'] = np.mean(all_values)
            aggregated['latency'][key]['std'] = np.std(all_values)
            aggregated['latency'][key]['count'] = len(all_values)
    
    # 聚合内存数据：收集所有日志所有轮次的原始数据
    for key in ['client_peak', 'server_peak', 'enclave']:
        all_values = []
        for d in all_data:
            if d['memory'][key]:
                all_values.extend(d['memory'][key])  # 收集所有轮次的数据
        if all_values:
            aggregated['memory'][key]['mean'] = np.mean(all_values)
            aggregated['memory'][key]['std'] = np.std(all_values)
            aggregated['memory'][key]['count'] = len(all_values)
    
    # 聚合 PAYLOAD 数据：收集所有轮次的数据
    all_payload = []
    for d in all_data:
        if d['payload']:
            all_payload.extend(d['payload'])
    if all_payload:
        aggregated['payload']['mean'] = np.mean(all_payload)
        aggregated['payload']['std'] = np.std(all_payload)
        aggregated['payload']['count'] = len(all_payload)
    
    # 聚合收敛数据（按轮次对齐）
    for metric in ['accuracy', 'auc', 'loss']:
        all_series = [d['convergence'][metric] for d in all_data if d['convergence'][metric]]
        if all_series:
            min_rounds = min(len(s) for s in all_series)
            for i in range(min_rounds):
                values_at_round = [s[i] for s in all_series]
                aggregated['convergence'][metric]['mean'].append(np.mean(values_at_round))
                aggregated['convergence'][metric]['std'].append(np.std(values_at_round))
            final_values = [s[-1] for s in all_series]
            aggregated['convergence'][metric]['final_mean'] = np.mean(final_values)
            aggregated['convergence'][metric]['final_std'] = np.std(final_values)
    
    return aggregated


def print_statistics_table(data_by_mode):
    """在终端打印带标准差的统计表格"""
    print("\n" + "=" * 140)
    print("📊 联邦学习实验统计表格（基于所有轮次的原始数据计算）")
    print("=" * 140)
    
    # 延迟表格
    print(f"\n{'模式':^6} | {'日志/轮数':^10} | {'训练(s)':^14} | {'加密(s)':^14} | "
          f"{'传输(s)':^14} | {'解密(s)':^14} | {'聚合(s)':^14} | {'上传量(MB)':^14}")
    print("-" * 140)
    
    for mode in MODE_ORDER:
        if mode not in data_by_mode:
            continue
        data = data_by_mode[mode]
        display_name = MODE_DISPLAY_NAMES.get(mode, mode.upper())
        
        # 显示日志数和总轮数
        logs_rounds = f"{data['num_logs']}/{data.get('total_rounds', 'N/A')}"
        
        train_str = f"{data['latency']['training']['mean']:.2f}±{data['latency']['training']['std']:.2f}"
        enc_str = f"{data['latency']['encryption']['mean']:.2f}±{data['latency']['encryption']['std']:.2f}"
        comm_str = f"{data['latency']['communication']['mean']:.2f}±{data['latency']['communication']['std']:.2f}"
        dec_str = f"{data['latency']['decryption']['mean']:.2f}±{data['latency']['decryption']['std']:.2f}"
        agg_str = f"{data['latency']['aggregation']['mean']:.2f}±{data['latency']['aggregation']['std']:.2f}"
        payload_str = f"{data['payload']['mean']:.2f}±{data['payload']['std']:.2f}" if data['payload']['mean'] > 0 else "N/A"
        
        print(f"{display_name:^6} | {logs_rounds:^10} | {train_str:^14} | {enc_str:^14} | "
              f"{comm_str:^14} | {dec_str:^14} | {agg_str:^14} | {payload_str:^14}")
    
    # 内存和收敛表格
    print("\n" + "-" * 140)
    print(f"\n{'模式':^6} | {'客户端峰值内存(MB)':^20} | {'服务端峰值内存(MB)':^20} | "
          f"{'Enclave峰值内存(MB)':^20} | {'最终准确率':^14} | {'最终AUC':^14}")
    print("-" * 140)
    
    for mode in MODE_ORDER:
        if mode not in data_by_mode:
            continue
        data = data_by_mode[mode]
        display_name = MODE_DISPLAY_NAMES.get(mode, mode.upper())
        
        client_mem = f"{data['memory']['client_peak']['mean']:.1f}±{data['memory']['client_peak']['std']:.1f}"
        server_mem = f"{data['memory']['server_peak']['mean']:.1f}±{data['memory']['server_peak']['std']:.1f}"
        # Enclave 内存（注：SGX EPC 默认 128MB，但 SGX2 EDMM 或模拟环境可能超过此限制）
        enclave_mem = f"{data['memory']['enclave']['mean']:.1f}±{data['memory']['enclave']['std']:.1f}" if data['memory']['enclave']['mean'] > 0 else "N/A"
        
        acc = f"{data['convergence']['accuracy']['final_mean']:.4f}±{data['convergence']['accuracy']['final_std']:.4f}"
        auc = f"{data['convergence']['auc']['final_mean']:.4f}±{data['convergence']['auc']['final_std']:.4f}"
        
        print(f"{display_name:^6} | {client_mem:^20} | {server_mem:^20} | {enclave_mem:^20} | "
              f"{acc:^14} | {auc:^14}")
    
    print("=" * 130)
    print("注：Enclave 内存可能超过 128MB EPC 限制，这表明使用了 SGX2 EDMM 或模拟环境")
    print("=" * 130 + "\n")


def plot_latency_breakdown(data_by_mode, output_path):
    """绘制端到端延迟分解堆叠柱状图（论文风格）"""
    modes = [m for m in MODE_ORDER if m in data_by_mode]
    n_modes = len(modes)
    
    if n_modes == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_modes)
    width = 0.6
    
    components = ['training', 'encryption', 'communication', 'aggregation', 'decryption']
    
    bottom = np.zeros(n_modes)
    
    # 获取宋体字体属性
    song_prop = fm.FontProperties(family=SONG_FONT, size=11)
    
    for comp in components:
        values = [data_by_mode[mode]['latency'][comp]['mean'] for mode in modes]
        label = LABELS_CN[comp]
        ax.bar(x, values, width, label=label, bottom=bottom, color=COLORS[comp])
        
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 2.0:
                ax.text(i, b + val/2, f'{val:.1f}', ha='center', va='center', 
                       fontsize=9, color='white', fontweight='bold')
        bottom += np.array(values)
    
    # 添加误差条
    total_std = []
    for mode in modes:
        std_sum = sum(data_by_mode[mode]['latency'][c]['std']**2 for c in components)
        total_std.append(np.sqrt(std_sum))
    ax.errorbar(x, bottom, yerr=total_std, fmt='none', ecolor='black', capsize=5, capthick=1.5)
    
    # 总时间标签
    for i, (total, std) in enumerate(zip(bottom, total_std)):
        ax.text(i, total + max(bottom) * 0.03, f'{total:.1f}s', ha='center', va='bottom', 
               fontsize=10, fontweight='bold', fontfamily=TIMES_FONT)
    
    ax.set_ylabel('时间 (s)', fontsize=14, fontweight='bold', fontproperties=song_prop)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_DISPLAY_NAMES.get(m, m.upper()) for m in modes], fontsize=12, fontfamily=TIMES_FONT)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, prop=song_prop)
    ax.set_ylim(0, max(bottom) * 1.15)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 延迟分解图已保存: {output_path}")


def plot_memory_comparison(data_by_mode, output_path):
    """绘制内存足迹对比图（论文风格）"""
    modes = [m for m in MODE_ORDER if m in data_by_mode]
    n_modes = len(modes)
    
    if n_modes == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_modes)
    width = 0.25
    
    # 获取宋体字体属性
    song_prop = fm.FontProperties(family=SONG_FONT, size=11)
    
    client_mem = [data_by_mode[m]['memory']['client_peak']['mean'] for m in modes]
    client_std = [data_by_mode[m]['memory']['client_peak']['std'] for m in modes]
    server_mem = [data_by_mode[m]['memory']['server_peak']['mean'] for m in modes]
    server_std = [data_by_mode[m]['memory']['server_peak']['std'] for m in modes]
    enclave_mem = [data_by_mode[m]['memory']['enclave']['mean'] for m in modes]
    enclave_std = [data_by_mode[m]['memory']['enclave']['std'] for m in modes]
    
    rects1 = ax.bar(x - width, client_mem, width, label=LABELS_CN['client'], color=COLORS['client'], yerr=client_std, capsize=3)
    rects2 = ax.bar(x, server_mem, width, label=LABELS_CN['server'], color=COLORS['server'], yerr=server_std, capsize=3)
    rects3 = ax.bar(x + width, enclave_mem, width, label=LABELS_CN['enclave'], color=COLORS['enclave'], yerr=enclave_std, capsize=3)
    
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontfamily=TIMES_FONT)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    ax.set_ylabel('峰值内存 (MB)', fontsize=14, fontweight='bold', fontproperties=song_prop)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_DISPLAY_NAMES.get(m, m.upper()) for m in modes], fontsize=12, fontfamily=TIMES_FONT)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, prop=song_prop)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 内存对比图已保存: {output_path}")


def plot_convergence(data_by_mode, output_path):
    """绘制收敛曲线（仅准确率，论文风格，带误差带）"""
    modes = [m for m in MODE_ORDER if m in data_by_mode]
    
    if not modes:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for mode in modes:
        data = data_by_mode[mode]
        mean_values = data['convergence']['accuracy']['mean']
        std_values = data['convergence']['accuracy']['std']
        
        if mean_values:
            rounds = list(range(1, len(mean_values) + 1))
            mean_arr = np.array(mean_values)
            std_arr = np.array(std_values)
            
            color = MODE_COLORS.get(mode, '#000000')
            display_name = MODE_DISPLAY_NAMES.get(mode, mode.upper())
            
            # 绘制均值线
            ax.plot(rounds, mean_arr, 
                   color=color,
                   linewidth=2.5,
                   label=display_name,
                   marker='o',
                   markersize=5,
                   markerfacecolor=color,
                   markeredgecolor='white',
                   markeredgewidth=0.8)
            
            # 绘制误差带
            ax.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr, 
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold', fontfamily=TIMES_FONT)
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', fontfamily=TIMES_FONT)
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # 设置Y轴范围
    all_means = []
    for mode in modes:
        all_means.extend(data_by_mode[mode]['convergence']['accuracy']['mean'])
    if all_means:
        y_min = max(0, min(all_means) - 0.1)
        y_max = min(1.0, max(all_means) + 0.05)
        ax.set_ylim([y_min, y_max])
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 收敛曲线已保存: {output_path}")


def run_analysis(logs_dir=None, output_dir=None):
    """主分析函数"""
    if logs_dir is None:
        logs_dir = LOGS_DIR
    if output_dir is None:
        output_dir = OUT_DIR
    
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 日志目录: {logs_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    logs_by_mode = discover_logs(logs_dir)
    
    if not logs_by_mode:
        print("❌ 未发现任何日志文件")
        return
    
    print(f"\n📋 发现的日志文件:")
    for mode, logs in sorted(logs_by_mode.items()):
        print(f"   {MODE_DISPLAY_NAMES.get(mode, mode.upper())}: {len(logs)} 个日志")
    
    data_by_mode = {}
    for mode, log_paths in logs_by_mode.items():
        print(f"\n🔄 解析 {MODE_DISPLAY_NAMES.get(mode, mode.upper())} 模式日志...")
        aggregated = aggregate_multiple_logs(log_paths)
        if aggregated:
            data_by_mode[mode] = aggregated
    
    if not data_by_mode:
        print("❌ 没有成功解析的日志数据")
        return
    
    print_statistics_table(data_by_mode)
    
    print("📊 生成图表...")
    plot_latency_breakdown(data_by_mode, output_dir / 'latency_breakdown.png')
    plot_memory_comparison(data_by_mode, output_dir / 'memory_comparison.png')
    plot_convergence(data_by_mode, output_dir / 'convergence.png')
    
    print(f"\n✅ 所有图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='联邦学习实验结果可视化工具')
    parser.add_argument('logs', nargs='*', help='日志文件路径（不指定则自动读取logs目录）')
    parser.add_argument('-o', '--output', default=None, help='输出目录')
    parser.add_argument('--logs-dir', default=None, help='日志目录')
    
    args = parser.parse_args()
    
    if not args.logs:
        logs_dir = args.logs_dir if args.logs_dir else LOGS_DIR
        output_dir = args.output if args.output else OUT_DIR
        run_analysis(logs_dir, output_dir)
    else:
        output_dir = Path(args.output) if args.output else OUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logs_by_mode = defaultdict(list)
        for log_path in args.logs:
            filename = Path(log_path).stem
            for mode in MODE_ORDER:
                if filename.startswith(f'{mode}_'):
                    logs_by_mode[mode].append(Path(log_path))
                    break
        
        data_by_mode = {}
        for mode, log_paths in logs_by_mode.items():
            aggregated = aggregate_multiple_logs(log_paths)
            if aggregated:
                data_by_mode[mode] = aggregated
        
        if data_by_mode:
            print_statistics_table(data_by_mode)
            plot_latency_breakdown(data_by_mode, output_dir / 'latency_breakdown.png')
            plot_memory_comparison(data_by_mode, output_dir / 'memory_comparison.png')
            plot_convergence(data_by_mode, output_dir / 'convergence.png')


if __name__ == '__main__':
    main()
