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
import matplotlib.patheffects as pe

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

# 颜色方案（黑白印刷风格）
COLORS = {
    'training': '#404040',      # 训练 - 深灰
    'encryption': '#606060',    # 加密 - 中深灰
    'communication': '#808080', # 传输 - 中灰
    'decryption': '#A0A0A0',    # 解密 - 浅灰
    'aggregation': '#C0C0C0',   # 聚合 - 更浅灰
    'evaluation': '#E0E0E0',    # 评估 - 最浅灰
    'client': '#404040',        # 客户端内存 - 深灰
    'server': '#808080',        # 服务端内存 - 中灰
    'enclave': '#C0C0C0',       # Enclave内存 - 浅灰
}

# 中文图例标签
LABELS_CN = {
    'training': '训练',
    'encryption': '加密',
    'communication': '传输',
    'decryption': '解密',
    'aggregation': '聚合',
    'evaluation': '评估',
    'client': '客户端内存峰值',
    'server': '服务端内存峰值',
    'enclave': 'Enclave内存峰值',
}

# 模式颜色（黑白）
MODE_COLORS = {
    'none': '#000000',   # 黑色
    'he': '#404040',     # 深灰
    'mpc': '#606060',    # 中深灰
    'sgx': '#808080',    # 中灰
    'tee': '#A0A0A0',    # 浅灰
}

# 模式线型（用于收敛曲线）
MODE_LINESTYLES = {
    'none': '-',         # 实线
    'he': '--',          # 虚线
    'mpc': '-.',         # 点划线
    'sgx': ':',          # 点线
    'tee': (0, (5, 1)),  # 长虚线
}

# 模式标记（用于收敛曲线）
MODE_MARKERS = {
    'none': 'o',
    'he': 's',
    'mpc': '^',
    'sgx': 'D',
    'tee': 'v',
}


def parse_log_file(log_path):
    """
    解析联邦学习日志文件，提取关键指标
    支持 HE 模式的流式传输时间推算
    """
    from datetime import datetime
    
    data = {
        'rounds': [],
        'latency': {
            'training': [],
            'encryption': [],
            'communication': [],  # 传输时间
            'decryption': [],
            'aggregation': [],
            'evaluation': [],     # 全局评估时间
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
    
    # 用于计算评估时间的时间戳
    round_agg_complete_time = {}   # {round: timestamp}
    round_eval_complete_time = {}  # {round: timestamp}
    
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
        # 用于计算评估时间的时间戳
        'timestamp': re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'agg_complete': re.compile(r'\[Round (\d+)\] 全局模型更新完成'),
        'eval_complete': re.compile(r'\[Round (\d+)\] 全局评估'),
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
            
            # 聚合完成时间戳（用于计算评估时间）
            m = patterns['agg_complete'].search(line)
            if m:
                rnd = int(m.group(1))
                ts_match = patterns['timestamp'].search(line)
                if ts_match:
                    round_agg_complete_time[rnd] = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                continue
            
            # 评估完成时间戳
            m = patterns['eval_complete'].search(line)
            if m:
                rnd = int(m.group(1))
                ts_match = patterns['timestamp'].search(line)
                if ts_match:
                    round_eval_complete_time[rnd] = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                continue
    
    # 计算评估时间
    for rnd in round_agg_complete_time:
        if rnd in round_eval_complete_time:
            eval_time = (round_eval_complete_time[rnd] - round_agg_complete_time[rnd]).total_seconds()
            data['latency']['evaluation'].append(max(0, eval_time))
    
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
        'latency': {k: {'mean': 0, 'std': 0, 'count': 0} for k in ['training', 'encryption', 'communication', 'decryption', 'aggregation', 'evaluation']},
        'memory': {k: {'mean': 0, 'std': 0, 'count': 0} for k in ['client_peak', 'server_peak', 'enclave']},
        'payload': {'mean': 0, 'std': 0, 'count': 0},
        'convergence': {
            'accuracy': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
            'auc': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
            'loss': {'mean': [], 'std': [], 'final_mean': 0, 'final_std': 0},
        },
    }
    
    # 聚合延迟数据：收集所有日志所有轮次的原始数据
    for key in ['training', 'encryption', 'communication', 'decryption', 'aggregation', 'evaluation']:
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
    print("\n" + "=" * 160)
    print("📊 联邦学习实验统计表格（基于所有轮次的原始数据计算）")
    print("=" * 160)
    
    # 延迟表格
    print(f"\n{'模式':^6} | {'日志/轮':^8} | {'训练(s)':^14} | {'加密(s)':^14} | "
          f"{'传输(s)':^12} | {'解密(s)':^14} | {'聚合(s)':^14} | {'评估(s)':^12} | {'上传量(MB)':^12}")
    print("-" * 160)
    
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
        eval_str = f"{data['latency']['evaluation']['mean']:.2f}±{data['latency']['evaluation']['std']:.2f}" if data['latency']['evaluation']['mean'] > 0 else "N/A"
        payload_str = f"{data['payload']['mean']:.1f}±{data['payload']['std']:.1f}" if data['payload']['mean'] > 0 else "N/A"
        
        print(f"{display_name:^6} | {logs_rounds:^8} | {train_str:^14} | {enc_str:^14} | "
              f"{comm_str:^12} | {dec_str:^14} | {agg_str:^14} | {eval_str:^12} | {payload_str:^12}")
    
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
    """绘制端到端延迟分解横向堆叠条形图（黑白印刷风格）"""
    # 完整的端到端延迟分解组件（按时间顺序）
    components = ['training', 'encryption', 'communication', 'decryption', 'aggregation', 'evaluation']

    # 填充图案（黑白印刷用）
    hatch_patterns = {
        'training': '',                    # 空白（深灰底）
        'encryption': '///',               # 斜线
        'communication': '...',            # 点
        'decryption': 'xxx',               # 交叉
        'aggregation': '\\\\\\',           # 反斜线
        'evaluation': '***',               # 星号
    }

    # 计算每个模式的总时间并排序（从上往下时间递减）
    modes = [m for m in MODE_ORDER if m in data_by_mode]
    mode_totals = []
    for mode in modes:
        total = sum(data_by_mode[mode]['latency'][c]['mean'] for c in components)
        mode_totals.append((mode, total))
    # 按总时间升序排序（barh中y=0在底部，所以时间短的在底部，时间长的在顶部）
    mode_totals.sort(key=lambda x: x[1])
    modes = [m for m, _ in mode_totals]

    n_modes = len(modes)

    if n_modes == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    y = np.arange(n_modes)
    height = 0.5  # 细柱子

    left = np.zeros(n_modes)

    # 获取宋体字体属性（通过描边模拟加粗）
    song_prop = fm.FontProperties(family=SONG_FONT, size=11)
    song_prop_label = fm.FontProperties(family=SONG_FONT, size=14)
    stroke_effect = [pe.withStroke(linewidth=0.8, foreground='black')]

    for comp in components:
        values = [data_by_mode[mode]['latency'][comp]['mean'] for mode in modes]
        label = LABELS_CN[comp]
        # 横向条形图，带黑边和填充图案
        ax.barh(y, values, height, label=label, left=left, color=COLORS[comp],
                edgecolor='black', linewidth=0.8, hatch=hatch_patterns[comp])

        # 在较大的分段中添加数值标签
        for i, (val, l) in enumerate(zip(values, left)):
            if val > 5.0:
                ax.text(l + val/2, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
        left += np.array(values)

    # 添加误差条
    total_std = []
    for mode in modes:
        std_sum = sum(data_by_mode[mode]['latency'][c]['std']**2 for c in components)
        total_std.append(np.sqrt(std_sum))
    ax.errorbar(left, y, xerr=total_std, fmt='none', ecolor='black', capsize=3, capthick=1)

    # 总时间标签
    for i, (total, std) in enumerate(zip(left, total_std)):
        ax.text(total + max(left) * 0.02, i, f'{total:.1f}s', ha='left', va='center',
               fontsize=10, fontweight='bold', fontfamily=TIMES_FONT)

    xlabel = ax.set_xlabel('时间/s', fontsize=14, fontproperties=song_prop_label)
    xlabel.set_path_effects(stroke_effect)
    ax.set_yticks(y)
    ax.set_yticklabels([MODE_DISPLAY_NAMES.get(m, m.upper()) for m in modes], fontsize=12, fontfamily=TIMES_FONT)
    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.9, prop=song_prop, ncol=2)
    for text in legend.get_texts():
        text.set_path_effects(stroke_effect)
    ax.set_xlim(0, max(left) * 1.15)

    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 延迟分解图已保存: {output_path}")


def plot_memory_comparison(data_by_mode, output_path):
    """绘制内存足迹对比图（黑白印刷风格）"""
    # 计算每个模式的最大峰值内存并排序（从左往右递增）
    modes = [m for m in MODE_ORDER if m in data_by_mode]
    mode_peaks = []
    for mode in modes:
        # 取客户端和服务端峰值中的最大值作为排序依据
        max_peak = max(
            data_by_mode[mode]['memory']['client_peak']['mean'],
            data_by_mode[mode]['memory']['server_peak']['mean']
        )
        mode_peaks.append((mode, max_peak))
    # 按峰值升序排序（从左往右递增）
    mode_peaks.sort(key=lambda x: x[1])
    modes = [m for m, _ in mode_peaks]

    # 手动交换第四个和第五个的位置
    if len(modes) >= 5:
        modes[3], modes[4] = modes[4], modes[3]

    n_modes = len(modes)

    if n_modes == 0:
        return

    # 填充图案（黑白印刷用）
    hatch_patterns = {
        'client': '',        # 空白（深灰底）
        'server': '///',     # 斜线
        'enclave': '...',    # 点
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_modes)
    width = 0.25

    # 获取宋体字体属性（通过描边模拟加粗）
    song_prop = fm.FontProperties(family=SONG_FONT, size=11)
    song_prop_label = fm.FontProperties(family=SONG_FONT, size=14)
    stroke_effect = [pe.withStroke(linewidth=0.8, foreground='black')]

    client_mem = [data_by_mode[m]['memory']['client_peak']['mean'] for m in modes]
    client_std = [data_by_mode[m]['memory']['client_peak']['std'] for m in modes]
    server_mem = [data_by_mode[m]['memory']['server_peak']['mean'] for m in modes]
    server_std = [data_by_mode[m]['memory']['server_peak']['std'] for m in modes]
    enclave_mem = [data_by_mode[m]['memory']['enclave']['mean'] for m in modes]
    enclave_std = [data_by_mode[m]['memory']['enclave']['std'] for m in modes]

    # 添加黑边和填充图案
    rects1 = ax.bar(x - width, client_mem, width, label=LABELS_CN['client'], color=COLORS['client'],
                    yerr=client_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['client'])
    rects2 = ax.bar(x, server_mem, width, label=LABELS_CN['server'], color=COLORS['server'],
                    yerr=server_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['server'])
    rects3 = ax.bar(x + width, enclave_mem, width, label=LABELS_CN['enclave'], color=COLORS['enclave'],
                    yerr=enclave_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['enclave'])

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

    ylabel = ax.set_ylabel('内存峰值/MB', fontsize=14, fontproperties=song_prop_label)
    ylabel.set_path_effects(stroke_effect)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_DISPLAY_NAMES.get(m, m.upper()) for m in modes], fontsize=12, fontfamily=TIMES_FONT)
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9, prop=song_prop)
    for text in legend.get_texts():
        text.set_path_effects(stroke_effect)

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
    """绘制收敛曲线（仅准确率，黑白印刷风格，不同线型，百分比）"""
    modes = [m for m in MODE_ORDER if m in data_by_mode]

    if not modes:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # 获取宋体字体属性（通过描边模拟加粗）
    song_prop = fm.FontProperties(family=SONG_FONT, size=14)
    stroke_effect = [pe.withStroke(linewidth=0.8, foreground='black')]

    for mode in modes:
        data = data_by_mode[mode]
        mean_values = data['convergence']['accuracy']['mean']
        std_values = data['convergence']['accuracy']['std']

        if mean_values:
            rounds = list(range(1, len(mean_values) + 1))
            # 转换为百分比
            mean_arr = np.array(mean_values) * 100
            std_arr = np.array(std_values) * 100

            color = MODE_COLORS.get(mode, '#000000')
            linestyle = MODE_LINESTYLES.get(mode, '-')
            marker = MODE_MARKERS.get(mode, 'o')
            display_name = MODE_DISPLAY_NAMES.get(mode, mode.upper())

            # 绘制均值线（黑白 + 不同线型 + 不同标记）
            ax.plot(rounds, mean_arr,
                   color=color,
                   linewidth=2.5,
                   linestyle=linestyle,
                   label=display_name,
                   marker=marker,
                   markersize=5,
                   markerfacecolor=color,
                   markeredgecolor='white',
                   markeredgewidth=0.8)

            # 绘制误差带
            ax.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr,
                           color=color, alpha=0.15)

    xlabel = ax.set_xlabel('轮数', fontsize=14, fontproperties=song_prop)
    ylabel = ax.set_ylabel('准确率/%', fontsize=14, fontproperties=song_prop)
    xlabel.set_path_effects(stroke_effect)
    ylabel.set_path_effects(stroke_effect)

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 设置Y轴范围（百分比）
    all_means = []
    for mode in modes:
        all_means.extend(data_by_mode[mode]['convergence']['accuracy']['mean'])
    if all_means:
        y_min = max(0, (min(all_means) - 0.1) * 100)
        y_max = min(100, (max(all_means) + 0.05) * 100)
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


def plot_combined_convergence_memory(data_by_mode, output_path):
    """绘制收敛曲线和内存对比的水平组合图（黑白印刷风格）"""
    # 完整的端到端延迟分解组件
    components = ['training', 'encryption', 'communication', 'decryption', 'aggregation', 'evaluation']

    # 填充图案（黑白印刷用）
    hatch_patterns = {
        'client': '',        # 空白（深灰底）
        'server': '///',     # 斜线
        'enclave': '...',    # 点
    }

    # 计算每个模式的最大峰值内存并排序（从左往右递增）
    modes_for_mem = [m for m in MODE_ORDER if m in data_by_mode]
    mode_peaks = []
    for mode in modes_for_mem:
        max_peak = max(
            data_by_mode[mode]['memory']['client_peak']['mean'],
            data_by_mode[mode]['memory']['server_peak']['mean']
        )
        mode_peaks.append((mode, max_peak))
    mode_peaks.sort(key=lambda x: x[1])
    modes_mem = [m for m, _ in mode_peaks]
    if len(modes_mem) >= 5:
        modes_mem[3], modes_mem[4] = modes_mem[4], modes_mem[3]

    modes_conv = [m for m in MODE_ORDER if m in data_by_mode]

    if not modes_conv or not modes_mem:
        return

    # 获取宋体字体属性（通过描边模拟加粗，因为 SimSun 没有粗体变体）
    song_prop = fm.FontProperties(family=SONG_FONT, size=14)
    song_prop_title = fm.FontProperties(family=SONG_FONT, size=16)
    song_prop_legend = fm.FontProperties(family=SONG_FONT, size=11)

    # 描边效果模拟加粗
    stroke_effect = [pe.withStroke(linewidth=0.8, foreground='black')]

    # 创建水平并排的两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ========== 左图：收敛曲线 ==========
    for mode in modes_conv:
        data = data_by_mode[mode]
        mean_values = data['convergence']['accuracy']['mean']
        std_values = data['convergence']['accuracy']['std']

        if mean_values:
            rounds = list(range(1, len(mean_values) + 1))
            # 转换为百分比
            mean_arr = np.array(mean_values) * 100
            std_arr = np.array(std_values) * 100

            color = MODE_COLORS.get(mode, '#000000')
            linestyle = MODE_LINESTYLES.get(mode, '-')
            marker = MODE_MARKERS.get(mode, 'o')
            display_name = MODE_DISPLAY_NAMES.get(mode, mode.upper())

            ax1.plot(rounds, mean_arr,
                    color=color,
                    linewidth=2.5,
                    linestyle=linestyle,
                    label=display_name,
                    marker=marker,
                    markersize=5,
                    markerfacecolor=color,
                    markeredgecolor='white',
                    markeredgewidth=0.8)

            ax1.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr,
                            color=color, alpha=0.15)

    xlabel1 = ax1.set_xlabel('轮数', fontsize=14, fontproperties=song_prop)
    ylabel1 = ax1.set_ylabel('准确率/%', fontsize=14, fontproperties=song_prop)
    xlabel1.set_path_effects(stroke_effect)
    ylabel1.set_path_effects(stroke_effect)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)

    all_means = []
    for mode in modes_conv:
        all_means.extend(data_by_mode[mode]['convergence']['accuracy']['mean'])
    if all_means:
        y_min = max(0, (min(all_means) - 0.1) * 100)
        y_max = min(100, (max(all_means) + 0.05) * 100)
        ax1.set_ylim([y_min, y_max])

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)

    # 左图子标题
    title1 = ax1.text(0.5, -0.12, '(a) 模型收敛曲线',
            transform=ax1.transAxes,
            fontproperties=song_prop_title,
            verticalalignment='top',
            horizontalalignment='center')
    title1.set_path_effects(stroke_effect)

    # ========== 右图：内存对比 ==========
    n_modes = len(modes_mem)
    x = np.arange(n_modes)
    width = 0.25

    client_mem = [data_by_mode[m]['memory']['client_peak']['mean'] for m in modes_mem]
    client_std = [data_by_mode[m]['memory']['client_peak']['std'] for m in modes_mem]
    server_mem = [data_by_mode[m]['memory']['server_peak']['mean'] for m in modes_mem]
    server_std = [data_by_mode[m]['memory']['server_peak']['std'] for m in modes_mem]
    enclave_mem = [data_by_mode[m]['memory']['enclave']['mean'] for m in modes_mem]
    enclave_std = [data_by_mode[m]['memory']['enclave']['std'] for m in modes_mem]

    rects1 = ax2.bar(x - width, client_mem, width, label=LABELS_CN['client'], color=COLORS['client'],
                    yerr=client_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['client'])
    rects2 = ax2.bar(x, server_mem, width, label=LABELS_CN['server'], color=COLORS['server'],
                    yerr=server_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['server'])
    rects3 = ax2.bar(x + width, enclave_mem, width, label=LABELS_CN['enclave'], color=COLORS['enclave'],
                    yerr=enclave_std, capsize=3, edgecolor='black', linewidth=0.8, hatch=hatch_patterns['enclave'])

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax2.annotate(f'{height:.0f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontfamily=TIMES_FONT)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    ylabel2 = ax2.set_ylabel('内存峰值/MB', fontsize=14, fontproperties=song_prop)
    ylabel2.set_path_effects(stroke_effect)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODE_DISPLAY_NAMES.get(m, m.upper()) for m in modes_mem], fontsize=12, fontfamily=TIMES_FONT)
    legend2 = ax2.legend(loc='upper right', fontsize=11, framealpha=0.9, prop=song_prop_legend)
    for text in legend2.get_texts():
        text.set_path_effects(stroke_effect)

    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # 右图子标题
    title2 = ax2.text(0.5, -0.12, '(b) 内存峰值对比',
            transform=ax2.transAxes,
            fontproperties=song_prop_title,
            verticalalignment='top',
            horizontalalignment='center')
    title2.set_path_effects(stroke_effect)

    # 调整布局，增加底部空间以容纳子标题
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 收敛+内存组合图已保存: {output_path}")


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
    plot_combined_convergence_memory(data_by_mode, output_dir / 'convergence_memory_combined.png')
    
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
