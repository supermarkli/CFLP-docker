#!/usr/bin/env python3
"""
联邦学习实验结果可视化工具

功能：
1. 端到端延迟分解图 (End-to-End Latency Breakdown)
2. 内存足迹对比 (Memory Footprint)
3. 收敛图 (Convergence Curve)

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

# 设置中文字体和数学公式字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 颜色方案
COLORS = {
    'training': '#2E86AB',      # 训练 - 蓝色
    'encryption': '#A23B72',    # 加密 - 紫红色
    'communication': '#F18F01', # 通信 - 橙色
    'decryption': '#C73E1D',    # 解密 - 红色
    'aggregation': '#3A7D44',   # 聚合 - 绿色
    'client': '#4ECDC4',        # 客户端内存 - 青色
    'server': '#FF6B6B',        # 服务端内存 - 珊瑚红
    'enclave': '#45B7D1',       # Enclave内存 - 天蓝
    'accuracy': '#2E86AB',      # 准确率 - 蓝色
    'auc': '#A23B72',           # AUC - 紫红
    'loss': '#F18F01',          # 损失 - 橙色
}


def parse_log_file(log_path):
    """
    解析联邦学习日志文件，提取关键指标
    
    返回:
        dict: 包含 rounds, latency, memory, convergence 数据的字典
    """
    data = {
        'rounds': [],
        'latency': {
            'training': [],      # 每轮所有客户端平均训练时间
            'encryption': [],    # 每轮所有客户端平均加密时间
            'decryption': [],    # 每轮解密时间
            'aggregation': [],   # 每轮聚合时间
        },
        'memory': {
            'client_peak': [],   # 客户端峰值内存
            'server_peak': [],   # 服务端峰值内存
            'enclave': [],       # Enclave 内存
        },
        'convergence': {
            'accuracy': [],
            'auc': [],
            'loss': [],
        },
        'client_latency': defaultdict(lambda: {'training': [], 'encryption': []}),
    }
    
    current_round = None
    round_client_training = []
    round_client_encryption = []
    round_client_memory = []
    
    # 正则表达式模式
    patterns = {
        # 新格式
        'round_client_latency': re.compile(
            r'\[Round (\d+)\]\[Client (\d+)\]\[LATENCY\] training=([\d.]+)s, encryption=([\d.]+)s'
        ),
        'round_client_resource': re.compile(
            r'\[Round (\d+)\]\[Client (\d+)\]\[RESOURCE\] peak_memory=([\d.]+) MB'
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
        # 旧格式
        'global_metrics_old': re.compile(
            r'\[第 (\d+) 轮\] 全局指标 - 准确率: ([\d.]+), AUC: ([\d.]+), 损失: ([\d.]+)'
        ),
        'enclave_resource_old': re.compile(
            r'\[第 (\d+) 轮\] Enclave资源 - CPU耗时: ([\d.]+)s, 内存使用: ([\d.]+) MB'
        ),
        'server_resource_old': re.compile(
            r'\[Round (\d+)\] Server Process \(Host\) Resources - CPU Time: ([\d.]+)s, Memory Usage: ([\d.]+) MB'
        ),
        # 全局评估（新格式）
        'global_eval': re.compile(
            r'\[Round (\d+)\] 全局评估: Acc=([\d.]+), AUC=([\d.]+)'
        ),
        'client_agg': re.compile(
            r'\[Round (\d+)\] 客户端聚合.*Loss=([\d.]+)'
        ),
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 新格式：客户端延迟
            m = patterns['round_client_latency'].search(line)
            if m:
                rnd, cid, train_t, enc_t = m.groups()
                rnd = int(rnd)
                if current_round != rnd:
                    # 保存上一轮的数据
                    if current_round is not None and round_client_training:
                        data['latency']['training'].append(np.mean(round_client_training))
                        data['latency']['encryption'].append(np.mean(round_client_encryption))
                        if round_client_memory:
                            data['memory']['client_peak'].append(np.max(round_client_memory))
                    current_round = rnd
                    round_client_training = []
                    round_client_encryption = []
                    round_client_memory = []
                    data['rounds'].append(rnd)
                round_client_training.append(float(train_t))
                round_client_encryption.append(float(enc_t))
                data['client_latency'][int(cid)]['training'].append(float(train_t))
                data['client_latency'][int(cid)]['encryption'].append(float(enc_t))
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
            
            # 服务端资源（新格式）
            m = patterns['round_server_resource'].search(line)
            if m:
                rnd, mem = m.groups()
                data['memory']['server_peak'].append(float(mem))
                continue
            
            # 服务端资源（旧格式）
            m = patterns['server_resource_old'].search(line)
            if m:
                rnd, cpu_t, mem = m.groups()
                if float(mem) not in data['memory']['server_peak']:
                    data['memory']['server_peak'].append(float(mem))
                continue
            
            # Enclave 资源（新格式）
            m = patterns['round_enclave_resource'].search(line)
            if m:
                rnd, cpu_t, mem = m.groups()
                data['memory']['enclave'].append(float(mem))
                continue
            
            # Enclave 资源（旧格式）
            m = patterns['enclave_resource_old'].search(line)
            if m:
                rnd, cpu_t, mem = m.groups()
                data['memory']['enclave'].append(float(mem))
                continue
            
            # 收敛数据（新格式）
            m = patterns['convergence_new'].search(line)
            if m:
                rnd, acc, auc, loss = m.groups()
                data['convergence']['accuracy'].append(float(acc))
                data['convergence']['auc'].append(float(auc))
                data['convergence']['loss'].append(float(loss))
                continue
            
            # 收敛数据（旧格式）
            m = patterns['global_metrics_old'].search(line)
            if m:
                rnd, acc, auc, loss = m.groups()
                data['convergence']['accuracy'].append(float(acc))
                data['convergence']['auc'].append(float(auc))
                data['convergence']['loss'].append(float(loss))
                continue
            
            # 全局评估 + 损失（新格式分开的情况）
            m = patterns['global_eval'].search(line)
            if m:
                rnd, acc, auc = m.groups()
                if len(data['convergence']['accuracy']) < int(rnd):
                    data['convergence']['accuracy'].append(float(acc))
                    data['convergence']['auc'].append(float(auc))
                continue
            
            m = patterns['client_agg'].search(line)
            if m:
                rnd, loss = m.groups()
                if len(data['convergence']['loss']) < int(rnd):
                    data['convergence']['loss'].append(float(loss))
                continue
    
    # 保存最后一轮的数据
    if round_client_training:
        data['latency']['training'].append(np.mean(round_client_training))
        data['latency']['encryption'].append(np.mean(round_client_encryption))
        if round_client_memory:
            data['memory']['client_peak'].append(np.max(round_client_memory))
    
    return data


def plot_latency_breakdown(data_dict, output_path, title="End-to-End Latency Breakdown"):
    """
    绘制端到端延迟分解堆叠柱状图
    
    Args:
        data_dict: {mode_name: parsed_data, ...}
        output_path: 输出图片路径
        title: 图表标题
    """
    modes = list(data_dict.keys())
    n_modes = len(modes)
    
    # 计算每种模式的平均延迟
    avg_latencies = {}
    for mode, data in data_dict.items():
        avg_latencies[mode] = {
            'training': np.mean(data['latency']['training']) if data['latency']['training'] else 0,
            'encryption': np.mean(data['latency']['encryption']) if data['latency']['encryption'] else 0,
            'decryption': np.mean(data['latency']['decryption']) if data['latency']['decryption'] else 0,
            'aggregation': np.mean(data['latency']['aggregation']) if data['latency']['aggregation'] else 0,
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_modes)
    width = 0.6
    
    # 堆叠顺序：training -> encryption -> aggregation -> decryption
    components = ['training', 'encryption', 'aggregation', 'decryption']
    labels = ['Training', 'Encryption', 'Aggregation', 'Decryption']
    
    bottom = np.zeros(n_modes)
    for comp, label in zip(components, labels):
        values = [avg_latencies[mode][comp] for mode in modes]
        bars = ax.bar(x, values, width, label=label, bottom=bottom, color=COLORS[comp])
        bottom += np.array(values)
        
        # 在每个分段中间添加数值标签（如果值足够大）
        for i, (val, b) in enumerate(zip(values, bottom - np.array(values))):
            if val > 0.5:  # 只显示大于0.5秒的值
                ax.text(i, b + val/2, f'{val:.1f}s', ha='center', va='center', 
                       fontsize=9, color='white', fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(bottom) * 1.15)
    
    # 添加总时间标签
    for i, total in enumerate(bottom):
        ax.text(i, total + 0.5, f'Total: {total:.1f}s', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 延迟分解图已保存: {output_path}")


def plot_memory_comparison(data_dict, output_path, title="Memory Footprint Comparison"):
    """
    绘制内存足迹对比图
    
    Args:
        data_dict: {mode_name: parsed_data, ...}
        output_path: 输出图片路径
        title: 图表标题
    """
    modes = list(data_dict.keys())
    n_modes = len(modes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_modes)
    width = 0.25
    
    # 客户端峰值内存
    client_mem = [np.max(data_dict[m]['memory']['client_peak']) 
                  if data_dict[m]['memory']['client_peak'] else 0 for m in modes]
    # 服务端峰值内存
    server_mem = [np.max(data_dict[m]['memory']['server_peak']) 
                  if data_dict[m]['memory']['server_peak'] else 0 for m in modes]
    # Enclave 内存
    enclave_mem = [np.max(data_dict[m]['memory']['enclave']) 
                   if data_dict[m]['memory']['enclave'] else 0 for m in modes]
    
    rects1 = ax.bar(x - width, client_mem, width, label='Client (Peak)', color=COLORS['client'])
    rects2 = ax.bar(x, server_mem, width, label='Server (Peak)', color=COLORS['server'])
    rects3 = ax.bar(x + width, enclave_mem, width, label='Enclave/TEE', color=COLORS['enclave'])
    
    # 添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 内存对比图已保存: {output_path}")


def plot_convergence(data_dict, output_path, title="Convergence Curves"):
    """
    绘制收敛曲线
    
    Args:
        data_dict: {mode_name: parsed_data, ...}
        output_path: 输出图片路径
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['accuracy', 'auc', 'loss']
    metric_labels = ['Test Accuracy', 'AUC', 'Training Loss']
    
    # 为每种模式分配不同的线型和颜色
    line_styles = ['-', '--', '-.', ':']
    mode_colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        for i, (mode, data) in enumerate(data_dict.items()):
            values = data['convergence'][metric]
            if values:
                rounds = list(range(1, len(values) + 1))
                ax.plot(rounds, values, 
                       linestyle=line_styles[i % len(line_styles)],
                       color=mode_colors[i],
                       marker='o' if len(values) < 30 else None,
                       markersize=4,
                       linewidth=2,
                       label=mode.upper())
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 收敛曲线已保存: {output_path}")


def plot_single_log(log_path, output_dir=None):
    """
    分析单个日志文件并生成图表
    """
    if output_dir is None:
        output_dir = Path(log_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode_name = Path(log_path).stem.replace('server_', '')
    data = parse_log_file(log_path)
    
    # 生成收敛图
    if data['convergence']['accuracy']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        rounds = list(range(1, len(data['convergence']['accuracy']) + 1))
        
        # Accuracy
        axes[0].plot(rounds, data['convergence']['accuracy'], 
                    color=COLORS['accuracy'], marker='o', linewidth=2)
        axes[0].set_xlabel('Round', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        # AUC
        axes[1].plot(rounds, data['convergence']['auc'], 
                    color=COLORS['auc'], marker='o', linewidth=2)
        axes[1].set_xlabel('Round', fontsize=11)
        axes[1].set_ylabel('AUC', fontsize=11)
        axes[1].set_title('AUC', fontsize=12, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.3)
        
        # Loss
        if data['convergence']['loss']:
            axes[2].plot(rounds[:len(data['convergence']['loss'])], 
                        data['convergence']['loss'], 
                        color=COLORS['loss'], marker='o', linewidth=2)
        axes[2].set_xlabel('Round', fontsize=11)
        axes[2].set_ylabel('Loss', fontsize=11)
        axes[2].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[2].grid(True, linestyle='--', alpha=0.3)
        
        plt.suptitle(f'{mode_name.upper()} Convergence', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        conv_path = output_dir / f'{mode_name}_convergence.png'
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 收敛图已保存: {conv_path}")
    
    # 生成延迟分解图（如果有数据）
    if data['latency']['training']:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        rounds = data['rounds'] if data['rounds'] else list(range(1, len(data['latency']['training']) + 1))
        
        # 堆叠面积图
        ax.stackplot(rounds,
                    data['latency']['training'],
                    data['latency']['encryption'],
                    data['latency']['aggregation'] if data['latency']['aggregation'] else [0]*len(rounds),
                    data['latency']['decryption'] if data['latency']['decryption'] else [0]*len(rounds),
                    labels=['Training', 'Encryption', 'Aggregation', 'Decryption'],
                    colors=[COLORS['training'], COLORS['encryption'], 
                           COLORS['aggregation'], COLORS['decryption']],
                    alpha=0.8)
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(f'{mode_name.upper()} Per-Round Latency Breakdown', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        latency_path = output_dir / f'{mode_name}_latency_per_round.png'
        plt.tight_layout()
        plt.savefig(latency_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 每轮延迟图已保存: {latency_path}")
    
    # 打印统计摘要
    print(f"\n📊 {mode_name.upper()} 统计摘要:")
    print(f"   总轮数: {len(data['convergence']['accuracy'])}")
    if data['convergence']['accuracy']:
        print(f"   最终准确率: {data['convergence']['accuracy'][-1]:.4f}")
        print(f"   最高准确率: {max(data['convergence']['accuracy']):.4f}")
    if data['convergence']['auc']:
        print(f"   最终 AUC: {data['convergence']['auc'][-1]:.4f}")
    if data['latency']['training']:
        print(f"   平均训练时间: {np.mean(data['latency']['training']):.2f}s")
        print(f"   平均加密时间: {np.mean(data['latency']['encryption']):.2f}s")
    if data['latency']['decryption']:
        print(f"   平均解密时间: {np.mean(data['latency']['decryption']):.2f}s")
    if data['latency']['aggregation']:
        print(f"   平均聚合时间: {np.mean(data['latency']['aggregation']):.2f}s")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='联邦学习实验结果可视化工具')
    parser.add_argument('logs', nargs='+', help='日志文件路径')
    parser.add_argument('-o', '--output', default='./plots', help='输出目录')
    parser.add_argument('--compare', action='store_true', help='对比多个日志')
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(args.logs) == 1 and not args.compare:
        # 单个日志分析
        plot_single_log(args.logs[0], output_dir)
    else:
        # 多个日志对比
        data_dict = {}
        for log_path in args.logs:
            mode_name = Path(log_path).stem.replace('server_', '')
            data_dict[mode_name] = parse_log_file(log_path)
            print(f"📂 已解析: {log_path}")
        
        # 生成对比图
        plot_latency_breakdown(data_dict, output_dir / 'latency_breakdown.png')
        plot_memory_comparison(data_dict, output_dir / 'memory_comparison.png')
        plot_convergence(data_dict, output_dir / 'convergence_comparison.png')


if __name__ == '__main__':
    main()

