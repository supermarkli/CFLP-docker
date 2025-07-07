import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

# 设置中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'out'))
os.makedirs(save_dir, exist_ok=True)

def plot_global_convergence_curve(acc_list, loss_list=None, auc_list=None):
    """
    绘制全局模型收敛曲线，将accuracy、loss和AUC分别绘制成独立的图表。
    acc_list: 每轮的accuracy列表
    loss_list: 每轮的loss列表（可选）
    auc_list: 每轮的AUC列表（可选）
    """
    rounds = np.arange(1, len(acc_list)+1)
    
    # 绘制Accuracy曲线
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, acc_list, marker='o', color='#2E86C1')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.title('全局模型测试准确率')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()

    # 绘制Loss曲线（如果提供）
    if loss_list is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, loss_list, marker='s', color='#E74C3C')    
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('全局模型训练损失')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    # 绘制AUC曲线（如果提供）
    if auc_list is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, auc_list, marker='^', color='#27AE60')
        plt.xlabel('训练轮次')
        plt.ylabel('AUC值')
        plt.title('全局模型测试AUC')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'auc_curve.png'), dpi=300)
        plt.close()




