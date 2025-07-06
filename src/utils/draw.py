import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

# 设置matplotlib全局字体为SimHei，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_exp_results_bar_chart(save_path=None, show=True):
    """
    绘制实验结果分组柱状图（美化版，支持中文）。
    :param save_path: 图片保存路径（可选）
    :param show: 是否显示图片
    """
    # 使用seaborn Set2浅色配色
    colors = sns.color_palette('Set2', n_colors=5)
    # 模型名称
    models = [
        'XGBoost',
        'Neural Net',
        'Logistic',
        'Logistic (Fed)',
        'Random Forest',
        'Neural Net (Fed)'
    ]
    # 指标名称
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    # 各模型的五项指标数据
    data = [
        [0.8209, 0.6887, 0.3240, 0.4407, 0.7794],
        [0.8191, 0.6621, 0.3459, 0.4544, 0.7728],
        [0.8182, 0.6506, 0.3571, 0.4614, 0.7590],
        [0.8182, 0.6494, 0.3391, 0.4625, 0.7580],
        [0.8181, 0.6809, 0.3102, 0.4262, 0.7731],
        [0.8178, 0.6550, 0.3449, 0.4519, 0.7694],
    ]
    data = np.array(data)

    x = np.arange(len(models))  # 模型的下标
    total_width = 0.8  # 每组柱子的总宽度，调小让组间距更大
    bar_width = total_width / len(metrics)

    plt.figure(figsize=(12, 7))
    bars = []
    for i, metric in enumerate(metrics):
        bar = plt.bar(x + i * bar_width, data[:, i], width=bar_width, label=metric, color=colors[i])
        bars.append(bar)
        # 给每个柱子顶部加上数值标签
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10)

    plt.xticks(x + total_width / 2 - bar_width / 2, models, rotation=20, fontsize=12)
    plt.ylabel('分数', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def set_bar_width(ax, pixel_width=60):
    """
    将seaborn/matplotlib柱状图的每根柱子宽度设置为指定像素宽度。
    :param ax: seaborn/matplotlib的Axes对象
    :param pixel_width: 每根柱子的像素宽度
    """
    fig = ax.figure
    fig_width_inch = fig.get_size_inches()[0]
    fig_dpi = fig.dpi
    total_pixel = fig_width_inch * fig_dpi
    n_bars = len(ax.patches)
    if n_bars == 0:
        return
    # 计算x轴总跨度
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    # 每像素对应的x轴长度
    x_per_pixel = x_span / total_pixel
    bar_width = pixel_width * x_per_pixel
    for patch in ax.patches:
        # patch.get_x() 是左边界，get_width() 是原宽度
        center = patch.get_x() + patch.get_width() / 2
        patch.set_width(bar_width)
        patch.set_x(center - bar_width / 2)

def draw_sec_overhead_bar_chart(save_path=None, show=True):
    """
    绘制不同安全机制下联邦学习10轮训练时长对比柱状图。
    :param save_path: 图片保存路径（可选）
    :param show: 是否显示图片
    """
    colors = sns.color_palette('Set2', n_colors=3)
    scenes = [
        '基础实验',
        '同态加密实验',
        'TEE 实验'
    ]
    durations = [3.068, 104.502, 3.222]
    rel_overhead = ['1.00×', '34.06×', '1.05×']
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=scenes, y=durations, palette=colors, width=0.8)
    set_bar_width(ax, pixel_width=60)
    for i, v in enumerate(durations):
        plt.text(i, v + max(durations) * 0.03, rel_overhead[i],
                 ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')
    plt.ylabel('10轮训练时长（秒）', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 120)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def draw_data_analysis_charts():
    """
    对台湾2005年信用卡违约数据集进行基础数据分析并绘图，结果保存到out目录。
    """
    data_path = os.path.join(os.path.dirname(base_dir), 'data', 'default of credit card clients.csv')
    df = pd.read_csv(data_path, header=1)
    df.columns = df.columns.str.strip()
    out_dir = os.path.join(base_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # 1. 信用额度分组与违约率
    bins = [0, 50000, 100000, 200000, 500000, 1000000, df['LIMIT_BAL'].max() + 1]
    labels = ['≤5万', '5-10万', '10-20万', '20-50万', '50-100万', '>100万']
    df['额度分组'] = pd.cut(df['LIMIT_BAL'], bins=bins, labels=labels, right=True)
    limit_default = df.groupby('额度分组')['default payment next month'].mean() * 100
    plt.figure(figsize=(8,6))
    ax1 = sns.barplot(x=limit_default.index, y=limit_default.values, palette='Set2', width=0.8)
    set_bar_width(ax1, pixel_width=60)
    for i, v in enumerate(limit_default.values):
        plt.text(i, v+0.5, f'{v:.2f}%', ha='center', fontsize=11)
    plt.ylabel('违约率（%）', fontsize=13)
    plt.xlabel('信用额度分组', fontsize=13)
    plt.tight_layout()
    limit_path = os.path.join(out_dir, 'credit_limit_default.png')
    plt.savefig(limit_path, dpi=300)
    plt.close()
    print(f"额度分组违约率图已保存到: {limit_path}")

    # 2. 21-40岁客户违约案例占比
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 60, 100], labels=['21-30', '31-40', '41-50', '51-60', '61+'])
    age_default = df[df['default payment next month']==1]['AGE_GROUP'].value_counts().sort_index()
    all_default = df[df['default payment next month']==1]['AGE_GROUP'].count()
    plt.figure(figsize=(8,6))
    ax2 = sns.barplot(x=age_default.index, y=age_default.values, palette='Set2', width=0.8)
    set_bar_width(ax2, pixel_width=60)
    for i, v in enumerate(age_default.values):
        plt.text(i, v+10, f'{v/all_default*100:.1f}%', ha='center', fontsize=11)
    plt.ylabel('违约案例数', fontsize=13)
    plt.xlabel('年龄分组', fontsize=13)
    plt.tight_layout()
    age_path = os.path.join(out_dir, 'age_default.png')
    plt.savefig(age_path, dpi=300)
    plt.close()
    print(f"年龄违约分布图已保存到: {age_path}")

    # 3. 性别与违约比例
    gender_map = {1: '男', 2: '女'}
    df['SEX_CN'] = df['SEX'].map(gender_map)
    gender_default = df.groupby('SEX_CN')['default payment next month'].mean() * 100
    plt.figure(figsize=(8,6))
    ax3 = sns.barplot(x=gender_default.index, y=gender_default.values, palette='Set2', width=0.8)
    set_bar_width(ax3, pixel_width=60)
    for i, v in enumerate(gender_default.values):
        plt.text(i, v+0.5, f'{v:.2f}%', ha='center', fontsize=11)
    plt.ylabel('违约率（%）', fontsize=13)
    plt.xlabel('性别', fontsize=13)
    plt.tight_layout()
    gender_path = os.path.join(out_dir, 'gender_default.png')
    plt.savefig(gender_path, dpi=300)
    plt.close()
    print(f"性别违约率图已保存到: {gender_path}")

    # 4. 教育程度与违约风险
    edu_map = {1: '研究生', 2: '大学', 3: '高中', 4: '其他', 5: '其他', 6: '其他', 0: '其他'}
    df['EDU_CN'] = df['EDUCATION'].map(edu_map)
    edu_default = df.groupby('EDU_CN')['default payment next month'].mean() * 100
    edu_count = df['EDU_CN'].value_counts().reindex(['研究生','大学','高中','其他'])
    plt.figure(figsize=(8,6))
    ax4 = sns.barplot(x=edu_default.index, y=edu_default.values, palette='Set2', width=0.8)
    set_bar_width(ax4, pixel_width=60)
    for i, v in enumerate(edu_default.values):
        plt.text(i, v+0.5, f'{v:.2f}%', ha='center', fontsize=11)
    plt.ylabel('违约率（%）', fontsize=13)
    plt.xlabel('教育程度', fontsize=13)
    plt.tight_layout()
    edu_path = os.path.join(out_dir, 'education_default.png')
    plt.savefig(edu_path, dpi=300)
    plt.close()
    print(f"教育违约率图已保存到: {edu_path}")

    # 5. PAY_1至PAY_6还款状态时序分布
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    pay_status = df[pay_cols].replace(-2, -1)  # -2视为-1
    pay_status = pay_status.applymap(lambda x: x if x >= -1 else -1)
    pay_status = pay_status.apply(pd.Series.value_counts)
    pay_status = pay_status.reindex(index=range(-1,10))
    pay_status = pay_status.fillna(0).astype(int)
    plt.figure(figsize=(10,6))
    sns.heatmap(pay_status, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': '人数'})
    plt.ylabel('还款状态编码', fontsize=13)
    plt.xlabel('月份(PAY_1=9月, PAY_6=4月)', fontsize=13)
    plt.tight_layout()
    pay_path = os.path.join(out_dir, 'pay_status_heatmap.png')
    plt.savefig(pay_path, dpi=300)
    plt.close()
    print(f"还款状态热力图已保存到: {pay_path}")

# 获取当前文件（draw.py）所在目录的上一级目录
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base_dir, 'out')
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, 'exp_results.png')

draw_exp_results_bar_chart(save_path=save_path, show=False)
print(f"图表已保存到: {save_path}")

# 保存到out/sec_overhead.png
sec_overhead_path = os.path.join(out_dir, 'sec_overhead.png')
draw_sec_overhead_bar_chart(save_path=sec_overhead_path, show=False)
print(f"图表已保存到: {sec_overhead_path}")

draw_data_analysis_charts()
