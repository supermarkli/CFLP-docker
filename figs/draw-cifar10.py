import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 尝试导入 torchvision，如果失败则使用 tensorflow
try:
    import torchvision
    import torchvision.datasets as datasets
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

# 设置字体路径
FONT_PATH = os.path.join(os.path.dirname(__file__), 'simsun.ttf')
font_prop = fm.FontProperties(fname=FONT_PATH, size=14)

# 中文类别标签
CHINESE_LABELS = {
    'airplane': '飞机',
    'automobile': '汽车',
    'bird': '鸟',
    'cat': '猫',
    'deer': '鹿',
    'dog': '狗',
    'frog': '青蛙',
    'horse': '马',
    'ship': '船',
    'truck': '卡车'
}

# CIFAR-10 类别顺序
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar10_torch():
    """使用 torchvision 加载 CIFAR-10"""
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    images = dataset.data  # (N, 32, 32, 3)
    labels = np.array(dataset.targets)
    return images, labels

def load_cifar10_tf():
    """使用 tensorflow/keras 加载 CIFAR-10"""
    import tensorflow as tf
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    return x_train, y_train.flatten()

def get_samples_per_class(images, labels, num_samples=8):
    """获取每个类别的样本图像"""
    samples = {}
    for class_idx in range(10):
        class_indices = np.where(labels == class_idx)[0]
        # 随机选取指定数量的样本
        np.random.seed(42)  # 固定随机种子以保证可复现
        selected = np.random.choice(class_indices, num_samples, replace=False)
        samples[class_idx] = images[selected]
    return samples

def draw_cifar10_grid(samples, num_samples=8, save_path='cifar10_samples.png'):
    """绘制 CIFAR-10 数据集网格图"""
    num_classes = 10
    
    # 创建图形，调整尺寸比例
    fig_width = 2 + num_samples * 0.8
    fig_height = num_classes * 0.75
    fig, axes = plt.subplots(num_classes, num_samples + 1, 
                              figsize=(fig_width, fig_height),
                              gridspec_kw={'width_ratios': [1.5] + [1]*num_samples})
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    for class_idx in range(num_classes):
        class_name = CLASSES[class_idx]
        chinese_label = CHINESE_LABELS[class_name]
        
        # 第一列显示类别标签
        axes[class_idx, 0].text(0.5, 0.5, chinese_label, 
                                 fontproperties=font_prop,
                                 ha='center', va='center',
                                 fontsize=14, fontweight='bold',
                                 color='black')
        axes[class_idx, 0].axis('off')
        
        # 显示该类别的样本图像
        for img_idx in range(num_samples):
            ax = axes[class_idx, img_idx + 1]
            img = samples[class_idx][img_idx]
            ax.imshow(img)
            ax.axis('off')
            # 添加细边框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)
    
    # 设置白色背景
    fig.patch.set_facecolor('white')
    
    # 保存高分辨率图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"图片已保存到: {save_path}")
    plt.close()

def main():
    print("正在加载 CIFAR-10 数据集...")

    if USE_TORCH:
        print("使用 torchvision 加载数据集")
        images, labels = load_cifar10_torch()
    else:
        print("使用 tensorflow 加载数据集")
        images, labels = load_cifar10_tf()

    print(f"数据集大小: {len(images)} 张图片")

    num_samples = 8
    samples = get_samples_per_class(images, labels, num_samples)

    base_dir = os.path.dirname(__file__)
    num_classes = 10
    fig_width = 2 + num_samples * 0.8
    fig_height = num_classes * 0.75
    fig, axes = plt.subplots(num_classes, num_samples + 1,
                              figsize=(fig_width, fig_height),
                              gridspec_kw={'width_ratios': [1.5] + [1]*num_samples})

    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.02, right=0.98, top=0.98, bottom=0.02)

    for class_idx in range(num_classes):
        class_name = CLASSES[class_idx]
        chinese_label = CHINESE_LABELS[class_name]

        axes[class_idx, 0].text(0.5, 0.5, chinese_label,
                                 fontproperties=font_prop,
                                 ha='center', va='center',
                                 fontsize=14, fontweight='bold',
                                 color='black')
        axes[class_idx, 0].axis('off')

        for img_idx in range(num_samples):
            ax = axes[class_idx, img_idx + 1]
            img = samples[class_idx][img_idx]
            ax.imshow(img)
            ax.axis('off')

    fig.patch.set_facecolor('white')

    png_path = os.path.join(base_dir, 'cifar10_samples.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PNG 已保存到: {png_path}")

    pdf_path = os.path.join(base_dir, 'cifar10_samples.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF 已保存到: {pdf_path}")

    plt.close()

if __name__ == '__main__':
    main()

