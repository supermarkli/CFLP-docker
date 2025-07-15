import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from torchvision import datasets, transforms
from src.utils.logging_config import get_logger
from sklearn.model_selection import train_test_split

# ===== 全局参数 =====
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'data', 'mnist_raw')
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, 'data', 'mnist_raw', 'mnist_data.npz')
NORMALIZE = True  
TEST_SIZE = 0.2

logger = get_logger()

def load_mnist_data(download_dir, normalize=True):
    """加载MNIST数据集并返回numpy数组"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
    ])
    train_dataset = datasets.MNIST(root=download_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=transform)

    def dataset_to_numpy(dataset):
        images = []
        labels = []
        for img, label in dataset:
            img_np = img.numpy()  # 保留原始 shape: [1, 28, 28]
            images.append(img_np)
            labels.append(label)
        return np.array(images), np.array(labels)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    logger.info(f"MNIST加载完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
    return X_train, y_train, X_test, y_test

def save_npz(X_train, y_train, X_test, y_test, output_path):
    """保存为npz格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    logger.info(f"已保存训练、测试集到: {output_path}")

def split_data_for_federation(X, y, num_clients=3):
    """将数据集随机分为num_clients份,每个客户端再划分训练集和测试集"""
    logger.info(f"开始为 {num_clients} 个客户端分割数据集...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    
    base_data_dir = os.path.join(PROJECT_ROOT, 'data')
    for i, idx in enumerate(split_indices):
        client_id = i + 1   
        client_dir = os.path.join(base_data_dir, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        
        # 获取该客户端的数据
        X_part = X[idx]
        y_part = y[idx]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=TEST_SIZE, random_state=42
        )
        
        # 保存训练集
        train_path = os.path.join(client_dir, 'mnist_train.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
        
        # 保存测试集
        test_path = os.path.join(client_dir, 'mnist_test.npz')
        np.savez_compressed(test_path, X_test=X_test, y_test=y_test)
        
        logger.info(f"客户端{client_id}数据已保存:")
        logger.info(f"  - 训练集: {train_path}, 样本数: {len(X_train)}")
        logger.info(f"  - 测试集: {test_path}, 样本数: {len(X_test)}")
    
    logger.info(f"为 {num_clients} 个客户端分割数据集完成。")

def delete_raw_data(directory):
    """删除指定的目录及其所有内容"""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logger.info(f"成功删除目录: {directory}")
        except OSError as e:
            logger.error(f"删除目录 {directory} 时出错: {e}")
    else:
        logger.warning(f"目录不存在，无法删除: {directory}")

def generate_mnist_data():
    logger.info("开始加载MNIST数据集...")
    X_train, y_train, X_test, y_test = load_mnist_data(DOWNLOAD_DIR, normalize=NORMALIZE)
    
    # 合并训练集和测试集用于联邦学习划分
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    # 为联邦学习划分数据
    split_data_for_federation(X, y, num_clients=3)
    
    # 删除原始数据
    delete_raw_data(DOWNLOAD_DIR)
    
    logger.info("MNIST数据处理完成！")

if __name__ == '__main__':
    logger.info("开始执行MNIST数据处理脚本...")
    generate_mnist_data()
    logger.info("脚本执行完毕。") 