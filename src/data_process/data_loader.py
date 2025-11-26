import os
import sys
import shutil
import ssl
import urllib.request
import yaml  # 替换 argparse
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 临时禁用 SSL 证书验证以解决 WSL 环境中的证书问题
# 注意：这仅用于下载数据集，不应用于生产环境
ssl._create_default_https_context = ssl._create_unverified_context

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import get_logger

logger = get_logger()

# ===== 全局参数 =====
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
TEST_SIZE = 0.2

def get_dataset(dataset_name, download_dir, normalize=True):
    """根据名称加载数据集"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
        ])
        train_dataset = datasets.MNIST(root=download_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if normalize else transforms.Lambda(lambda x: x)
        ])
        train_dataset = datasets.CIFAR10(root=download_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=download_dir, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    def dataset_to_numpy(dataset):
        images = []
        labels = []
        logger.info(f"正在转换 {dataset_name} 数据集为 Numpy 数组...")
        for img, label in dataset:
            img_np = img.numpy() 
            images.append(img_np)
            labels.append(label)
        return np.array(images), np.array(labels)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    
    logger.info(f"{dataset_name} 加载完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
    return X_train, y_train, X_test, y_test

def split_data_for_federation(X, y, dataset_name, num_clients=3):
    """将数据集随机分为num_clients份,每个客户端再划分训练集和测试集"""
    logger.info(f"开始为 {num_clients} 个客户端分割 {dataset_name} 数据集...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    
    for i, idx in enumerate(split_indices):
        client_id = i + 1   
        client_dir = os.path.join(DATA_ROOT, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        
        # 获取该客户端的数据
        X_part = X[idx]
        y_part = y[idx]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=TEST_SIZE, random_state=42
        )
        
        # 保存文件名保持一致，或者带上前缀？
        # 为了兼容现有代码，我们还是使用 generic names 或者在 client 端根据配置加载
        # 考虑到 plan 说 "保持现有的 .npz 文件存储格式，以便客户端读取"，但原文件名为 mnist_train.npz
        # 如果切换数据集，文件名叫 mnist_train.npz 会很奇怪。
        # 但如果改名，client_grpc.py 中的 load_client_data 也需要修改。
        # Plan Step 5 says: "修改 server_grpc.py 和 client_grpc.py 以支持动态模型和数据加载"
        # So I should probably use a generic name like 'data_train.npz' or include the dataset name.
        # Let's use f'{dataset_name}_train.npz' and update client code later.
        
        train_path = os.path.join(client_dir, f'{dataset_name}_train.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
        
        test_path = os.path.join(client_dir, f'{dataset_name}_test.npz')
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

def main():
    # 从配置文件加载参数
    config_path = os.path.join(PROJECT_ROOT, 'src', 'default.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件时出错: {e}")
        return

    dataset_name = config['data']['dataset']
    num_clients = config['federation']['expected_clients']
    
    download_dir = os.path.join(DATA_ROOT, f'{dataset_name}_raw')

    logger.info(f"开始处理 {dataset_name} 数据集...")
    X_train, y_train, X_test, y_test = get_dataset(dataset_name, download_dir, normalize=True)
    
    # 合并训练集和测试集用于联邦学习划分
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    # 为联邦学习划分数据
    split_data_for_federation(X, y, dataset_name, num_clients=num_clients)
    
    # 删除原始数据
    delete_raw_data(download_dir)
    
    logger.info(f"{dataset_name} 数据处理完成！")

if __name__ == '__main__':
    main()

