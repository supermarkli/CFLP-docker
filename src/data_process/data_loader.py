import os
import sys
import shutil
import ssl
import yaml
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 临时禁用 SSL 证书验证以解决 WSL 环境中的证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import get_logger

logger = get_logger()

# ===== 全局参数 =====
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


def get_dataset(dataset_name, download_dir, normalize=True):
    """根据名称加载数据集，返回官方训练集和官方测试集"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x)
        ])
        train_dataset = datasets.MNIST(root=download_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x)
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


def save_global_test_set(X_test, y_test, dataset_name):
    """保存全局测试集（用于服务端评估全局模型）"""
    global_dir = os.path.join(DATA_ROOT, 'global')
    os.makedirs(global_dir, exist_ok=True)
    
    test_path = os.path.join(global_dir, f'{dataset_name}_test.npz')
    np.savez_compressed(test_path, X_test=X_test, y_test=y_test)
    logger.info(f"全局测试集已保存: {test_path}, 样本数: {len(X_test)}")
    return test_path


def split_data_dirichlet(X, y, dataset_name, num_clients=3, alpha=0.5, num_classes=10, local_val_size=0.1):
    """
    使用 Dirichlet 分布为客户端分割 Non-IID 数据集。
    
    原理：每个客户端拥有所有类别的数据，但是比例不同。
    
    Args:
        X (np.array): 特征数据（官方训练集）。
        y (np.array): 标签数据（官方训练集）。
        dataset_name (str): 数据集名称。
        num_clients (int): 客户端数量。
        alpha (float): Dirichlet 分布参数，α 越小越 Non-IID。
        num_classes (int): 数据集类别数。
        local_val_size (float): 客户端本地验证集比例。
    """
    logger.info(f"开始为 {num_clients} 个客户端分割 Non-IID (Dirichlet, α={alpha}) 数据集...")
    
    np.random.seed(42)
    class_distributions = np.random.dirichlet([alpha] * num_clients, size=num_classes)
    
    # 按类别组织数据
    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    # 为每个客户端分配数据
    client_data_indices = {i: [] for i in range(num_clients)}
    
    for class_id in range(num_classes):
        class_data_indices = np.array(class_indices[class_id])
        n_class_samples = len(class_data_indices)
        
        if n_class_samples == 0:
            continue
        
        proportions = class_distributions[class_id]
        client_counts = (proportions * n_class_samples).astype(int)
        client_counts[-1] = n_class_samples - client_counts[:-1].sum()
        
        np.random.shuffle(class_data_indices)
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + client_counts[client_id]
            client_data_indices[client_id].extend(class_data_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 保存各客户端数据
    for client_id in range(num_clients):
        client_idx = np.array(client_data_indices[client_id])
        
        if len(client_idx) == 0:
            logger.warning(f"客户端 {client_id + 1} 没有分配到任何数据")
            continue
        
        client_dir = os.path.join(DATA_ROOT, f'client{client_id + 1}')
        os.makedirs(client_dir, exist_ok=True)
        
        X_part = X[client_idx]
        y_part = y[client_idx]
        
        # 划分本地训练集和本地验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_part, y_part, test_size=local_val_size, random_state=42, stratify=y_part
        )
        
        # 统计类别分布
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        label_distribution = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
        
        # 保存本地训练集
        train_path = os.path.join(client_dir, f'{dataset_name}_train.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
        
        # 保存本地验证集
        val_path = os.path.join(client_dir, f'{dataset_name}_val.npz')
        np.savez_compressed(val_path, X_val=X_val, y_val=y_val)
        
        logger.info(f"客户端{client_id + 1} (Dirichlet α={alpha}) 数据已保存:")
        logger.info(f"  - 本地训练集: {train_path}, 样本数: {len(X_train)}")
        logger.info(f"  - 本地验证集: {val_path}, 样本数: {len(X_val)}")
        logger.info(f"  - 类别分布: {label_distribution}")
    
    logger.info(f"为 {num_clients} 个客户端分割数据集完成。")


def delete_raw_data(directory):
    """删除指定的目录及其所有内容"""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logger.info(f"成功删除目录: {directory}")
        except OSError as e:
            logger.error(f"删除目录 {directory} 时出错: {e}")


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
    dirichlet_alpha = config['data'].get('dirichlet_alpha', 0.5)
    local_val_size = config['data'].get('local_val_size', 0.1)
    num_classes = 10 if dataset_name.lower() in ['mnist', 'cifar10'] else 10
    
    download_dir = os.path.join(DATA_ROOT, f'{dataset_name}_raw')

    logger.info(f"开始处理 {dataset_name} 数据集 (Dirichlet α={dirichlet_alpha})...")
    
    # 加载数据集
    X_train, y_train, X_test, y_test = get_dataset(dataset_name, download_dir, normalize=True)
    
    # 保存全局测试集
    save_global_test_set(X_test, y_test, dataset_name)
    
    # 使用 Dirichlet 分布划分客户端数据
    split_data_dirichlet(X_train, y_train, dataset_name, num_clients=num_clients, 
                         alpha=dirichlet_alpha, num_classes=num_classes,
                         local_val_size=local_val_size)
    
    # 删除原始数据
    delete_raw_data(download_dir)
    
    logger.info(f"{dataset_name} 数据处理完成！")


if __name__ == '__main__':
    main()
