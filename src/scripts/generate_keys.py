import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pickle
from datetime import datetime
import shutil
import stat
import phe

from src.utils.logging_config import get_logger

logger = get_logger()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 定义certs相关目录
CERTS_DIR = os.path.join(PROJECT_ROOT, 'certs')
CLIENT_CERTS_DIR = os.path.join(CERTS_DIR, 'client')
SERVER_CERTS_DIR = os.path.join(CERTS_DIR, 'server')
BACKUP_DIR = os.path.join(CERTS_DIR, 'backup')

def create_directories():
    """创建必要的目录"""
    try:
        for directory in [CERTS_DIR, CLIENT_CERTS_DIR, SERVER_CERTS_DIR, BACKUP_DIR]:
            os.makedirs(directory, exist_ok=True)
            # 设置目录权限为700
            os.chmod(directory, stat.S_IRWXU)
        logger.info("目录创建成功")
    except Exception as e:
        logger.error(f"目录创建失败: {str(e)}")
        raise

def generate_keypair():
    """生成Paillier密钥对"""
    try:
        public_key, private_key = phe.generate_paillier_keypair()
        logger.info("密钥对生成成功")
        return public_key, private_key
    except Exception as e:
        logger.error(f"密钥对生成失败: {str(e)}")
        raise

def save_keys(public_key, private_key):
    """保存密钥到文件"""
    try:
        # 保存公钥
        public_key_path = os.path.join(CLIENT_CERTS_DIR, 'public_key.pkl')
        with open(public_key_path, 'wb') as f:
            pickle.dump(public_key, f)
        # 设置文件权限为600
        os.chmod(public_key_path, stat.S_IRUSR | stat.S_IWUSR)
        
        # 保存私钥
        private_key_path = os.path.join(SERVER_CERTS_DIR, 'private_key.pkl')
        with open(private_key_path, 'wb') as f:
            pickle.dump(private_key, f)
        # 设置文件权限为600
        os.chmod(private_key_path, stat.S_IRUSR | stat.S_IWUSR)
        
        logger.info("密钥保存成功")
    except Exception as e:
        logger.error(f"密钥保存失败: {str(e)}")
        raise

def create_backup():
    """创建密钥备份"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(BACKUP_DIR, f'keys_backup_{timestamp}')
        os.makedirs(backup_path)
        
        # 备份公钥
        shutil.copy2(
            os.path.join(CLIENT_CERTS_DIR, 'public_key.pkl'),
            os.path.join(backup_path, 'public_key.pkl')
        )
        
        # 备份私钥
        shutil.copy2(
            os.path.join(SERVER_CERTS_DIR, 'private_key.pkl'),
            os.path.join(backup_path, 'private_key.pkl')
        )
        
        logger.info(f"密钥备份创建成功: {backup_path}")
    except Exception as e:
        logger.error(f"密钥备份创建失败: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        logger.info("开始生成密钥对...")
        
        # 创建目录
        create_directories()
        
        # 生成密钥对
        public_key, private_key = generate_keypair()
        
        # 保存密钥
        save_keys(public_key, private_key)
        
        # 创建备份
        create_backup()
        
        logger.info("密钥生成和保存完成")
        
    except Exception as e:
        logger.error(f"密钥生成过程失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 