import yaml
import os
from pathlib import Path

def _load_config():
    """从 default.yaml 加载配置，并使用环境变量覆盖。"""
    config_path = Path(__file__).resolve().parent.parent / 'default.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 优先使用环境变量覆盖配置
    # gRPC & Server
    config['grpc']['server_host'] = os.environ.get("GRPC_SERVER_HOST", config.get('grpc', {}).get('server_host'))
    config['grpc']['server_port'] = int(os.environ.get("GRPC_SERVER_PORT", config.get('grpc', {}).get('server_port')))
    
    # 同态加密
    env_he = os.environ.get("USE_HOMOMORPHIC_ENCRYPTION")
    if env_he is not None:
        config['encryption']['enabled'] = env_he.lower() in ('true', '1', 't')
    
    # 客户端ID (仅从环境变量获取，不由YAML设定)
    config['client_id'] = os.environ.get("CLIENT_ID")

    return config

# 全局配置对象，在模块首次导入时加载一次
config = _load_config() 