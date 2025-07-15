import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datetime import datetime
from pathlib import Path

# 从配置工具中导入全局配置
from src.utils.config_utils import config

def setup_logging(create_file=False):
    """设置日志配置，包括控制台和文件输出
    
    配置两个处理器：
    1. 控制台处理器：只显示 INFO 级别以上的简要信息
    2. 文件处理器：记录 DEBUG 级别以上的详细信息
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 过滤掉 matplotlib 的 DEBUG 日志
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    
    # 如果logger已经有处理器,不重复添加
    if logger.handlers:
        # 如果请求创建文件但文件处理器不存在，则继续
        if create_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            pass
        else:
            return logger
        
    # 创建logs目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建格式化器
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台处理器（只显示重要信息）
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 创建文件处理器（记录详细信息）
    if create_file:
        # 从配置中获取隐私模式
        privacy_mode = config.get('federation', {}).get('privacy_mode', 'unknown')
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"server_{privacy_mode}.log"
        log_file = log_dir / log_filename

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # 记录初始信息
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger

def get_logger(create_file=False):
    """获取logger实例。"""
    logger = logging.getLogger()
    # 仅在没有处理器或需要创建文件但文件处理器不存在时设置
    if not logger.handlers or (create_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers)):
        logger = setup_logging(create_file)
    return logger


