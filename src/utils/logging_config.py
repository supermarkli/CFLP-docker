import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(create_file=False):
    """设置日志配置，包括控制台和文件输出
    
    配置两个处理器：
    1. 控制台处理器：只显示 INFO 级别以上的简要信息
    2. 文件处理器：记录 DEBUG 级别以上的详细信息
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 如果logger已经有处理器,不重复添加
    if logger.handlers:
        return logger
        
    # 创建logs目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"
    
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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 始终添加控制台 handler
    logger.addHandler(console_handler)

    # 创建文件处理器（记录详细信息）
    if create_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # 记录初始信息
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger

def get_logger(create_file=False):
    """获取logger实例"""
    logger = logging.getLogger()
    if not logger.handlers:
        logger = setup_logging(create_file)
    return logger


