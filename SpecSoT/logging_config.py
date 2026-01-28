# coding=utf-8
"""
SpecSoT 日志系统

设计原则：
1. 控制台只显示关键信息（INFO级别）：配置、阶段开始/完成、统计结果
2. 文件记录所有详细信息（DEBUG级别）：tensor、通信、中间状态
3. 实时刷新，避免输出延迟

日志模块划分：
- Model: 模型加载和推理核心逻辑
- Comm: 分布式通信
- Draft: Draft tree 生成

使用方式：
    from SpecSoT.logging_config import get_logger
    logger = get_logger(rank=0)  # 或 rank=-1 表示单机
    logger.info("关键信息")      # 显示在控制台和文件
    logger.debug("详细信息")     # 只写入文件
"""

import os
import sys
import logging
from typing import Optional

# 全局缓存
_loggers = {}


class FlushHandler(logging.StreamHandler):
    """带自动刷新的 Handler"""
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    """带自动刷新的文件 Handler"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(
    rank: int = -1,
    log_dir: str = None,
    name: str = "",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        rank: 进程 rank，-1 表示单机模式
        log_dir: 日志目录
        name: 日志名称后缀
        console_level: 控制台日志级别
        file_level: 文件日志级别
    
    Returns:
        配置好的 Logger
    """
    logger_name = f"SpecSoT{f'-Rank{rank}' if rank >= 0 else ''}{f'-{name}' if name else ''}"
    
    if logger_name in _loggers:
        return _loggers[logger_name]
    
    # 确定日志目录
    if log_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名
    if rank >= 0:
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
    else:
        log_file = os.path.join(log_dir, "specsot.log")
    
    # 创建 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # 控制台格式：简洁
    console_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 文件格式：详细
    file_fmt = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%H:%M:%S')
    
    # 控制台处理器
    console = FlushHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # 文件处理器
    file_handler = FlushFileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    _loggers[logger_name] = logger
    return logger


def cleanup():
    """清理所有 logger"""
    for logger in _loggers.values():
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()
    _loggers.clear()


# 向后兼容的别名
get_unified_logger = get_logger
get_comm_logger = lambda rank, log_dir=None: get_logger(rank, log_dir, "Comm", logging.WARNING)
get_prefill_logger = lambda rank, log_dir=None: get_logger(rank, log_dir, "Prefill")

# 工具函数
def get_tensor_info(tensor) -> str:
    """获取 tensor 信息字符串"""
    import torch
    if tensor is None:
        return "None"
    if not torch.is_tensor(tensor):
        return f"<{type(tensor).__name__}>"
    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, size={size_mb:.2f}MB"


__all__ = [
    'get_logger',
    'get_unified_logger',
    'get_comm_logger', 
    'get_prefill_logger',
    'get_tensor_info',
    'cleanup',
    'FlushHandler',
    'FlushFileHandler',
]
