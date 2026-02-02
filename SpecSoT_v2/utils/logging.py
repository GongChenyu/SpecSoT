# coding=utf-8
"""
SpecSoT 统一日志管理模块

该模块提供统一的日志管理功能，用于 SpecSoT 推理过程中的日志记录。

设计原则：
1. 单机模式：输出单个日志文件 specsot.log
2. 分布式模式：每个 rank 输出一个日志文件 rank_{rank}.log
3. 控制台只输出关键信息（INFO级别以上），详细通信日志只写入文件（DEBUG级别）
4. 实时输出：通过 flush 确保日志及时显示，解决输出延迟问题

日志级别设计：
- DEBUG: 详细的通信数据、cache同步细节、tensor信息等（只写文件）
- INFO: 关键阶段开始/结束、进度信息、关键事件（控制台+文件）
- WARNING: 超时、异常情况
- ERROR: 错误信息

使用方式：
    >>> from SpecSoT_v2.utils.logging import get_logger
    >>> logger = get_logger(rank=0, log_dir="logs")
    >>> logger.info("关键信息，会显示在控制台和文件")
    >>> logger.debug("详细信息，只写入文件")
"""

import os
import sys
import logging
from typing import Optional

# 禁用 stdout/stderr 缓冲，确保日志实时输出
# 这对于解决 decoding 阶段 print 语句输出延迟问题很重要
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


# 全局日志记录器缓存（避免重复创建）
_logger_cache = {}


class FlushHandler(logging.StreamHandler):
    """
    带有自动刷新的StreamHandler
    
    确保每条日志都立即刷新到控制台，避免输出延迟问题。
    这是解决 Python stdout 缓冲导致日志延迟显示的关键。
    """
    
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    """
    带有自动刷新的FileHandler
    
    确保每条日志都立即写入文件，避免输出延迟问题。
    """
    
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(
    rank: int = -1,
    log_dir: Optional[str] = None,
    name: str = "",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    获取统一的日志记录器
    
    该函数会返回一个配置好的logger，支持：
    - 控制台输出（只显示关键信息，默认INFO级别）
    - 文件输出（记录所有详细信息，默认DEBUG级别）
    - 实时刷新（避免输出延迟）
    
    重要说明：
    - 相同的 logger_name 会返回同一个 logger 实例
    - 这确保了不同模块（如 run_specsot.py, distributed_prefill.py）
      使用同一个 rank 时写入同一个日志文件
    
    Args:
        rank: 当前进程的rank
            - rank >= 0: 分布式模式，日志文件名为 rank_{rank}.log
            - rank = -1: 单机模式或launcher，日志文件名为 specsot.log
        log_dir: 日志目录，默认为项目根目录下的 logs 文件夹
        name: logger名称后缀，用于区分不同模块
        console_level: 控制台日志级别，默认 INFO
        file_level: 文件日志级别，默认 DEBUG
        
    Returns:
        配置好的logger对象
    """
    # 确定logger名称
    logger_name = f"SpecSoT{f'-Rank{rank}' if rank >= 0 else ''}{f'-{name}' if name else ''}"
    
    # 检查缓存 - 如果已经创建过，直接返回
    if logger_name in _logger_cache:
        return _logger_cache[logger_name]
    
    # 确定日志目录
    if log_dir is None:
        # 路径: utils/logging.py -> utils -> SpecSoT_v2 -> SpecSoT -> SD+SoT
        specsot_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        specsot_dir = os.path.dirname(specsot_v2_dir)
        project_dir = os.path.dirname(specsot_dir)  # SD+SoT/
        log_dir = os.path.join(project_dir, 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定日志文件名
    if rank >= 0:
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
    else:
        log_file = os.path.join(log_dir, "specsot.log")
    
    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的handlers（避免重复添加）
    logger.handlers.clear()
    
    # 设置日志格式
    # 控制台格式：简洁，只显示关键信息
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # 文件格式：详细，包含文件名和行号
    file_format = '%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    console_fmt = logging.Formatter(console_format, datefmt=date_format)
    file_fmt = logging.Formatter(file_format, datefmt=date_format)
    
    # 控制台处理器（带刷新）
    console_handler = FlushHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # 文件处理器（带刷新）- 使用 'w' 模式覆盖之前的日志
    file_handler = FlushFileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    # 缓存logger
    _logger_cache[logger_name] = logger
    
    # 首次创建时输出日志文件路径
    logger.info(f"日志将保存到: {log_file}")
    
    return logger


def get_comm_logger(
    rank: int,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    获取通信模块专用的日志记录器
    
    通信日志的特点：
    - 控制台只输出 WARNING 及以上级别（减少刷屏）
    - 文件输出所有级别（便于调试）
    
    Args:
        rank: 当前进程的rank
        log_dir: 日志目录
        
    Returns:
        配置好的logger对象
    """
    return get_logger(
        rank=rank,
        log_dir=log_dir,
        name="Comm",
        console_level=logging.WARNING,
        file_level=logging.DEBUG,
    )


def get_prefill_logger(
    rank: int,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    获取Prefill模块专用的日志记录器
    
    Prefill日志使用与主logger相同的配置，
    这样可以将prefill日志合并到rank日志中。
    
    Args:
        rank: 当前进程的rank
        log_dir: 日志目录
        
    Returns:
        配置好的logger对象
    """
    return get_logger(
        rank=rank,
        log_dir=log_dir,
        name="Prefill",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )


def get_tensor_info(tensor) -> str:
    """
    获取tensor的详细信息字符串
    
    用于在日志中记录tensor的形状、类型、设备和大小信息
    
    Args:
        tensor: PyTorch tensor或None
        
    Returns:
        包含形状、类型、设备、大小的信息字符串
        
    示例:
        >>> tensor = torch.randn(1, 128, 2560, dtype=torch.float16, device='cuda:0')
        >>> get_tensor_info(tensor)
        'shape=(1, 128, 2560), dtype=float16, device=cuda:0, size=0.625MB'
    """
    import torch
    
    if tensor is None:
        return "None"
    
    if not torch.is_tensor(tensor):
        return f"<{type(tensor).__name__}>"
    
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype).replace('torch.', '')
    device = str(tensor.device)
    numel = tensor.numel()
    size_bytes = tensor.element_size() * numel
    size_mb = size_bytes / (1024 * 1024)
    
    return f"shape={shape}, dtype={dtype}, device={device}, size={size_mb:.3f}MB"


def format_tensor_brief(tensor) -> str:
    """
    获取tensor的简要信息字符串
    
    用于在单行日志中快速显示tensor信息
    
    Args:
        tensor: PyTorch tensor或None
        
    Returns:
        简要的tensor信息（如 "shape=(2,512,3584)"）
    """
    import torch
    
    if tensor is None:
        return "None"
    
    if not torch.is_tensor(tensor):
        return f"<{type(tensor).__name__}>"
    
    return f"shape={tuple(tensor.shape)}"


def format_timing(elapsed_ms: float) -> str:
    """
    格式化时间信息
    
    Args:
        elapsed_ms: 耗时（毫秒）
        
    Returns:
        格式化的时间字符串
    """
    if elapsed_ms >= 1000:
        return f"{elapsed_ms/1000:.2f}s"
    else:
        return f"{elapsed_ms:.2f}ms"


def format_message_info(msg_type: str, src: int, dst: int, seq_id: int = None) -> str:
    """
    格式化消息信息
    
    Args:
        msg_type: 消息类型名称
        src: 源rank
        dst: 目标rank
        seq_id: 序列ID（可选）
        
    Returns:
        格式化的消息信息字符串
    """
    if seq_id is not None:
        return f"[{msg_type}] {src}->{dst} seq={seq_id}"
    else:
        return f"[{msg_type}] {src}->{dst}"


def log_phase_start(logger: logging.Logger, phase_name: str, **kwargs):
    """
    记录阶段开始的日志
    
    Args:
        logger: 日志记录器
        phase_name: 阶段名称
        **kwargs: 额外的键值对信息
    """
    info_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    if info_str:
        logger.info(f"[{phase_name}] 开始 | {info_str}")
    else:
        logger.info(f"[{phase_name}] 开始")


def log_phase_end(logger: logging.Logger, phase_name: str, elapsed_ms: float = None, **kwargs):
    """
    记录阶段结束的日志
    
    Args:
        logger: 日志记录器
        phase_name: 阶段名称
        elapsed_ms: 耗时（毫秒）
        **kwargs: 额外的键值对信息
    """
    parts = []
    if elapsed_ms is not None:
        parts.append(f"耗时={elapsed_ms:.2f}ms")
    parts.extend(f"{k}={v}" for k, v in kwargs.items())
    
    info_str = " | ".join(parts)
    if info_str:
        logger.info(f"[{phase_name}] 完成 | {info_str}")
    else:
        logger.info(f"[{phase_name}] 完成")


def log_progress(logger: logging.Logger, current: int, total: int, message: str = ""):
    """
    记录进度信息
    
    Args:
        logger: 日志记录器
        current: 当前进度
        total: 总数
        message: 附加消息
    """
    percent = (current / total * 100) if total > 0 else 0
    if message:
        logger.info(f"[进度] {current}/{total} ({percent:.1f}%) - {message}")
    else:
        logger.info(f"[进度] {current}/{total} ({percent:.1f}%)")


def cleanup():
    """
    清理所有缓存的日志记录器
    
    在程序结束时调用，确保所有日志被刷新并关闭
    """
    global _logger_cache
    
    for logger_name, logger in _logger_cache.items():
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()
    
    _logger_cache.clear()


# 向后兼容的别名
get_unified_logger = get_logger
cleanup_loggers = cleanup


# 导出的公共接口
__all__ = [
    # Handler 类
    'FlushHandler',
    'FlushFileHandler',
    # Logger 获取函数
    'get_logger',
    'get_unified_logger',
    'get_comm_logger',
    'get_prefill_logger',
    # 格式化工具
    'get_tensor_info',
    'format_tensor_brief',
    'format_timing',
    'format_message_info',
    # 日志辅助函数
    'log_phase_start',
    'log_phase_end',
    'log_progress',
    'cleanup',
    'cleanup_loggers',
]
