# coding=utf-8
"""
GPU 显存监控器

提供后台监控 GPU 显存峰值的功能。

使用示例：
    with GPUMemoryMonitor() as monitor:
        # 执行 GPU 计算
        ...
    print(f"峰值显存: {monitor.peak_usage} MB")
"""

import time
from threading import Thread, Event
from typing import Optional

import torch
import pynvml

# =============================================================================
# GPU 显存监控器
# =============================================================================

class GPUMemoryMonitor:
    """GPU 显存监控器 - 后台监控显存峰值"""

    def __init__(self, device_index: int = None, interval: float = 0.01):
        if device_index is None and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        self.device_index = device_index
        self.interval = interval
        self.peak_usage = 0
        self._stop_event = Event()
        self._thread = None

    def _monitor(self):
        """后台监控线程"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            while not self._stop_event.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                current_usage = info.used / (1024 ** 2)
                self.peak_usage = max(self.peak_usage, current_usage)
                time.sleep(self.interval)
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def __enter__(self):
        self._stop_event.clear()
        self._thread = Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False



