#!/usr/bin/env python3
"""
测试per-layer同步的简单脚本
"""

import sys
import subprocess
import time

def cleanup_port(port):
    """清理占用的端口"""
    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
            shell=True,
            capture_output=True,
            text=True
        )
        print(f"清理端口 {port} 完成")
    except Exception as e:
        print(f"清理端口失败: {e}")

def main():
    print("=" * 60)
    print("Per-Layer Sync 测试")
    print("=" * 60)
    
    # 清理端口
    cleanup_port(29500)
    time.sleep(1)
    
    # 运行debug launcher
    print("\n启动 debug_launcher.py ...\n")
    
    try:
        subprocess.run(
            [sys.executable, "debug_launcher.py"],
            cwd="/data/home/chenyu/Coding/SD+SoT/Communication-Optimize",
            check=True
        )
        print("\n✓ 测试成功完成")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n测试被中断")
        sys.exit(1)

if __name__ == "__main__":
    main()
