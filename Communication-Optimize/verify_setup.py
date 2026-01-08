"""
验证脚本：检查环境和配置是否正确
"""

import sys
import torch
import torch.distributed as dist

def check_cuda():
    """检查CUDA环境"""
    print("=" * 60)
    print("检查CUDA环境")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✓ CUDA可用")
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
        
    if torch.cuda.device_count() < 3:
        print("⚠️  警告：GPU数量少于3个，单机多卡模式可能无法运行")
        
    return True

def check_nccl():
    """检查NCCL后端"""
    print("\n" + "=" * 60)
    print("检查NCCL后端")
    print("=" * 60)
    
    if not dist.is_nccl_available():
        print("❌ NCCL不可用")
        return False
        
    print("✓ NCCL可用")
    return True

def check_transformers():
    """检查transformers库"""
    print("\n" + "=" * 60)
    print("检查Transformers")
    print("=" * 60)
    
    try:
        import transformers
        print(f"✓ Transformers版本: {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers未安装")
        return False

def check_model_path():
    """检查模型路径"""
    print("\n" + "=" * 60)
    print("检查模型路径")
    print("=" * 60)
    
    import os
    
    model_paths = [
        "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B",
        "/data/home/chenyu/Coding/SD+SoT/models/Qwen2.5-7B-Instruct"
    ]
    
    found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ 找到模型: {path}")
            
            # 检查必要文件
            required_files = ["config.json", "tokenizer_config.json"]
            for file in required_files:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file}")
                else:
                    print(f"  ❌ {file} 缺失")
                    
            found = True
        else:
            print(f"✗ 未找到: {path}")
            
    if not found:
        print("⚠️  警告：未找到Qwen模型，请检查模型路径")
        
    return found

def check_distributed_files():
    """检查分布式相关文件"""
    print("\n" + "=" * 60)
    print("检查项目文件")
    print("=" * 60)
    
    import os
    
    files = [
        "modeling_qwen3_kv.py",
        "modeling_qwen3_kv_distributed.py",
        "cache_sync_manager.py",
        "distributed_inference.py",
        "launch_distributed.sh"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} 缺失")
            all_exist = False
            
    return all_exist

def check_permissions():
    """检查脚本权限"""
    print("\n" + "=" * 60)
    print("检查脚本权限")
    print("=" * 60)
    
    import os
    import stat
    
    scripts = [
        "launch_distributed.sh",
        "test_both_strategies.sh"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            st = os.stat(script)
            is_executable = bool(st.st_mode & stat.S_IXUSR)
            if is_executable:
                print(f"✓ {script} 可执行")
            else:
                print(f"⚠️  {script} 不可执行，运行: chmod +x {script}")
        else:
            print(f"✗ {script} 不存在")

def create_logs_dir():
    """创建日志目录"""
    print("\n" + "=" * 60)
    print("创建日志目录")
    print("=" * 60)
    
    import os
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("✓ 创建logs目录")
    else:
        print("✓ logs目录已存在")

def main():
    print("\n" + "=" * 60)
    print("SP+PP分布式推理环境验证")
    print("=" * 60)
    
    checks = [
        ("CUDA环境", check_cuda),
        ("NCCL后端", check_nccl),
        ("Transformers库", check_transformers),
        ("模型路径", check_model_path),
        ("项目文件", check_distributed_files),
    ]
    
    results = []
    for name, func in checks:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 检查 {name} 时出错: {e}")
            results.append((name, False))
    
    # 额外检查
    check_permissions()
    create_logs_dir()
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓" if result else "❌"
        print(f"{status} {name}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ 所有检查通过，可以开始运行分布式推理")
        print("\n快速开始:")
        print("  ./launch_distributed.sh /path/to/Qwen3-4B localhost 29500 128 pairwise")
    else:
        print("❌ 部分检查未通过，请修复上述问题后再运行")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
