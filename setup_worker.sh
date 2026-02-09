#!/bin/bash
# ==============================================================================
# SpecSoT Worker 环境配置脚本
# 适用于: NVIDIA Jetson AGX Orin (Ubuntu, ARM64, Python 3.9)
# 
# 使用方法:
#   chmod +x setup_worker.sh
#   ./setup_worker.sh
#
# 功能:
#   1. 配置 SSH 免密登录
#   2. 创建 Conda 环境
#   3. 安装 PyTorch (Jetson 专用版本)
#   4. 安装项目依赖
# ==============================================================================

set -e  # 遇到错误立即退出

# ==============================================================================
# 配置参数 (请根据实际情况修改)
# ==============================================================================

# Master 机器 IP (需要配置 SSH 免密登录)
MASTER_IP="192.168.1.100"

# Master 用户名
MASTER_USER="chenyu"

# 项目目录
PROJECT_DIR="/data/home/chenyu/Coding/SD+SoT"

# Conda 环境名称
CONDA_ENV_NAME="sdsot"

# Python 版本
PYTHON_VERSION="3.9"

# JetPack 版本 (根据你的 Jetson 设备选择)
# JetPack 5.x 对应 L4T R35.x
# JetPack 6.x 对应 L4T R36.x
JETPACK_VERSION="5"

# 国内镜像源
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
CONDA_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda"

# ==============================================================================
# 颜色输出
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# ==============================================================================
# 1. 系统检查
# ==============================================================================
check_system() {
    log_step "Step 1: 系统检查"
    
    # 检查是否为 ARM64 架构
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        log_warn "当前架构为 $ARCH，此脚本针对 Jetson ARM64 设备优化"
    else
        log_info "检测到 ARM64 架构 (Jetson)"
    fi
    
    # 检查 JetPack 版本
    if [ -f /etc/nv_tegra_release ]; then
        log_info "Tegra 版本信息:"
        cat /etc/nv_tegra_release
    fi
    
    # 检查 CUDA 版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_info "CUDA 版本: $CUDA_VERSION"
    else
        log_warn "未检测到 CUDA，请确保已安装 JetPack"
    fi
    
    # 检查 GPU
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
}

# ==============================================================================
# 2. SSH 配置
# ==============================================================================
setup_ssh() {
    log_step "Step 2: SSH 免密登录配置"
    
    SSH_DIR="$HOME/.ssh"
    AUTHORIZED_KEYS="$SSH_DIR/authorized_keys"
    
    # 创建 .ssh 目录
    if [ ! -d "$SSH_DIR" ]; then
        mkdir -p "$SSH_DIR"
        chmod 700 "$SSH_DIR"
        log_info "创建 .ssh 目录"
    fi
    
    # 生成 SSH 密钥 (如果不存在)
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        log_info "生成 SSH 密钥对..."
        ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_DIR/id_rsa"
    else
        log_info "SSH 密钥已存在"
    fi
    
    # 设置权限
    chmod 600 "$SSH_DIR/id_rsa"
    chmod 644 "$SSH_DIR/id_rsa.pub"
    
    # 确保 authorized_keys 文件存在
    touch "$AUTHORIZED_KEYS"
    chmod 600 "$AUTHORIZED_KEYS"
    
    # 配置 SSH 服务
    log_info "配置 SSH 服务..."
    
    # 确保 SSH 服务运行
    if command -v systemctl &> /dev/null; then
        sudo systemctl enable ssh 2>/dev/null || true
        sudo systemctl start ssh 2>/dev/null || true
    fi
    
    log_info "SSH 配置完成"
    log_warn "请在 Master 机器上执行以下命令来配置免密登录:"
    echo ""
    echo "  ssh-copy-id $(whoami)@$(hostname -I | awk '{print $1}')"
    echo ""
    echo "或手动将 Master 的公钥添加到本机的 $AUTHORIZED_KEYS"
}

# ==============================================================================
# 3. 配置 Conda 镜像源
# ==============================================================================
setup_conda_mirror() {
    log_step "Step 3: 配置 Conda 镜像源"
    
    CONDARC="$HOME/.condarc"
    
    cat > "$CONDARC" << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - ${CONDA_MIRROR}/pkgs/main
  - ${CONDA_MIRROR}/pkgs/r
  - ${CONDA_MIRROR}/pkgs/msys2
custom_channels:
  conda-forge: ${CONDA_MIRROR}/cloud
  pytorch: ${CONDA_MIRROR}/cloud
EOF
    
    log_info "Conda 镜像源配置完成: $CONDARC"
}

# ==============================================================================
# 4. 配置 pip 镜像源
# ==============================================================================
setup_pip_mirror() {
    log_step "Step 4: 配置 pip 镜像源"
    
    PIP_DIR="$HOME/.pip"
    PIP_CONF="$PIP_DIR/pip.conf"
    
    mkdir -p "$PIP_DIR"
    
    cat > "$PIP_CONF" << EOF
[global]
index-url = ${PIP_MIRROR}
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
    
    log_info "pip 镜像源配置完成: $PIP_CONF"
}

# ==============================================================================
# 5. 创建 Conda 环境
# ==============================================================================
create_conda_env() {
    log_step "Step 5: 创建 Conda 环境"
    
    # 检查 Conda 是否安装
    if ! command -v conda &> /dev/null; then
        log_error "Conda 未安装，请先安装 Miniconda/Anaconda"
        log_info "安装 Miniconda (ARM64):"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        echo "  bash Miniconda3-latest-Linux-aarch64.sh"
        exit 1
    fi
    
    # 初始化 Conda
    source $(conda info --base)/etc/profile.d/conda.sh
    
    # 检查环境是否存在
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Conda 环境 '${CONDA_ENV_NAME}' 已存在"
        read -p "是否重建环境? (y/N): " rebuild
        if [ "$rebuild" = "y" ] || [ "$rebuild" = "Y" ]; then
            conda env remove -n "$CONDA_ENV_NAME" -y
            conda create -n "$CONDA_ENV_NAME" python="${PYTHON_VERSION}" -y
        fi
    else
        log_info "创建 Conda 环境: ${CONDA_ENV_NAME} (Python ${PYTHON_VERSION})"
        conda create -n "$CONDA_ENV_NAME" python="${PYTHON_VERSION}" -y
    fi
    
    # 激活环境
    conda activate "$CONDA_ENV_NAME"
    log_info "已激活环境: $CONDA_ENV_NAME"
    log_info "Python 路径: $(which python)"
    log_info "Python 版本: $(python --version)"
}

# ==============================================================================
# 6. 安装 PyTorch (Jetson 专用)
# ==============================================================================
install_pytorch_jetson() {
    log_step "Step 6: 安装 PyTorch (Jetson 专用)"
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
    
    log_info "检测 JetPack/L4T 版本..."
    
    # 根据 JetPack 版本选择 PyTorch wheel
    # 参考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
    
    if [ "$JETPACK_VERSION" = "6" ]; then
        # JetPack 6.x (L4T R36.x) - CUDA 12.x
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl"
        log_warn "JetPack 6.x 需要 Python 3.10，当前脚本配置为 Python 3.9"
        log_warn "请手动从 NVIDIA 官网下载对应版本"
    else
        # JetPack 5.x (L4T R35.x) - CUDA 11.4
        # PyTorch 2.1.0 for JetPack 5.1.2
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp39-cp39-linux_aarch64.whl"
    fi
    
    log_info "尝试从 NVIDIA 官方源安装 PyTorch..."
    
    # 先安装依赖
    pip install numpy cython
    
    # 尝试安装 PyTorch
    if pip install "$TORCH_URL" 2>/dev/null; then
        log_info "PyTorch 安装成功 (NVIDIA 官方 wheel)"
    else
        log_warn "无法从 NVIDIA 源安装，尝试使用 pip 安装..."
        log_info "对于 Jetson，建议从 NVIDIA 官网手动下载 wheel 文件"
        echo ""
        echo "下载地址: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo ""
        
        # 尝试 pip 安装 (可能不支持 CUDA)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        log_warn "已安装 CPU 版本 PyTorch，如需 GPU 支持请手动安装 Jetson 版本"
    fi
    
    # 验证安装
    log_info "验证 PyTorch 安装..."
    python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
}

# ==============================================================================
# 7. 安装项目依赖
# ==============================================================================
install_dependencies() {
    log_step "Step 7: 安装项目依赖"
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
    
    # 升级 pip
    pip install --upgrade pip setuptools wheel
    
    # 安装核心依赖 (使用国内源)
    log_info "安装 Transformers 生态..."
    pip install -i "$PIP_MIRROR" \
        transformers>=4.40.0 \
        accelerate>=0.26.0 \
        huggingface_hub>=0.20.0 \
        sentencepiece>=0.1.99 \
        tokenizers>=0.15.0
    
    log_info "安装分布式通信库..."
    pip install -i "$PIP_MIRROR" pyzmq>=25.0.0
    
    log_info "安装数据处理库..."
    pip install -i "$PIP_MIRROR" \
        numpy>=1.24.0 \
        pandas>=2.0.0
    
    log_info "安装 GPU 监控库..."
    pip install -i "$PIP_MIRROR" pynvml>=11.5.0
    
    log_info "安装可视化库..."
    pip install -i "$PIP_MIRROR" \
        matplotlib>=3.7.0 \
        seaborn>=0.12.0
    
    log_info "安装序列化库..."
    pip install -i "$PIP_MIRROR" protobuf>=3.19.0
    
    # 安装项目本身 (如果有 setup.py)
    if [ -f "${PROJECT_DIR}/SpecSoT/setup.py" ]; then
        log_info "安装 SpecSoT 项目..."
        cd "${PROJECT_DIR}/SpecSoT"
        pip install -e .
    fi
    
    log_info "依赖安装完成"
}

# ==============================================================================
# 8. 验证安装
# ==============================================================================
verify_installation() {
    log_step "Step 8: 验证安装"
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
    
    log_info "Python 环境信息:"
    echo "  Python: $(python --version)"
    echo "  pip: $(pip --version)"
    echo "  路径: $(which python)"
    
    log_info "验证核心库..."
    
    python << 'EOF'
import sys
errors = []

# 检查 PyTorch
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    errors.append(f"  ✗ PyTorch: {e}")

# 检查 Transformers
try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"  ✗ Transformers: {e}")

# 检查 ZMQ
try:
    import zmq
    print(f"  ✓ PyZMQ {zmq.__version__}")
except ImportError as e:
    errors.append(f"  ✗ PyZMQ: {e}")

# 检查 Pandas
try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except ImportError as e:
    errors.append(f"  ✗ Pandas: {e}")

# 检查 pynvml
try:
    import pynvml
    print(f"  ✓ pynvml {pynvml.__version__}")
except ImportError as e:
    errors.append(f"  ✗ pynvml: {e}")

# 检查 accelerate
try:
    import accelerate
    print(f"  ✓ Accelerate {accelerate.__version__}")
except ImportError as e:
    errors.append(f"  ✗ Accelerate: {e}")

if errors:
    print("\n缺失的库:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\n所有核心库安装成功!")
EOF
}

# ==============================================================================
# 9. 创建启动脚本
# ==============================================================================
create_startup_script() {
    log_step "Step 9: 创建启动脚本"
    
    STARTUP_SCRIPT="${PROJECT_DIR}/SpecSoT/start_worker.sh"
    
    cat > "$STARTUP_SCRIPT" << EOF
#!/bin/bash
# SpecSoT Worker 启动脚本
# 此脚本由 setup_worker.sh 自动生成

# 激活 Conda 环境
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

# 设置 CUDA 设备 (根据需要修改)
export CUDA_VISIBLE_DEVICES=0

# 切换到项目目录
cd ${PROJECT_DIR}/SpecSoT

# 启动 Worker (参数由 Master 通过 SSH 传递)
exec python run_specsot.py "\$@"
EOF
    
    chmod +x "$STARTUP_SCRIPT"
    log_info "启动脚本已创建: $STARTUP_SCRIPT"
}

# ==============================================================================
# 10. 打印配置摘要
# ==============================================================================
print_summary() {
    log_step "配置完成!"
    
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    
    echo "=========================================="
    echo "配置摘要"
    echo "=========================================="
    echo ""
    echo "本机信息:"
    echo "  IP 地址: $LOCAL_IP"
    echo "  用户名: $(whoami)"
    echo "  Conda 环境: $CONDA_ENV_NAME"
    echo "  Python 路径: $(conda run -n $CONDA_ENV_NAME which python)"
    echo ""
    echo "在 Master 机器上配置分布式:"
    echo "  1. 配置 SSH 免密登录:"
    echo "     ssh-copy-id $(whoami)@$LOCAL_IP"
    echo ""
    echo "  2. 测试 SSH 连接:"
    echo "     ssh $(whoami)@$LOCAL_IP 'echo OK'"
    echo ""
    echo "  3. 启动分布式推理:"
    echo "     python run_specsot.py \\"
    echo "       --distributed True \\"
    echo "       --devices \"${MASTER_IP}#0,${LOCAL_IP}#0\" \\"
    echo "       --ssh_user $(whoami) \\"
    echo "       --remote_python \"$(conda run -n $CONDA_ENV_NAME which python)\" \\"
    echo "       --remote_workdir \"${PROJECT_DIR}/SpecSoT\""
    echo ""
    echo "=========================================="
}

# ==============================================================================
# 主函数
# ==============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  SpecSoT Worker 环境配置脚本"
    echo "  适用于 NVIDIA Jetson AGX Orin"
    echo "=============================================="
    echo ""
    
    # 检查是否以 root 运行
    if [ "$EUID" -eq 0 ]; then
        log_warn "请勿使用 root 用户运行此脚本"
        log_info "请使用普通用户运行: ./setup_worker.sh"
        exit 1
    fi
    
    # 询问是否继续
    read -p "是否开始配置? (Y/n): " confirm
    if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
        log_info "已取消"
        exit 0
    fi
    
    # 执行配置步骤
    check_system
    setup_ssh
    setup_conda_mirror
    setup_pip_mirror
    create_conda_env
    install_pytorch_jetson
    install_dependencies
    verify_installation
    create_startup_script
    print_summary
}

# 运行主函数
main "$@"
