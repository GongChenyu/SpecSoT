#!/bin/bash
# ==============================================================================
# SpecSoT Master SSH 配置脚本
# 用于快速配置 Master 到所有 Worker 的 SSH 免密登录
#
# 使用方法:
#   chmod +x setup_master_ssh.sh
#   ./setup_master_ssh.sh
# ==============================================================================

set -e

# ==============================================================================
# 配置参数 (请根据实际情况修改)
# ==============================================================================

# Worker 机器列表 (IP地址，空格分隔)
WORKER_IPS=("192.168.1.101" "192.168.1.102" "192.168.1.103")

# Worker 用户名 (默认与当前用户相同)
WORKER_USER="${USER}"

# SSH 密钥路径
SSH_KEY="$HOME/.ssh/id_rsa"

# ==============================================================================
# 颜色输出
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${BLUE}>>> $1${NC}"; }

# ==============================================================================
# 1. 生成 SSH 密钥
# ==============================================================================
generate_ssh_key() {
    log_step "检查 SSH 密钥"
    
    if [ ! -f "$SSH_KEY" ]; then
        log_info "生成新的 SSH 密钥对..."
        ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_KEY"
    else
        log_info "SSH 密钥已存在: $SSH_KEY"
    fi
    
    chmod 600 "$SSH_KEY"
    chmod 644 "${SSH_KEY}.pub"
}

# ==============================================================================
# 2. 分发公钥到 Worker
# ==============================================================================
distribute_keys() {
    log_step "分发公钥到 Worker 机器"
    
    for ip in "${WORKER_IPS[@]}"; do
        log_info "配置 ${WORKER_USER}@${ip} ..."
        
        # 检查连通性
        if ! ping -c 1 -W 2 "$ip" &> /dev/null; then
            log_warn "无法 ping 通 $ip，跳过"
            continue
        fi
        
        # 分发公钥
        if ssh-copy-id -i "${SSH_KEY}.pub" "${WORKER_USER}@${ip}" 2>/dev/null; then
            log_info "成功配置 $ip"
        else
            log_warn "配置 $ip 失败，可能需要手动输入密码"
            ssh-copy-id -i "${SSH_KEY}.pub" "${WORKER_USER}@${ip}"
        fi
    done
}

# ==============================================================================
# 3. 测试连接
# ==============================================================================
test_connections() {
    log_step "测试 SSH 连接"
    
    all_ok=true
    
    for ip in "${WORKER_IPS[@]}"; do
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${ip}" "echo 'OK'" &> /dev/null; then
            log_info "✓ ${ip} 连接成功"
        else
            log_error "✗ ${ip} 连接失败"
            all_ok=false
        fi
    done
    
    if [ "$all_ok" = true ]; then
        log_info "所有 Worker 连接测试通过!"
    else
        log_warn "部分 Worker 连接失败，请检查配置"
    fi
}

# ==============================================================================
# 4. 生成分布式启动命令
# ==============================================================================
generate_launch_cmd() {
    log_step "生成分布式启动命令"
    
    # 获取本机 IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    
    # 构建设备列表
    DEVICES="${LOCAL_IP}#0"
    for ip in "${WORKER_IPS[@]}"; do
        DEVICES="${DEVICES},${ip}#0"
    done
    
    # 计算层分割点
    NUM_WORKERS=$((${#WORKER_IPS[@]} + 1))
    
    echo ""
    echo "=========================================="
    echo "分布式启动命令示例"
    echo "=========================================="
    echo ""
    echo "# 基本分布式推理 (${NUM_WORKERS} 个设备)"
    echo "python run_specsot.py \\"
    echo "    --distributed True \\"
    echo "    --devices \"${DEVICES}\" \\"
    echo "    --layer_splits \"xxx\" \\"  # 需要根据模型调整
    echo "    --ssh_user ${WORKER_USER} \\"
    echo "    --ssh_key ${SSH_KEY} \\"
    echo "    --mode evaluation \\"
    echo "    --task planning \\"
    echo "    --num_samples 5"
    echo ""
    echo "# inference 模式"
    echo "python run_specsot.py \\"
    echo "    --mode inference \\"
    echo "    --prompt \"你的问题\""
    echo ""
    echo "=========================================="
}

# ==============================================================================
# 5. 配置 SSH Config (可选)
# ==============================================================================
setup_ssh_config() {
    log_step "配置 SSH Config (可选)"
    
    SSH_CONFIG="$HOME/.ssh/config"
    
    read -p "是否添加 Worker 到 SSH config? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        return
    fi
    
    # 备份现有配置
    if [ -f "$SSH_CONFIG" ]; then
        cp "$SSH_CONFIG" "${SSH_CONFIG}.bak"
    fi
    
    # 添加 Worker 配置
    for i in "${!WORKER_IPS[@]}"; do
        ip="${WORKER_IPS[$i]}"
        hostname="worker$((i+1))"
        
        # 检查是否已存在
        if grep -q "Host $hostname" "$SSH_CONFIG" 2>/dev/null; then
            log_info "$hostname 已在配置中"
            continue
        fi
        
        cat >> "$SSH_CONFIG" << EOF

Host ${hostname}
    HostName ${ip}
    User ${WORKER_USER}
    IdentityFile ${SSH_KEY}
    StrictHostKeyChecking no
EOF
        log_info "已添加 $hostname ($ip)"
    done
    
    chmod 600 "$SSH_CONFIG"
    log_info "SSH Config 配置完成"
    echo ""
    echo "现在可以使用以下方式连接:"
    for i in "${!WORKER_IPS[@]}"; do
        echo "  ssh worker$((i+1))"
    done
}

# ==============================================================================
# 主函数
# ==============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  SpecSoT Master SSH 配置脚本"
    echo "=============================================="
    echo ""
    
    log_info "当前配置的 Worker 列表:"
    for ip in "${WORKER_IPS[@]}"; do
        echo "  - ${WORKER_USER}@${ip}"
    done
    echo ""
    
    read -p "是否继续? (Y/n): " confirm
    if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
        exit 0
    fi
    
    generate_ssh_key
    distribute_keys
    test_connections
    setup_ssh_config
    generate_launch_cmd
    
    log_info "配置完成!"
}

main "$@"
