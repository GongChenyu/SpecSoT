#!/bin/bash

# SP+PP分布式推理启动脚本
# 用于在三台设备上启动分布式推理

# 配置参数
MODEL_PATH="${1:-/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B}"
MASTER_ADDR="${2:-localhost}"
MASTER_PORT="${3:-29500}"
CHUNK_SIZE="${4:-128}"
SYNC_STRATEGY="${5:-pairwise}"  # pairwise 或 ring
DEVICE_MODE="${6:-single_node}"  # single_node 或 multi_node
WORLD_SIZE=3

PROMPT="请详细介绍一下人工智能的发展历史，包括其起源、重要里程碑、关键技术突破、主要应用领域，以及未来的发展趋势。"
MAX_NEW_TOKENS=200

echo "=========================================="
echo "分布式推理启动脚本"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "Chunk大小: $CHUNK_SIZE"
echo "同步策略: $SYNC_STRATEGY"
echo "设备模式: $DEVICE_MODE"
echo "设备数量: $WORLD_SIZE"
echo "=========================================="

# 检查是否在单机多卡模式
if [ "$MASTER_ADDR" == "localhost" ] || [ "$MASTER_ADDR" == "127.0.0.1" ]; then
    echo "单机多卡模式"
    
    # 在后台启动Rank 0
    CUDA_VISIBLE_DEVICES=0 python distributed_inference.py \
        --model_path "$MODEL_PATH" \
        --rank 0 \
        --world_size $WORLD_SIZE \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        --chunk_size $CHUNK_SIZE \
        --sync_strategy "$SYNC_STRATEGY" \
        --device_mode "$DEVICE_MODE" \
        --prompt "$PROMPT" \
        --max_new_tokens $MAX_NEW_TOKENS \
        > logs/rank0.log 2>&1 &
    PID0=$!
    echo "Rank 0 已启动 (PID: $PID0)"
    
    # 在后台启动Rank 1
    CUDA_VISIBLE_DEVICES=1 python distributed_inference.py \
        --model_path "$MODEL_PATH" \
        --rank 1 \
        --world_size $WORLD_SIZE \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        --chunk_size $CHUNK_SIZE \
        --sync_strategy "$SYNC_STRATEGY" \
        --device_mode "$DEVICE_MODE" \
        --prompt "$PROMPT" \
        --max_new_tokens $MAX_NEW_TOKENS \
        > logs/rank1.log 2>&1 &
    PID1=$!
    echo "Rank 1 已启动 (PID: $PID1)"
    
    # 在后台启动Rank 2
    CUDA_VISIBLE_DEVICES=2 python distributed_inference.py \
        --model_path "$MODEL_PATH" \
        --rank 2 \
        --world_size $WORLD_SIZE \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        --chunk_size $CHUNK_SIZE \
        --sync_strategy "$SYNC_STRATEGY" \
        --device_mode "$DEVICE_MODE" \
        --prompt "$PROMPT" \
        --max_new_tokens $MAX_NEW_TOKENS \
        > logs/rank2.log 2>&1 &
    PID2=$!
    echo "Rank 2 已启动 (PID: $PID2)"
    
    echo "=========================================="
    echo "所有进程已启动"
    echo "Rank 0 PID: $PID0"
    echo "Rank 1 PID: $PID1"
    echo "Rank 2 PID: $PID2"
    echo "=========================================="
    echo "查看日志："
    echo "  tail -f logs/rank0.log"
    echo "  tail -f logs/rank1.log"
    echo "  tail -f logs/rank2.log"
    echo "=========================================="
    
    # 等待所有进程完成
    wait $PID0 $PID1 $PID2
    
    echo "所有进程已完成"
    
else
    echo "多机模式"
    echo "请在每台机器上手动运行对应的命令："
    echo ""
    echo "# 机器1 (Rank 0):"
    echo "python distributed_inference.py --model_path $MODEL_PATH --rank 0 --world_size $WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT --chunk_size $CHUNK_SIZE --sync_strategy $SYNC_STRATEGY --device_mode $DEVICE_MODE --prompt \"$PROMPT\" --max_new_tokens $MAX_NEW_TOKENS"
    echo ""
    echo "# 机器2 (Rank 1):"
    echo "python distributed_inference.py --model_path $MODEL_PATH --rank 1 --world_size $WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT --chunk_size $CHUNK_SIZE --sync_strategy $SYNC_STRATEGY --device_mode $DEVICE_MODE --prompt \"$PROMPT\" --max_new_tokens $MAX_NEW_TOKENS"
    echo ""
    echo "# 机器3 (Rank 2):"
    echo "python distributed_inference.py --model_path $MODEL_PATH --rank 2 --world_size $WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT --chunk_size $CHUNK_SIZE --sync_strategy $SYNC_STRATEGY --device_mode $DEVICE_MODE --prompt \"$PROMPT\" --max_new_tokens $MAX_NEW_TOKENS"
fi
