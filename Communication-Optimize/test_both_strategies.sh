#!/bin/bash

# 简单的测试脚本 - 测试两种同步策略

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "测试1: Pairwise 同步策略"
echo "=========================================="

./launch_distributed.sh \
    /data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B \
    localhost \
    29500 \
    128 \
    pairwise \
    single_node

sleep 1

echo ""
echo "=========================================="
echo "测试2: Ring 同步策略"
echo "=========================================="

./launch_distributed.sh \
    /data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B \
    localhost \
    29501 \
    128 \
    ring \
    single_node

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
