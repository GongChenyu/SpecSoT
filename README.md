# Speculative-Decoding-Enabled-Skeleton-of-Thought


单机集中式
python run_specsot_unified.py --distributed False

单机多卡分布式
python run_specsot_unified.py \
  --distributed True \
  --world_size 3 \
  --gpu_ids 5,6,7 \
  --layer_splits 14,28


多机多卡分布式（待完善，希望可以通过shell控制）
A（主节点）
python run_specsot_unified.py \
  --role worker \
  --distributed True \
  --rank 0 \
  --world_size 2 \
  --gpu_ids 0 \
  --layer_splits 14 \
  --master_ip 192.168.1.10  # 关键：指定主节点 IP


B（工作节点）
python run_specsot_unified.py \
  --role worker \
  --distributed True \
  --rank 1 \
  --world_size 2 \
  --gpu_ids 0 \
  --layer_splits 14 \
  --master_ip 192.168.1.10  # 关键：指向机器 A 的 IP











