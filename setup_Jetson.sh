# Jetson Info

# master:
# ssh doit@10.7.135.14
# pwd doit1234

# worker 1:
# ssh doit@10.7.182.160
# pwd doit1234

# worker 2:
# ssh doit@10.7.124.3
# pwd 123456

# https://docs.pytorch.org/TensorRT/getting_started/jetpack.html


apt show nvidia-jetpack
nvcc --version
ls /usr/local/cuda/lib64/libcusparseLt.so

conda create -n specsot python=3.10
conda activate specsot
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -r requirements.txt

# SSH 登录密钥
# master
ssh-keygen -t rsa   # 全部输入回车
ssh-copy-id doit@10.7.182.160
ssh-copy-id doit@10.7.124.3

# 测试
ssh doit@10.7.182.160 "date"
ssh doit@10.7.124.3 "date"






