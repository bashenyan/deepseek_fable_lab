#!/usr/bin/env bash
set -euo pipefail

cd /workspace/deepseek_fable_lab

echo "[1/4] 进入工程目录"
pwd

echo "[2/4] 创建虚拟环境（如果不存在）"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

echo "[3/4] 安装基础依赖"
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[4/4] 验证 PyTorch 与 GPU"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(idx, torch.cuda.get_device_name(idx))
PY

echo "初始化完成。"
