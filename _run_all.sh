#!/bin/bash
# 服务器端总入口：先冒烟验证层内适配器那条路径，通过后跑四组对照。
# 断点续跑由 fewshot_run.py 自己负责（_fewshot_results.jsonl），中断后重跑本脚本即可。
cd /root/autodl-tmp/proj || exit 1
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1          # 权重已在缓存里，禁止联网，避免长跑中途卡网络
export PYTHONUNBUFFERED=1

echo "===== start $(date) ====="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader

python _smoke_inlayer.py
smoke=$?
echo "===== smoke exit=$smoke ====="
if [ "$smoke" -ne 0 ]; then
    echo "冒烟失败，不跑正式对照"
    exit 1
fi

python fewshot_run.py \
    --groups A,B,C,D \
    --seeds 0,1,2 \
    --ks 5 \
    --steps 2700 \
    --cooldown 0

echo "===== end $(date) ====="
