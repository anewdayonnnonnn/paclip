#!/bin/bash
# 四组各起一个进程并行 —— 单进程只吃到 39% GPU / 1.4G 显存，16 核 503G 内存闲着。
# 四组的 (mode, k) 组合互不相同，所以 ckpt 名与热力图目录都不会撞；
# _fewshot_results.jsonl 是共用的追加写（每行 fsync），resume 依然全局生效。
cd /root/autodl-tmp/proj || exit 1
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "===== parallel start $(date) ====="

python _smoke_inlayer.py > _par.smoke.log 2>&1
if [ $? -ne 0 ]; then
    echo "冒烟失败，不跑正式对照"; exit 1
fi
echo "smoke ok"

for g in A B C D; do
    nohup python fewshot_run.py --groups $g --seeds 0,1,2 --ks 5 \
        --steps 2700 --cooldown 0 > _par_$g.log 2>&1 &
    echo "  group $g pid=$!"
    sleep 25          # 错开启动，避免四个进程同一刻抢文本塔前向
done

wait
echo "===== parallel end $(date) ====="
python - <<'PY'
import json, collections
rows = collections.defaultdict(list)
for line in open("_fewshot_results.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    rows[(r["mode"], r["k"])].append(r["dice"])
print("\n=== 汇总 ===")
for k, v in sorted(rows.items(), key=lambda x: str(x[0])):
    mode = "层内" if k[0] == "inlayer" else "输出端"
    kk = "全部" if k[1] is None else f"K={k[1]}"
    mean = sum(v) / len(v)
    print(f"{mode:<4} {kk:<6} n={len(v)}  Dice 均值={mean:.4f}  "
          f"[{min(v):.4f},{max(v):.4f}]")
PY
