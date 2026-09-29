"""小样本 + 层内适配器：全自动对照实验。

一条命令跑完，**全程无需人工介入**：

    python fewshot_run.py
    python fewshot_run.py --ks 5,10,20 --seeds 0,1,2 --steps 2700
    python fewshot_run.py --groups A,C            # 只跑部分组

跑四组对照（每组多种子，报 mean / 区间）：

    A  输出端 adapter + 全监督  —— 复现现有基线 Dice 0.6295，确认脚本没跑偏
    B  层内 adapter  + 全监督   —— 隔离「插在哪」这一个变量
    C  层内 adapter  + 小样本   —— K = 5 / 10 / 20 的退化曲线
    D  输出端 adapter + 小样本  —— 隔离小样本下「插在哪」是否更关键

设计要点：
  - **按优化步数对齐**（默认 2700 步 ≈ 原 15 epoch × 179 步），而不是按 epoch。
    小样本时每"epoch"只有 1 个 batch，按 epoch 对齐就没法比了。
  - 支持集只从 **train 划分**抽，按病例抽；与 test 病例断言不相交。
  - 层内 adapter 会改变文本塔，所以**每步都要重算锚点**（`encode_anchors`）——
    既是为了数值正确，也是因为复用上一步的锚点会 backward 到已释放的计算图。
  - 每步 `model.lock_backbone_eval()`：`model.train()` 会把 BERT 的 dropout(0.1) 打开，
    而评估时是关的；层内 adapter 在塔内，这个不一致会直接污染结果。
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.metrics import pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.thymoma_dataset import (
    ThymomaSliceDataset,
    case_ids,
    make_slice_splits,
    sample_support,
)

import thymoma_local as TL

NPZ_DIR = "thymoma_slices"
ORGAN = "thymoma"
PROMPT_SET = "sentence"          # 改动 7 那版整句提示词，与基线 ckpt 同源


def _c(x) -> str:
    """Windows 控制台中文乱码兜底：把不可编码字符换成 '?'。"""
    try:
        return str(x).encode(sys.stdout.encoding or "utf-8", "replace").decode(
            sys.stdout.encoding or "utf-8")
    except Exception:
        return str(x)


# ---------------------------------------------------------------------- #
# 断点续跑
# ---------------------------------------------------------------------- #
# 本机是 RTX 4060 Laptop，跑满时会把适配器拉爆直接硬断电（Kernel-Power 41），
# 已经崩过两次。所以每跑完一组就立刻落盘 —— 不能等 12 组全跑完再统一写，
# 那样一崩就是 0 产出。
RESULTS_JSONL = "_fewshot_results.jsonl"


def load_done(path: str) -> Dict[tuple, dict]:
    """读已完成的 (mode, k, seed) → 结果，崩溃后重跑自动跳过。"""
    done: Dict[tuple, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # 崩在写一半的那行：丢掉，重跑这一组
                continue
            done[(r["mode"], r["k"], r["seed"])] = r
    return done


def append_result(path: str, r: dict) -> None:
    """追加一行并 fsync —— 断电掉的是这一行还是下一行，不能是已完成的这行。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------- #
# 建模
# ---------------------------------------------------------------------- #
def build_model(device, levels, mode: str, seed: int, bottleneck: int = 64):
    """mode: 'output' = 只用输出端 adapter（现有做法）；'inlayer' = 只用层内 adapter。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    inlayer = None
    if mode == "inlayer":
        inlayer = {"organs": [ORGAN], "organ": ORGAN, "bottleneck": bottleneck,
                   "layers": [8, 9, 10, 11], "positions": ("attn", "ffn")}
    cfg = Config(device=str(device), levels=levels)
    model = TextSideAnomalyModel(cfg, inlayer=inlayer).to(device)

    if mode == "inlayer":
        # 组 B/C 要与组 A/D 只差「插在哪」一个变量 → 把输出端 adapter 压成恒等并冻住
        with torch.no_grad():
            model.text_adapter.lambda_t.data.zero_()
        for p in model.text_adapter.parameters():
            p.requires_grad = False
    return cfg, model


# ---------------------------------------------------------------------- #
# 训练 / 评估
# ---------------------------------------------------------------------- #
def train(model, prompts, support_files, device, steps: int, batch_size: int,
          lr: float = 1e-4, seed: int = 0, tag: str = ""):
    torch.manual_seed(seed)
    ds = ThymomaSliceDataset(NPZ_DIR, files=support_files)
    bs = max(1, min(batch_size, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    crit = TotalLoss(margin=0.3, w_text=1.0, w_global=0.0, w_local=1.0,
                     margin_lo=-0.3, w_div=0.0, w_level=0.0)

    t0 = time.time()
    it = iter(loader)
    run = 0.0
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        model.train()
        model.lock_backbone_eval()          # 主干锁 eval，只让适配器 train
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        enc = model.encode_anchors(prompts)   # ← 每步重算（层内 adapter 改变了文本塔）
        out = model(images, enc)
        loss = crit(enc, out, labels, masks)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        run += loss["total"].item()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    [{tag}] step {step+1}/{steps} loss={run/(step+1):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return run / steps


@torch.no_grad()
def evaluate(model, prompts, files, device, batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(ThymomaSliceDataset(NPZ_DIR, files=files),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    enc = model.encode_anchors(prompts)
    maps, masks = [], []
    for batch in loader:
        out = model(batch["image"].to(device), enc)
        maps.append(out["anomaly_map"].cpu())
        masks.append(batch["mask"])
    return pixel_metrics(torch.cat(maps).numpy(), torch.cat(masks).numpy())


# ---------------------------------------------------------------------- #
# 单组
# ---------------------------------------------------------------------- #
def run_one(mode: str, k: Optional[int], seed: int, device, splits, prompts,
            steps: int, batch_size: int, ckpt_dir: str, heat_dir: Optional[str]):
    cnt = "全监督" if k is None else f"K={k}"
    tag = f"{'层内' if mode == 'inlayer' else '输出端'}/{cnt}/s{seed}"
    print(f"\n{'='*72}\n[{tag}] 开始（支持集 {'全部' if k is None else k} 张）\n{'='*72}", flush=True)

    support = sample_support(splits["train"], k, seed)
    sup_cases = case_ids(support)
    test_cases = case_ids(splits["test"])
    leak = sup_cases & test_cases
    assert not leak, f"支持集与 test 病例重叠，会泄漏：{leak}"

    cfg, model = build_model(device, prompts.levels, mode, seed)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  支持集 {len(support)} 张（{len(sup_cases)} 个病例），可训练参数 {n_train}")

    loss = train(model, prompts, support, device, steps, batch_size, seed=seed, tag=tag)
    m = evaluate(model, prompts, splits["test"], device, batch_size)
    print(f"  → Dice={m['dice']:.4f}  AUROC={m['pixel_auroc']:.4f}  IoU={m['iou']:.4f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    safe = f"{mode}_k{k if k is not None else 'full'}_s{seed}"
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{safe}.pt"))
    if heat_dir:
        TL.prepare_heatmap_dir(heat_dir, clean=False)
        calib = TL.fit_calibration(model, prompts, splits, device, split="test",
                                   batch_size=batch_size)
        TL.save_heatmaps(model, prompts, splits["test"], device, heat_dir, n=6, calib=calib)

    return {"mode": mode, "k": k, "seed": seed, "n_support": len(support),
            "n_trainable": n_train, "loss": loss,
            "dice": m["dice"], "auroc": m["pixel_auroc"], "iou": m["iou"]}


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2700,
                    help="每组每种的优化步数（默认 2700 ≈ 原 15 epoch × 179 步）")
    ap.add_argument("--ks", type=str, default="5",
                    help="小样本的 K 列表（默认只跑 K=5：一次只用支持集里 5 张切片）")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--groups", type=str, default="A,B,C,D")
    ap.add_argument("--ckpt-dir", type=str, default="fewshot_ckpt")
    ap.add_argument("--no-heatmaps", action="store_true")
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="每组之间等待秒数：让供电与温度回落，削掉连续满载的瞬态尖峰。"
                         "只影响总耗时，**不改变任何训练结果**。0 = 关闭")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略 _fewshot_results.jsonl 从头重跑（默认自动跳过已完成的组）")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    groups = {g.strip().upper() for g in args.groups.split(",")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = TL.PROMPT_SETS[PROMPT_SET]
    splits = make_slice_splits(NPZ_DIR, seed=0)
    print(f"[setup] device={device} 提示词={PROMPT_SET} 训练切片={len(splits['train'])} "
          f"test={len(splits['test'])}")

    # 组定义：(标签, mode, K 列表)
    plan = []
    if "A" in groups:
        plan.append(("A 输出端 + 全监督", "output", [None]))
    if "B" in groups:
        plan.append(("B 层内 + 全监督", "inlayer", [None]))
    if "C" in groups:
        plan.append(("C 层内 + 小样本", "inlayer", ks))
    if "D" in groups:
        plan.append(("D 输出端 + 小样本", "output", ks))

    # 先摊平成任务表：要判断「是不是最后一组」，最后一组跑完不必再冷却
    jobs = [(label, mode, k, seed)
            for label, mode, klist in plan
            for k in klist
            for seed in seeds]

    done = {} if args.fresh else load_done(RESULTS_JSONL)
    if done:
        print(f"[resume] 已有 {len(done)} 组完成，自动跳过（加 --fresh 可强制重跑）")

    results = []
    for i, (label, mode, k, seed) in enumerate(jobs):
        if (mode, k, seed) in done:
            r = done[(mode, k, seed)]
            print(f"[跳过] {label} seed={seed} 已完成 Dice={r['dice']:.4f}", flush=True)
            results.append(r)
            continue

        heat = None
        if not args.no_heatmaps and seed == seeds[0]:
            heat = f"heatmaps_fewshot_{mode}_k{k if k is not None else 'full'}"
        r = run_one(mode, k, seed, device, splits, prompts,
                    args.steps, args.batch_size, args.ckpt_dir, heat)
        append_result(RESULTS_JSONL, r)          # ← 先落盘，再冷却
        results.append(r)

        if args.cooldown > 0 and i < len(jobs) - 1:
            print(f"[cooldown] 等待 {args.cooldown:.0f}s 让供电/温度回落…", flush=True)
            time.sleep(args.cooldown)

    # ---------- 汇总 ----------
    lines = ["", "=" * 84,
             f"小样本 + 层内适配器 对照汇总（提示词={PROMPT_SET}，步数={args.steps}，"
             f"种子={seeds}）",
             "=" * 84,
             f"{'配置':<26}{'支持集':>7}{'可训练参数':>12}{'Dice':>20}{'AUROC':>12}",
             "-" * 84]
    print("\n".join(lines))
    for label, mode, klist in plan:
        for k in klist:
            rs = [r for r in results if r["mode"] == mode and r["k"] == k]
            if not rs:
                continue
            d = np.array([r["dice"] for r in rs])
            a = np.array([r["auroc"] for r in rs])
            cnt = "全部" if k is None else str(k)
            row = (f"{label:<26}{cnt:>7}{rs[0]['n_trainable']:>12}"
                   f"{d.mean():>10.4f} [{d.min():.4f},{d.max():.4f}]"
                   f"{a.mean():>12.4f}")
            print(row)
            lines.append(row)
    ref = [r for r in results if r["mode"] == "output" and r["k"] is None]
    if ref:
        d = np.array([r["dice"] for r in ref])
        tail = (f"\n基线参考：输出端 + 全监督 Dice = {d.mean():.4f} "
                f"[{d.min():.4f}, {d.max():.4f}]（docx 改动 7/8 记录 0.6275）")
        print(tail)
        lines.append(tail)

    with open("_fewshot_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[done] 汇总已写入 _fewshot_summary.txt")


if __name__ == "__main__":
    main()
