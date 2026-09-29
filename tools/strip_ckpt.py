"""把 ckpt 里的冻结主干剥掉，只留真正随实验变化的部分。

背景：`thymoma_local.py` / `fewshot_run.py` 存的是 `model.state_dict()`，其中
`clip.*` 是**冻结的 BiomedCLIP 主干**（约 745MB）。21 个实验 ckpt 里这部分
**逐位相同**，而真正区分实验的 `text_adapter` / `fusion_weights` / `inlayer_bank`
只有 1~4MB。主干随时能从 HuggingFace 重新下，所以剥掉它**信息零丢失**：
748MB → 约 0.5MB。

配合 `TextSideAnomalyModel.load_compat()` 用（它本来就容忍缺键），加载时
模型自己会建好主干，再把这几个张量盖上去。

用法：
    python tools/strip_ckpt.py thymoma_local.pt ...          # 写 <名字>.slim.pt
    python tools/strip_ckpt.py --in-place *.pt               # 就地替换（谨慎，先备份）
    python tools/strip_ckpt.py --check a.pt b.slim.pt        # 对比两个 ckpt 的非主干张量
"""
import argparse
import os
import sys

import torch

# 冻结主干的前缀；带这些前缀的键一律丢掉
STRIP_PREFIXES = ("clip.",)


def load_any(path, map_location="cpu"):
    """读自己的 ckpt。

    必须显式 `weights_only=False`：torch 2.6 起 `torch.load` 的默认值从 False 翻成了 True，
    而部分 ckpt（如 _smoke_ckpt/output_kfull_s0.pt）是用 legacy .tar 格式存的，
    weights_only=True 会直接抛 "Cannot use weights_only=True with files saved in the
    legacy .tar format"。本仓库的 ckpt 都是自己产出的，可信。
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def strip_one(src: str, dst: str) -> tuple:
    sd = load_any(src)
    if not isinstance(sd, dict):
        raise SystemExit(f"{src} 不是 state_dict（拿到 {type(sd)}），本工具不处理")

    keep = {k: v for k, v in sd.items() if not k.startswith(STRIP_PREFIXES)}
    dropped = [k for k in sd if k.startswith(STRIP_PREFIXES)]
    if not dropped:
        print(f"  ⚠ {src} 里没有可剥离的主干键（已经是精简格式？）")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(keep, dst)
    return len(dropped), sum(v.numel() for v in keep.values() if torch.is_tensor(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--in-place", action="store_true",
                    help="就地替换原文件（默认另存为 <名字>.slim.pt）")
    ap.add_argument("--check", action="store_true",
                    help="只对比两个 ckpt 的非主干张量是否逐位相同，不写文件")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.check:
        a, b = args.ckpts[:2]
        sa = load_any(a)
        sb = load_any(b)
        ka = {k: v for k, v in sa.items() if not k.startswith(STRIP_PREFIXES)}
        kb = {k: v for k, v in sb.items() if not k.startswith(STRIP_PREFIXES)}
        if set(ka) != set(kb):
            print(f"✗ 键不同：只在 {a} 里 {sorted(set(ka)-set(kb))[:3]}，"
                  f"只在 {b} 里 {sorted(set(kb)-set(ka))[:3]}")
            return 1
        bad = [k for k in ka if not torch.equal(ka[k], kb[k])]
        print(f"{'✓ 非主干张量逐位相同' if not bad else '✗ 有差异'}："
              f"共 {len(ka)} 个键，差异 {len(bad)} 个 {bad[:3]}")
        return 0 if not bad else 1

    total_before = total_after = 0
    for src in args.ckpts:
        before = os.path.getsize(src)
        if args.in_place:
            # 必须**先写临时文件再替换**。早先直接把 dst 设成 src，strip_one 覆盖完原文件后
            # 那句 os.remove(src) 又把刚写好的删掉，还顺手 rename 失败 —— 实测吞掉了一个 ckpt。
            tmp = src + ".tmp-strip"
            n_drop, n_keep = strip_one(src, tmp)
            os.replace(tmp, src)          # Windows 上 os.replace 可覆盖已存在文件
            dst, after = src, os.path.getsize(src)
        else:
            dst = os.path.splitext(src)[0] + ".slim.pt"
            n_drop, n_keep = strip_one(src, dst)
            after = os.path.getsize(dst)
        total_before += before
        total_after += after
        print(f"  {os.path.basename(src):32s} {before/1e6:7.1f} MB -> {after/1e6:6.2f} MB"
              f"   丢 {n_drop} 个主干键，留 {n_keep} 个参数")
    print(f"\n合计 {total_before/1e9:.2f} GB -> {total_after/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
