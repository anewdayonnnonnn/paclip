"""脑基线/多尺度 多种子最终评测 —— 每个 ckpt 都用**训练时的同一口径**评。

教训（2026-09-29）：config.py 的 ms_layers 默认已从单层改成 [5,8,11]，而 train.py
建 Config 时没显式设它，于是"重训基线"其实训的是多尺度；再用单层口径去评，就得出
"重训比基线差 0.08"的假结论。训练口径与评估口径必须成对，本脚本把口径写死在 RUNS 里。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.environ.get("CODE_DIR", "D:/新建文件夹"))

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics, pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

DATA = os.environ.get("BRAIN_DL", "D:/brain_dl")
MS = [5, 8, 11]

RUNS = [("基线单层(原 ckpt)", f"{DATA}/ckpt/checkpoint_epoch20.pt", []),
        ("多尺度(原 ckpt_ms)", f"{DATA}/ckpt_ms/checkpoint_epoch20.pt", MS)]
for s in (0, 1, 2):
    RUNS.append((f"基线单层(seed{s})", f"{DATA}/ckpt_base_s{s}/checkpoint_epoch20.pt", []))
for s in (0, 1, 2):
    RUNS.append((f"多尺度(seed{s})", f"{DATA}/ckpt_ms_s{s}/checkpoint_epoch20.pt", MS))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = SliceAnomalyDataset(f"{DATA}/brain_data/test", mask_root=f"{DATA}/brain_masks",
                         image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()
rois = np.stack([np.asarray(ds[i]["roi"], dtype=bool) for i in range(len(ds))])


def out_red(m):
    res = []
    for sel in (labels == 1, labels == 0):
        mm, rr = m[sel], rois[sel]
        hot = mm >= np.percentile(mm, 90, axis=(1, 2), keepdims=True)
        res.append(hot[~rr].sum() / max(1, hot.sum()) * 100)
    return res


def red_precision(m, pct=90):
    tp = fp = 0
    for i in range(len(m)):
        if masks[i].sum() == 0:
            continue
        r = rois[i] if rois[i].sum() >= 5 else np.ones_like(rois[i], bool)
        red = (m[i] >= np.percentile(m[i][r], pct)) & rois[i]
        tp += (red & (masks[i] > 0)).sum()
        fp += (red & (masks[i] <= 0)).sum()
    return tp / max(1, tp + fp) * 100


print(f"\ntest N={len(labels)}  异常={(labels == 1).sum()}  正常={(labels == 0).sum()}")
print(f"{'配置':22s} | {'ms_layers':>10} | {'imgAUROC':>8} {'Dice':>7} {'IoU':>7} "
      f"{'pxAUROC':>8} | {'脑外红斑(异常/正常)':>18} | {'红灯真病灶率':>10}")
print("-" * 118)

rows = []
for name, ckpt, ms in RUNS:
    if not os.path.exists(ckpt):
        print(f"!! 缺 {ckpt}，跳过 {name}")
        continue
    cfg = Config(device=str(device))
    cfg.ms_layers = list(ms)
    model = TextSideAnomalyModel(cfg).to(device)
    torch.serialization.add_safe_globals([Config])
    ck = torch.load(ckpt, map_location=device)
    model.text_adapter.load_state_dict(ck["adapter"])
    with torch.no_grad():
        model.fusion_weights.copy_(ck["fusion_weights"])
    model.eval()

    sc, mp = [], []
    with torch.no_grad():
        anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
        for img, _, _ in batches:
            o = model(img, anchors)
            sc.append(o["cls_probs"].cpu())
            mp.append(o["anomaly_map"].cpu())
    sc = torch.cat(sc).numpy()
    mp = torch.cat(mp).numpy()

    im = image_metrics(sc, labels)
    pm = pixel_metrics(mp, masks)
    oa, on = out_red(mp)
    rp = red_precision(mp)
    print(f"{name:22s} | {str(ms):>10} | {im['auroc']:>8.4f} {pm['dice']:>7.4f} "
          f"{pm['iou']:>7.4f} {pm['pixel_auroc']:>8.4f} | {oa:>8.1f}% /{on:>7.1f}% | "
          f"{rp:>9.1f}%")
    rows.append((name, im["auroc"], pm["dice"], pm["iou"], pm["pixel_auroc"], rp))
    del model, ck
    torch.cuda.empty_cache()

print("-" * 118)
for tag in ("基线单层", "多尺度"):
    v = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows if r[0].startswith(tag)])
    if len(v):
        m = v.mean(axis=0)
        print(f"{tag+' 多种子均值':22s} | {'':>10} | {m[0]:>8.4f} {m[1]:>7.4f} "
              f"{m[2]:>7.4f} {m[3]:>8.4f} | {'':>18} | {m[4]:>9.1f}%"
              f"   （{len(v)} 个，Dice 极差 {v[:, 1].max()-v[:, 1].min():.4f}）")
