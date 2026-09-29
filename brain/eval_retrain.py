"""评估重训出来的基线，和 eval_final.py 里那行「基线单层」对表。

口径完全照抄 eval_final.py（同 root / 同 mask / 同 ms_layers=[] / 同 pixel_metrics），
只换 checkpoint —— 这样数字能不能对上就只取决于训练本身。
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
ROOT = f"{DATA}/brain_data/test"
MASK = f"{DATA}/brain_masks"

RUNS = [("基线单层(原 ckpt)", f"{DATA}/ckpt/checkpoint_epoch20.pt")]
for s in (0, 1, 2, 3, 4):
    p = f"{DATA}/ckpt_seed{s}/checkpoint_epoch20.pt"
    if os.path.exists(p):
        RUNS.append((f"基线单层(seed {s})", p))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = SliceAnomalyDataset(ROOT, mask_root=MASK, image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()
rois = np.stack([np.asarray(ds[i]["roi"], dtype=bool) for i in range(len(ds))])


def out_red(m):
    res = []
    for sel in (labels == 1, labels == 0):
        mm, rr = m[sel], rois[sel]
        thr = np.percentile(mm, 90, axis=(1, 2), keepdims=True)
        hot = mm >= thr
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
print(f"{'':20s} | {'imgAUROC':>8} {'Dice':>7} {'IoU':>7} {'pxAUROC':>8} | "
      f"{'脑外红斑(异常/正常)':>18} | {'红灯真病灶率':>12}")
print("-" * 104)

for name, ckpt in RUNS:
    if not os.path.exists(ckpt):
        print(f"!! 缺 {ckpt}，跳过 {name}")
        continue
    cfg = Config(device=str(device))
    cfg.ms_layers = []                       # 基线 = 单层，与 eval_final 的 RUNS[0] 一致
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
    print(f"{name:20s} | {im['auroc']:>8.4f} {pm['dice']:>7.4f} {pm['iou']:>7.4f} "
          f"{pm['pixel_auroc']:>8.4f} | {oa:>8.1f}% /{on:>7.1f}% | "
          f"{red_precision(mp):>11.1f}%")
    del model, ck
    torch.cuda.empty_cache()
