"""最终对比 + 出图。

评测：基线单层 | 多尺度 | 多尺度+ROI门控 | ROI训练 | ROI训练+门控
指标：Dice / IoU / pixelAUROC（沿用全局最优阈值口径）+ 脑外红斑占比（用户抱怨的那个量）

出图：同一批切片，基线旧显示 vs 多尺度旧显示 vs ROI门控新显示 vs 原图+GT
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics, pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS
from text_side_anomaly.roi import gate_by_roi
from text_side_anomaly.visualize import AmapCalibration, colorize, draw_labels, render

ROOT = "D:/brain_dl/brain_data/test"
MASK = "D:/brain_dl/brain_masks"
OUT = "D:/brain_dl/heatmaps_final"
AMAP_SCALE = 0.3

RUNS = [
    ("基线单层", "D:/brain_dl/ckpt/checkpoint_epoch20.pt", [], False, False),
    ("多尺度", "D:/brain_dl/ckpt_ms/checkpoint_epoch20.pt", [5, 8, 11], False, False),
    ("多尺度+门控", "D:/brain_dl/ckpt_ms/checkpoint_epoch20.pt", [5, 8, 11], True, False),
    ("ROI训练", "D:/brain_dl/ckpt_roi/checkpoint_epoch20.pt", [5, 8, 11], False, True),
    ("ROI训练+门控", "D:/brain_dl/ckpt_roi/checkpoint_epoch20.pt", [5, 8, 11], True, True),
]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
device = torch.device("cuda")


def build(ckpt, ms):
    cfg = Config(device=str(device))
    cfg.ms_layers = list(ms)
    model = TextSideAnomalyModel(cfg).to(device)
    torch.serialization.add_safe_globals([Config])
    ck = torch.load(ckpt, map_location=device)
    model.text_adapter.load_state_dict(ck["adapter"])
    with torch.no_grad():
        model.fusion_weights.copy_(ck["fusion_weights"])
    model.eval()
    return model


ds = SliceAnomalyDataset(ROOT, mask_root=MASK, image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()
rois = np.stack([np.asarray(ds[i]["roi"], dtype=bool) for i in range(len(ds))])

results = {}
for name, ckpt, ms, gate, _ in RUNS:
    if not os.path.exists(ckpt):
        print(f"!! 缺 {ckpt}，跳过 {name}")
        continue
    model = build(ckpt, ms)
    sc, mp = [], []
    with torch.no_grad():
        anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
        for img, _, _ in batches:
            o = model(img, anchors)
            sc.append(o["cls_probs"].cpu())
            mp.append(o["anomaly_map"].cpu())
    sc = torch.cat(sc).numpy()
    mp = torch.cat(mp).numpy()
    if gate:
        mp = np.stack([gate_by_roi(mp[i], rois[i]) for i in range(len(mp))])
    results[name] = (sc, mp)

print(f"\ntest N={len(labels)}  异常={(labels == 1).sum()}  正常={(labels == 0).sum()}")
print(f"{'':16s} | {'imgAUROC':>8} {'Dice':>7} {'IoU':>7} {'pxAUROC':>8} | "
      f"{'脑外红斑(异常/正常)':>20} | {'红灯真病灶率':>12}")
print("-" * 100)


def out_red(m):
    """高分区落在 ROI 外的比例（异常切片 / 正常切片）。"""
    res = []
    for sel in (labels == 1, labels == 0):
        mm, rr = m[sel], rois[sel]
        thr = np.percentile(mm, 90, axis=(1, 2), keepdims=True)
        hot = mm >= thr
        res.append(hot[~rr].sum() / max(1, hot.sum()) * 100)
    return res


def red_precision(m, pct=90):
    """ROI 内被判红的像素中，真病灶占多少。固定操作点，不用 oracle 阈值。"""
    tp = fp = 0
    for i in range(len(m)):
        if masks[i].sum() == 0:
            continue
        r = rois[i] if rois[i].sum() >= 5 else np.ones_like(rois[i], bool)
        red = (m[i] >= np.percentile(m[i][r], pct)) & rois[i]
        tp += (red & (masks[i] > 0)).sum()
        fp += (red & (masks[i] <= 0)).sum()
    return tp / max(1, tp + fp) * 100


for name, (sc, mp) in results.items():
    im = image_metrics(sc, labels)
    pm = pixel_metrics(mp, masks)
    oa, on = out_red(mp)
    print(f"{name:16s} | {im['auroc']:>8.4f} {pm['dice']:>7.4f} {pm['iou']:>7.4f} "
          f"{pm['pixel_auroc']:>8.4f} | {oa:>9.1f}% /{on:>7.1f}% | {red_precision(mp):>11.1f}%")

# ---------------- 出图 ----------------
keys = [k for k in results]
if len(keys) < 2:
    sys.exit(0)
for k in keys:
    if k.endswith("门控"):
        break

best = keys[-1]
ms_key = "多尺度" if "多尺度" in results else keys[1]


def fit_in_roi(amaps, sel):
    """只用 ROI 内、且是正常切片的像素拟合标定。

    坑：门控把 ROI 外压到 floor，直接对整张门控图取分位数会让 P50 落在 floor 上，
    结果脑内全部像素被映射成高值 —— 整片脑烧成橙红。必须把 ROI 外排除掉再统计。
    """
    vals = np.concatenate([amaps[i][rois[i]] if rois[i].sum() >= 5 else amaps[i].ravel()
                           for i in np.where(sel)[0]])
    return AmapCalibration(float(np.percentile(vals, 50)), float(np.percentile(vals, 99.5)))


calib_g = fit_in_roi(results[best][1], labels == 0)
print(f"\n标定（正常切片 ROI 内）: lo={calib_g.lo:+.4f}  hi={calib_g.hi:+.4f}")

tumor_idx = [i for i in range(len(labels)) if labels[i] == 1]
les = np.array([masks[i].sum() for i in tumor_idx])
order = np.argsort(les)
picked = [tumor_idx[i] for i in order[len(order) // 3: len(order) // 3 + 4]]
nidx = [i for i in range(len(labels)) if labels[i] == 0]
picked += nidx[len(nidx) // 2: len(nidx) // 2 + 2]

os.makedirs(OUT, exist_ok=True)
for i in picked:
    gray = Image.open(ds.paths[i][0]).convert("L").resize((224, 224), Image.BILINEAR)
    g01 = np.asarray(gray, dtype=np.float32) / 255.0
    mp = ds.mask_paths[i]
    gt_bin = (np.asarray(Image.open(mp).convert("L").resize((224, 224), Image.NEAREST),
                         dtype=np.float32) > 127).astype(np.uint8) if mp else None

    def old_pipe(amap):
        o = np.clip((amap - np.percentile(amap, 25)) / AMAP_SCALE, 0, 1)
        up = np.asarray(Image.fromarray(o.astype(np.float32), mode="F").resize(
            (224, 224), Image.BILINEAR), dtype=np.float32)
        return colorize(np.clip(up, 0, 1), g01, gt_bin)

    panels = [old_pipe(results["基线单层"][1][i]), old_pipe(results[ms_key][1][i])]
    panels.append(render(results[best][1][i], g01, calib_g, gt_bin, guided=True)[0])

    # 注意：*255 必须放在 if 外面。正常切片没有 GT，若把它写进 if 里，
    # float 的 0~1 直接转 uint8 会全变成 0，整格显示成黑的。
    from PIL import ImageFilter
    g3 = (np.stack([g01] * 3, -1) * 255).astype(np.uint8)
    if gt_bin is not None:
        e = np.asarray(Image.fromarray((gt_bin * 255).astype(np.uint8)).filter(
            ImageFilter.FIND_EDGES), dtype=np.float32)
        g3[e > 30] = [0, 255, 0]
    panels.append(g3)

    labs = ["基线 旧显示", f"{ms_key} 旧显示", f"{best} 新显示", "原图+GT"]
    canvas = Image.new("RGB", ((224 + 8) * 4, 224 + 24), (255, 255, 255))
    for j, p in enumerate(panels):
        canvas.paste(Image.fromarray(p.astype(np.uint8)), (j * (224 + 8), 24))
    draw_labels(canvas, labs, panel_w=224)          # 用中文字体，默认字体会变方块
    tag = "tumor" if labels[i] == 1 else "normal"
    canvas.save(f"{OUT}/{tag}_{os.path.basename(ds.paths[i][0]).replace('.png', '')}.png")

print(f"\n出图 -> {OUT}/  面板顺序: {' | '.join(labs)}")
