"""基线 vs 多尺度：三项指标 + 并排热力图。

验收口径（与昨晚数字可直接对比）：
  pixelAUROC  排序质量，不受阈值影响
  Dice/IoU    沿用 metrics.py 的全局展平 + 全局最优阈值口径
  热力图      同一切片并排，肉眼看「发灰」是否改善
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics, pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

AMAP_SCALE = 0.3
ROOT = "D:/brain_dl/brain_data/test"
MASK = "D:/brain_dl/brain_masks"
OUT = "D:/brain_dl/heatmaps_cmp"

RUNS = [
    ("基线(单层)", "D:/brain_dl/ckpt/checkpoint_epoch20.pt", []),
    ("多尺度5,8,11", "D:/brain_dl/ckpt_ms/checkpoint_epoch20.pt", [5, 8, 11]),
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


def colorize(norm01, gray_pil, gt_pil=None, alpha=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    h, w = gray_pil.size[1], gray_pil.size[0]
    up = Image.fromarray((np.clip(norm01, 0, 1) * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR)
    norm = np.asarray(up, dtype=np.float32) / 255.0
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)
    gray = np.asarray(gray_pil.convert("RGB"), dtype=np.float32)
    blend = ((1 - alpha) * gray + alpha * heat).astype(np.uint8)
    if gt_pil is not None:
        e = np.asarray(gt_pil.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        blend[e > 30] = [0, 255, 0]
    return Image.fromarray(blend)


ds = SliceAnomalyDataset(ROOT, mask_root=MASK, image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()

results = {}
for name, ckpt, ms in RUNS:
    if not os.path.exists(ckpt):
        print(f"!! 找不到 {ckpt}，跳过 {name}")
        continue
    model = build(ckpt, ms)
    scores, maps = [], []
    with torch.no_grad():
        anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
        for img, _, _ in batches:
            out = model(img, anchors)
            scores.append(out["cls_probs"].cpu())
            maps.append(out["anomaly_map"].cpu())
    scores = torch.cat(scores).numpy()
    maps = torch.cat(maps).numpy()
    results[name] = (scores, maps)

print(f"\ntest N={len(labels)}  异常={(labels == 1).sum()}  正常={(labels == 0).sum()}")
print(f"{'':>14} | {'imgAUROC':>8} {'imgAP':>7} {'Dice':>7} {'IoU':>7} {'pxAUROC':>8} | "
      f"{'amap均值':>9} {'病灶内':>8} {'病灶外':>8}")
print("-" * 100)
for name, (scores, maps) in results.items():
    im = image_metrics(scores, labels)
    pm = pixel_metrics(maps, masks)
    m1 = maps[labels == 1].mean()
    m0 = maps[labels == 0].mean()
    print(f"{name:>14} | {im['auroc']:>8.4f} {im['ap']:>7.4f} {pm['dice']:>7.4f} "
          f"{pm['iou']:>7.4f} {pm['pixel_auroc']:>8.4f} | {maps.mean():>+9.4f} "
          f"{m1:>+8.4f} {m0:>+8.4f}")

# ---- 并排热力图 ----
if len(results) < 2:
    sys.exit(0)

names = list(results.keys())
os.makedirs(OUT, exist_ok=True)
# 挑异常图里图像级分数最高、且病灶格数中等的切片，最能看出定位差别
tumor_idx = [i for i in range(len(labels)) if labels[i] == 1]
les = np.array([masks[i].sum() for i in tumor_idx])
order = np.argsort(les)
picked = [tumor_idx[i] for i in order[len(order) // 3: len(order) // 3 + 6]]

for i in picked:
    path = ds.paths[i][0]
    gray = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
    mp = ds.mask_paths[i]
    gt = (Image.open(mp).convert("L").resize((224, 224), Image.NEAREST)
          .point(lambda v: 255 if v > 127 else 0)) if mp else None

    panels = []
    for name in names:
        amap = results[name][1][i]
        base = np.percentile(amap, 25)
        panels.append(colorize(np.clip((amap - base) / AMAP_SCALE, 0, 1), gray, gt))
    # 最右再放一张纯 GT
    gray_gt = gray.convert("RGB").copy()
    if gt is not None:
        e = np.asarray(gt.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        arr = np.asarray(gray_gt).copy()
        arr[e > 30] = [0, 255, 0]
        gray_gt = Image.fromarray(arr)
    panels.append(gray_gt)

    W, H = panels[0].size
    canvas = Image.new("RGB", ((W + 8) * len(panels), H + 18), (255, 255, 255))
    for j, p in enumerate(panels):
        canvas.paste(p, (j * (W + 8), 18))
    name = os.path.basename(path).replace(".png", "")
    canvas.save(f"{OUT}/cmp_{name}.png")

print(f"\n并排热力图 -> {OUT}/   顺序: {' | '.join(names)} | 原图+GT绿边")
