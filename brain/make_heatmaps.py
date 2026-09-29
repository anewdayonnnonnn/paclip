"""显示管线对比：旧标定(双线性 + 固定 0.3) vs 新标定(引导滤波 + 分位数)。

四个面板：基线/旧 | 多尺度/旧 | 多尺度/新 | 原图+GT绿边
标定只用正常切片拟合（部署时不需要病灶标注）。
特意混入正常切片，验证「正常脑不会凭空发亮」。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS
from text_side_anomaly.visualize import AmapCalibration, colorize, draw_labels, render

AMAP_SCALE = 0.3          # 旧显示管线的固定绝对尺度
ROOT = "D:/brain_dl/brain_data/test"
MASK = "D:/brain_dl/brain_masks"
OUT = "D:/brain_dl/heatmaps_display"

RUNS = [
    ("基线/旧", "D:/brain_dl/ckpt/checkpoint_epoch20.pt", []),
    ("多尺度/旧", "D:/brain_dl/ckpt_ms/checkpoint_epoch20.pt", [5, 8, 11]),
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

maps_by_run = {}
for name, ckpt, ms in RUNS:
    model = build(ckpt, ms)
    out_maps = []
    with torch.no_grad():
        anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
        for img, _, _ in batches:
            out_maps.append(model(img, anchors)["anomaly_map"].cpu())
    maps_by_run[name] = torch.cat(out_maps).numpy()

# ---- 只用正常切片拟合标定（部署时不需要病灶标注）----
ms_maps = maps_by_run["多尺度/旧"]
calib = AmapCalibration.fit_from_normals(ms_maps[labels == 0], lo_pct=50, hi_pct=99.5)
print(f"标定（正常切片拟合）: lo={calib.lo:+.4f}  hi={calib.hi:+.4f}  "
      f"旧管线固定尺度 P25={np.percentile(ms_maps[labels==0], 25):+.4f}")

# 旧管线在该模型上的实际动态范围，用来解释为什么它发灰
normals = ms_maps[labels == 0]
print(f"正常切片 amap 分位  P50={np.percentile(normals,50):+.4f}  "
      f"P99.5={np.percentile(normals,99.5):+.4f}")
print(f"旧管线(P25基线/0.3)下正常切片的色阶占用 = "
      f"{np.clip((np.percentile(normals,99.5)-np.percentile(normals,25))/AMAP_SCALE,0,1):.3f}")
print(f"新管线(分位数标定)下正常切片的色阶占用 = "
      f"{np.clip((np.percentile(normals,99.5)-calib.lo)/(calib.hi-calib.lo),0,1):.3f}")

# ---- 挑切片：4 张病灶大小中等的异常 + 2 张正常 ----
tumor_idx = [i for i in range(len(labels)) if labels[i] == 1]
les = np.array([masks[i].sum() for i in tumor_idx])
order = np.argsort(les)
picked = [tumor_idx[i] for i in order[len(order) // 3: len(order) // 3 + 4]]
normal_idx = [i for i in range(len(labels)) if labels[i] == 0]
picked += normal_idx[len(normal_idx) // 2: len(normal_idx) // 2 + 2]

os.makedirs(OUT, exist_ok=True)
for i in picked:
    path = ds.paths[i][0]
    gray = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
    g01 = np.asarray(gray, dtype=np.float32) / 255.0
    mp = ds.mask_paths[i]
    gt = (np.asarray(Image.open(mp).convert("L").resize((224, 224), Image.NEAREST),
                     dtype=np.float32) if mp else None)
    gt_bin = (gt > 127).astype(np.uint8) if gt is not None else None

    panels = []
    # 前两个：基线/多尺度 走旧管线（P25 基线 + 固定 0.3 + 双线性）
    for name in ["基线/旧", "多尺度/旧"]:
        amap = maps_by_run[name][i]
        old = np.clip((amap - np.percentile(amap, 25)) / AMAP_SCALE, 0, 1)
        from PIL import Image as _I
        up = np.asarray(_I.fromarray(old.astype(np.float32), mode="F").resize(
            (224, 224), _I.BILINEAR), dtype=np.float32)
        panels.append(colorize(np.clip(up, 0, 1), g01, gt_bin))
    # 第三个：多尺度 走新管线（分位数标定 + 引导滤波）
    panels.append(render(ms_maps[i], g01, calib, gt_bin, guided=True)[0])
    # 第四个：原图 + GT。*255 必须在 if 外，否则正常切片（无 GT）会整格变黑。
    from PIL import ImageFilter
    g3 = (np.stack([g01] * 3, -1) * 255).astype(np.uint8)
    if gt_bin is not None:
        e = np.asarray(Image.fromarray((gt_bin * 255).astype(np.uint8)).filter(
            ImageFilter.FIND_EDGES), dtype=np.float32)
        g3[e > 30] = [0, 255, 0]
    panels.append(g3)

    W, H = 224, 224
    labs = ["基线 旧显示", "多尺度 旧显示", "多尺度 新显示", "原图+GT"]
    canvas = Image.new("RGB", ((W + 8) * 4, H + 24), (255, 255, 255))
    for j, p in enumerate(panels):
        canvas.paste(Image.fromarray(p.astype(np.uint8)), (j * (W + 8), 24))
    draw_labels(canvas, labs, panel_w=W)          # 中文字体，默认字体会变方块
    tag = "tumor" if labels[i] == 1 else "normal"
    name = os.path.basename(path).replace(".png", "")
    canvas.save(f"{OUT}/{tag}_{name}.png")

print(f"\n出图 -> {OUT}/   共 {len(picked)} 张")
