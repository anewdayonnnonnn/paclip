"""标定探针：现有 ckpt 下，「全局阈值」vs「逐图标定」的 Dice 差多少。

背景：昨晚 pixelAUROC 已经 0.94（排序是好的），但 Dice 只有 0.55。
      怀疑瓶颈不在特征，而在"怎么把 amap 读成二值图"这一步。
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

CKPT = sys.argv[1] if len(sys.argv) > 1 else "D:/brain_dl/ckpt/checkpoint_epoch20.pt"
ROOT = "D:/brain_dl/brain_data/test"
MASK = "D:/brain_dl/brain_masks"


def per_image_best_dice(maps, masks):
    """每张图各自取最优阈值 —— 逐图标定的理论上界。"""
    out = []
    for m, g in zip(maps, masks):
        fg = g.reshape(-1) > 0
        if fg.sum() == 0:
            continue
        fm = m.reshape(-1)
        best = 0.0
        for thr in np.unique(fm):
            p = fm >= thr
            d = 2 * (p & fg).sum() / (p.sum() + fg.sum() + 1e-6)
            if d > best:
                best = d
        out.append(best)
    return float(np.mean(out)), len(out)


device = torch.device("cuda")
cfg = Config(device=str(device))
model = TextSideAnomalyModel(cfg).to(device)
torch.serialization.add_safe_globals([Config])
ck = torch.load(CKPT, map_location=device)
model.text_adapter.load_state_dict(ck["adapter"])
with torch.no_grad():
    model.fusion_weights.copy_(ck["fusion_weights"])
model.eval()

ds = SliceAnomalyDataset(ROOT, mask_root=MASK, image_size=cfg.image_size,
                         grid=cfg.image_size // 16)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

maps, masks, labels = [], [], []
with torch.no_grad():
    anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
    for b in loader:
        out = model(b["image"].to(device), anchors)
        maps.append(out["anomaly_map"].cpu())
        masks.append(b["mask"])
        labels.append(b["label"])

maps = torch.cat(maps).numpy()
masks = torch.cat(masks).numpy()
labels = torch.cat(labels).numpy()

print(f"N={len(labels)}  异常={(labels == 1).sum()}  正常={(labels == 0).sum()}")
print(f"amap 全局均值={maps.mean():+.4f}  全局标准差={maps.std():.4f}")
print(f"逐图均值范围 [{maps.mean(axis=(1, 2)).min():+.3f}, {maps.mean(axis=(1, 2)).max():+.3f}]")

print("\n--- Dice / IoU / pixelAUROC（沿用 metrics.py 的全局展平口径）---")
rows = [
    ("① 原版（全局最优阈值）", maps),
    ("② 逐图 z-score", (maps - maps.mean(axis=(1, 2), keepdims=True))
     / (maps.std(axis=(1, 2), keepdims=True) + 1e-8)),
    ("③ 逐图 min-max", (maps - maps.min(axis=(1, 2), keepdims=True))
     / (maps.max(axis=(1, 2), keepdims=True) - maps.min(axis=(1, 2), keepdims=True) + 1e-8)),
    ("④ 逐图减 P25 基线", maps - np.percentile(maps, 25, axis=(1, 2), keepdims=True)),
]
for name, m in rows:
    r = pixel_metrics(m, masks)
    print(f"{name:22s} Dice={r['dice']:.4f}  IoU={r['iou']:.4f}  pixelAUROC={r['pixel_auroc']:.4f}")

d, n = per_image_best_dice(maps, masks)
print(f"\n⑤ 逐图各自最优阈值（上界）  Dice={d:.4f}   有病灶图数={n}")
print("   ↑ 若 ⑤ 远高于 ①，说明瓶颈是标定而非特征。")
