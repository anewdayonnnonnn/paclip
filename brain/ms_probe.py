"""多尺度 patch 特征探针（免重训）。

图像塔是冻结的，所以可以直接替换 patch 特征来源、复用现有 ckpt 看信号。
ms_layers=[11] 应与原版单层结果一致 —— 用来校验实现正确性。
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics, pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

CKPT = "D:/brain_dl/ckpt/checkpoint_epoch20.pt"
ROOT = "D:/brain_dl/brain_data/test"
MASK = "D:/brain_dl/brain_masks"

CONFIGS = [(), (11,), (5, 8, 11), (8, 11), (3, 5, 8, 11)]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

device = torch.device("cuda")
ds = SliceAnomalyDataset(ROOT, mask_root=MASK, image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()


def build(ms):
    cfg = Config(device=str(device))
    cfg.ms_layers = list(ms)
    model = TextSideAnomalyModel(cfg).to(device)
    torch.serialization.add_safe_globals([Config])
    ck = torch.load(CKPT, map_location=device)
    model.text_adapter.load_state_dict(ck["adapter"])
    with torch.no_grad():
        model.fusion_weights.copy_(ck["fusion_weights"])
    model.eval()
    return model


print(f"test N={len(labels)}  异常={(labels == 1).sum()}  正常={(labels == 0).sum()}")
print(f"{'ms_layers':>14} | {'imgAUROC':>8} {'Dice':>7} {'IoU':>7} {'pxAUROC':>8} | "
      f"{'amap均值':>9} {'amapstd':>7} {'跨patch标准差':>12}")
print("-" * 92)

ref_patch = None
for ms in CONFIGS:
    model = build(ms)
    scores, maps, patches = [], [], []
    with torch.no_grad():
        anchors = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS)
        for img, _, _ in batches:
            out = model(img, anchors)
            scores.append(out["cls_probs"].cpu())
            maps.append(out["anomaly_map"].cpu())
    scores = torch.cat(scores).numpy()
    maps = torch.cat(maps).numpy()

    im = image_metrics(scores, labels)
    pm = pixel_metrics(maps, masks)

    # 跨 patch 区分度：随机取样若干张图算 patch 特征标准差
    with torch.no_grad():
        sub = torch.stack([b[0][0] for b in batches[:64]])
        _, f_patch, _ = model.encode_image(sub)
        pstd = f_patch.std(dim=1).mean().item()
        pcos = (f_patch @ f_patch.transpose(1, 2)).mean().item()

    tag = "原版单层" if not ms else str(list(ms))
    print(f"{tag:>14} | {im['auroc']:>8.4f} {pm['dice']:>7.4f} {pm['iou']:>7.4f} "
          f"{pm['pixel_auroc']:>8.4f} | {maps.mean():>+9.4f} {maps.std():>7.4f} {pstd:>12.5f}")

    if ms == (11,):
        print("   [校验] 上面这行应与「原版单层」几乎一致；否则说明实现有问题")
