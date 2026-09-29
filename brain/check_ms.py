"""验证：ckpt_seed* 到底是用单层还是多尺度训出来的。

同一个 checkpoint，分别按 ms_layers=[] 和 [5,8,11] 评估。训练与评估口径一致的那边
会明显更好 —— 这就是「训练用了多尺度、我却用单层评」这个错配的判别式。
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ds = SliceAnomalyDataset(f"{DATA}/brain_data/test", mask_root=f"{DATA}/brain_masks",
                         image_size=224, grid=14)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
batches = [(b["image"].to(device), b["label"], b["mask"]) for b in loader]
labels = torch.cat([b[1] for b in batches]).numpy()
masks = torch.cat([b[2] for b in batches]).numpy()


def run(ckpt, ms):
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
    im = image_metrics(torch.cat(sc).numpy(), labels)
    pm = pixel_metrics(torch.cat(mp).numpy(), masks)
    del model, ck
    torch.cuda.empty_cache()
    return im["auroc"], pm["dice"], pm["iou"], pm["pixel_auroc"]


CASES = [
    ("原 ckpt/（旧记录 基线单层 0.5535）", f"{DATA}/ckpt/checkpoint_epoch20.pt"),
    ("重训 ckpt_seed0", f"{DATA}/ckpt_seed0/checkpoint_epoch20.pt"),
    ("原 ckpt_ms/（旧记录 多尺度 0.6039）", f"{DATA}/ckpt_ms/checkpoint_epoch20.pt"),
]

print(f"\n{'模型':36s} | {'评估口径':>10} | {'imgAUROC':>8} {'Dice':>7} {'IoU':>7} {'pxAUROC':>8}")
print("-" * 96)
for name, ckpt in CASES:
    if not os.path.exists(ckpt):
        print(f"!! 缺 {ckpt}")
        continue
    for tag, ms in (("单层 []", []), ("多尺度[5,8,11]", [5, 8, 11])):
        a, d, i, p = run(ckpt, ms)
        print(f"{name:36s} | {tag:>10} | {a:>8.4f} {d:>7.4f} {i:>7.4f} {p:>8.4f}")
