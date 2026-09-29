"""在留出的 brain MRI test 集上推理：图像级 + 像素级指标，并出热力图。

用 text_side_anomaly 原版模型/指标，只补一个 checkpoint 加载 + 出图流程。
热力图同时给两种标定：
  raw  = 原版 np.clip(amap/0.3)
  cal  = 减去背景基线后再用同一固定尺度
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics, pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

AMAP_SCALE = 0.3


def load_model(ckpt_path, device):
    cfg = Config(device=str(device))
    model = TextSideAnomalyModel(cfg).to(device)
    # checkpoint 里带 cfg(Config 对象)，torch>=2.6 默认 weights_only=True 会拒绝
    torch.serialization.add_safe_globals([Config])
    ck = torch.load(ckpt_path, map_location=device)
    model.text_adapter.load_state_dict(ck["adapter"])
    with torch.no_grad():
        model.fusion_weights.copy_(ck["fusion_weights"])
    model.eval()
    return model, cfg


def colorize(norm01, gray_pil, gt_pil=None, alpha=0.45):
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mask_root", default=None)
    ap.add_argument("--out_dir", default="D:/brain_dl/heatmaps_brain")
    ap.add_argument("--n_hm", type=int, default=6)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    anchors = DEFAULT_BRAIN_MRI_PROMPTS

    ds = SliceAnomalyDataset(args.data_root, mask_root=args.mask_root,
                             image_size=cfg.image_size, grid=cfg.image_size // 16)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    scores, labels, maps, masks = [], [], [], []
    for batch in loader:
        out = model(batch["image"].to(device), model.encode_anchors(anchors))
        scores.append(out["cls_probs"].cpu())
        labels.append(batch["label"])
        maps.append(out["anomaly_map"].cpu())
        masks.append(batch["mask"])

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    maps = torch.cat(maps).numpy()
    masks = torch.cat(masks).numpy()

    img = image_metrics(scores, labels)
    pix = pixel_metrics(maps, masks)

    print("\n===== brain MRI 留出 test 集 =====")
    print(f"  样本: {len(labels)}  (正常 {(labels==0).sum()} / 异常 {(labels==1).sum()})")
    print(f"  图像级  AUROC={img['auroc']:.4f}  AP={img['ap']:.4f}  "
          f"F1={img['f1']:.4f}  ACC={img['acc']:.4f}")
    print(f"  像素级  Dice={pix['dice']:.4f}  IoU={pix['iou']:.4f}  "
          f"pixelAUROC={pix['pixel_auroc']:.4f}")

    # 正常/异常切片上的 amap 分布（判断标定是否失配）
    pos, neg = maps[labels == 1], maps[labels == 0]
    print(f"\n  amap 分布  P99(异常图)={np.percentile(pos,99):+.4f}  "
          f"P99(正常图)={np.percentile(neg,99):+.4f}  "
          f"max(异常图)={pos.max():+.4f}")

    # ---- 出图：正常/异常各取几张 ----
    os.makedirs(args.out_dir, exist_ok=True)
    picked = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0][:args.n_hm]
        picked += [(int(i), cls) for i in idx]

    for i, cls in picked:
        path = ds.paths[i][0]
        gray = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
        x = np.asarray(gray, dtype=np.float32) / 255.0
        amap = maps[i]
        # 背景基线：用整图的 25 分位（不需要 GT，可部署）
        base = np.percentile(amap, 25)

        m = ds.mask_paths[i]
        gt = Image.open(m).convert("L").resize((224, 224), Image.NEAREST).point(
            lambda v: 255 if v > 127 else 0) if m else None

        raw = np.clip(amap / AMAP_SCALE, 0, 1)
        cal = np.clip((amap - base) / AMAP_SCALE, 0, 1)

        panel = [colorize(raw, gray, gt), colorize(cal, gray, gt)]
        W, H = panel[0].size
        canvas = Image.new("RGB", (W * 2 + 10, H), (255, 255, 255))
        for j, p in enumerate(panel):
            canvas.paste(p, (j * (W + 10), 0))
        name = os.path.basename(path).replace(".png", "")
        tag = "tumor" if cls == 1 else "normal"
        canvas.save(f"{args.out_dir}/{tag}_{name}_s{scores[i]:.2f}.png")

    print(f"\n  热力图 -> {args.out_dir}/  (左=原版固定尺度, 右=减背景基线)")


if __name__ == "__main__":
    main()
