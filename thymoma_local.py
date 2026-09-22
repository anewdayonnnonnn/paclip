"""用胸腺瘤 ROI 数据训练定位分支（局部对齐），量化验证热力图定位。

无正常（无病灶）病例 → 不做全局二分类（w_global=0），只训练：
    - 文本分离损失（normal/abnormal 锚点拉开）
    - 局部对齐损失（病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点）

评估用 patch 级 AUROC / Dice / IoU（metrics.pixel_metrics），并出热力图叠加 GT 肿瘤轮廓。
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import _gray2clip
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.metrics import pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import ThreeLevelPrompts
from text_side_anomaly.thymoma_dataset import (
    ThymomaSliceDataset,
    make_slice_splits,
)

ROOT = "data/本院图像和ROI_能共享_2025-11-18整理/全部图像及ROI"
AMAP_SCALE = 0.3  # 局部异常图 (pa-pn) 的固定尺度，同 generate_heatmaps.py

# 胸腺瘤 CT 三级提示（前纵隔肿块）
THYMOMA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": ["a normal chest CT with clear mediastinum"],
        "3": ["a normal chest CT with no mediastinal mass"],
    },
    abnormal={
        "1": ["a chest CT with an anterior mediastinal mass"],
        "2": ["a chest CT with a thymoma"],
        "3": ["a chest CT with a mediastinal tumor"],
    },
)


def pre_tokenize(model, prompts, device):
    """预 tokenize 三层提示词（tokenize 结果固定，避免每 step 重复 tokenize）。"""
    return {
        lvl: {
            k: model.tokenizer(texts).to(device)
            for k, texts in [("normal", prompts.normal[lvl]), ("abnormal", prompts.abnormal[lvl])]
        }
        for lvl in prompts.levels
    }


def encode_anchors_cached(model, tok):
    """用预 tokenize 的 tokens 编码三层锚点（每 step 仍重算 adapter 前向）。"""
    anchors = {}
    for lvl, d in tok.items():
        t_n = model.encode_text_tokens(d["normal"])
        t_a = model.encode_text_tokens(d["abnormal"])
        anchors[lvl] = {
            "normal": F.normalize(t_n.mean(0) if t_n.size(0) > 1 else t_n.squeeze(0), dim=0),
            "abnormal": F.normalize(t_a.mean(0) if t_a.size(0) > 1 else t_a.squeeze(0), dim=0),
        }
    return anchors


@torch.no_grad()
def evaluate(model, anchors, loader, device):
    """测试集 patch 级定位指标（14×14）。"""
    model.eval()
    enc = model.encode_anchors(anchors)
    maps, masks = [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, enc)
        maps.append(out["anomaly_map"].cpu())
        masks.append(batch["mask"])
    maps = torch.cat(maps).numpy()
    masks = torch.cat(masks).numpy()
    return pixel_metrics(maps, masks)


def overlay_heatmap_with_gt(gray_pil, amap, score, gt_mask_pil, out_path):
    """热力图叠加到窗化灰度 CT，并把 GT 肿瘤轮廓画成绿色。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    h, w = gray_pil.size[1], gray_pil.size[0]

    local = np.clip(amap / AMAP_SCALE, 0.0, 1.0)
    up = Image.fromarray((local * 255).astype(np.uint8)).convert("L")
    up = up.resize((w, h), Image.BILINEAR)
    norm = np.asarray(up, dtype=np.float32) / 255.0
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)

    # 胸腺瘤定位场景全局分支未训练（w_global=0），cls_probs 无意义，
    # 故不用 score 门控，直接用局部异常图的固定尺度出图。
    alpha = 0.45
    gray = np.asarray(gray_pil.convert("RGB"), dtype=np.float32)
    blend = ((1.0 - alpha) * gray + alpha * heat).astype(np.uint8)

    # GT 轮廓（绿色）
    edge = np.asarray(gt_mask_pil.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    blend[edge > 30] = [0, 255, 0]
    Image.fromarray(blend).save(out_path)


@torch.no_grad()
def save_heatmaps(model, anchors, files, device, out_dir, n=6):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    enc = model.encode_anchors(anchors)

    # 跨病例采样：每个病例取一个切片，避免全采到同一病例
    picked, seen = [], set()
    for f in files:
        case = os.path.basename(f)[:3]
        if case not in seen:
            picked.append(f)
            seen.add(case)
        if len(picked) >= n:
            break

    for f in picked:
        d = np.load(f)
        x = d["image"].astype(np.float32)       # (224,224) [0,1]
        mask224 = d["mask224"].astype(np.float32)

        gray = Image.fromarray((x * 255).astype(np.uint8))
        img = _gray2clip(x).unsqueeze(0).to(device)
        out = model(img, enc)
        amap = out["anomaly_map"][0].cpu().numpy()
        score = out["cls_probs"][0].item()
        gt = Image.fromarray((mask224 * 255).astype(np.uint8))

        fn = os.path.basename(f).replace(".npz", "")
        overlay_heatmap_with_gt(gray, amap, score, gt, f"{out_dir}/{fn}_score{score:.2f}.png")
        print(f"  {fn} amap[min/mean/max]={amap.min():+.3f}/{amap.mean():+.3f}/{amap.max():+.3f}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="加载 checkpoint，只评估+出图")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(device=str(device), epochs=args.epochs, batch_size=16, lr=1e-4)

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = THYMOMA_PROMPTS
    tok = pre_tokenize(model, anchors, device)
    ckpt = "thymoma_local.pt"

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"已加载 checkpoint -> {ckpt}")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # 无正常样本：只训练文本分离 + 局部对齐
        crit = TotalLoss(margin=cfg.margin, w_text=1.0, w_global=0.0, w_local=1.0)

        splits = make_slice_splits("thymoma_slices", seed=0)
        train_ds = ThymomaSliceDataset("thymoma_slices", files=splits["train"])
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        print(f"[thymoma-local] train_slices={len(train_ds)} "
              f"device={device} 可训练参数={sum(p.numel() for p in trainable)}")

        for epoch in range(cfg.epochs):
            model.train()
            total = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                labels = batch["label"].to(device)
                enc = encode_anchors_cached(model, tok)
                out = model(images, enc)
                loss = crit(enc, out, labels, masks)
                opt.zero_grad()
                loss["total"].backward()
                opt.step()
                total += loss["total"].item()
            print(f"[epoch {epoch + 1}/{cfg.epochs}] loss={total / len(train_loader):.4f}")

        torch.save(model.state_dict(), ckpt)
        print(f"已保存 checkpoint -> {ckpt}")

    splits = make_slice_splits("thymoma_slices", seed=0)
    test_ds = ThymomaSliceDataset("thymoma_slices", files=splits["test"])
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    m = evaluate(model, anchors, test_loader, device)
    print("\n===== 定位评估（test，patch 级 14×14）=====")
    print(f"  pixel AUROC={m['pixel_auroc']:.4f}  Dice={m['dice']:.4f}  IoU={m['iou']:.4f}")

    save_heatmaps(model, anchors, splits["test"], device, "heatmaps_thymoma", n=6)
    print("热力图已生成到 heatmaps_thymoma/（绿色=GT 肿瘤轮廓，红色=预测热区）")


if __name__ == "__main__":
    main()
