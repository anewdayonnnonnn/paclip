"""训练 / 推理 / 评估脚本（brain MRI 异常检测）。

用法：
    python -m text_side_anomaly.train --data_root data --mask_root masks --data_format slice

训练期间每一步重算文本锚点（Adapter 参数在更新，锚点随权重变化）。
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SliceAnomalyDataset, VolumeAnomalyDataset
from .losses import TotalLoss
from .metrics import image_metrics, pixel_metrics
from .model import TextSideAnomalyModel
from .prompts import DEFAULT_BRAIN_MRI_PROMPTS


def build_tokenizer(cfg: Config):
    import open_clip

    return open_clip.get_tokenizer(cfg.model_name)


def build_dataset(cfg: Config):
    kw = dict(data_root=cfg.data_root, mask_root=cfg.mask_root,
              image_size=cfg.image_size, grid=cfg.image_size // 16)
    if cfg.data_format == "volume":
        return VolumeAnomalyDataset(**kw, modality=cfg.modality,
                                    slice_strategy=cfg.slice_strategy, normalize=cfg.normalize)
    return SliceAnomalyDataset(**kw)


@torch.no_grad()
def predict_batch(model, loader, anchors, cfg, device):
    """推理：返回图像级异常分数、标签、异常图、掩码。"""
    model.eval()
    scores, labels, maps, masks = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        enc = model.encode_anchors(anchors)
        out = model(images, enc)
        scores.append(out["cls_probs"].cpu())
        labels.append(batch["label"])
        maps.append(out["anomaly_map"].cpu())
        masks.append(batch["mask"])
    return (
        torch.cat(scores).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(maps).numpy(),
        torch.cat(masks).numpy(),
    )


@torch.no_grad()
def evaluate(model, loader, anchors, cfg, device):
    scores, labels, maps, masks = predict_batch(model, loader, anchors, cfg, device)
    img = image_metrics(scores, labels)
    pix = pixel_metrics(maps, masks)
    return img, pix, scores, labels, maps, masks


def main(args):
    cfg = Config(
        data_root=args.data_root,
        mask_root=args.mask_root or None,
        data_format=args.data_format,
        bottleneck=args.bottleneck,
        lambda_t=args.lambda_t,
        margin=args.margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = DEFAULT_BRAIN_MRI_PROMPTS

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = TotalLoss(
        margin=cfg.margin, w_text=cfg.w_text, w_global=cfg.w_global, w_local=cfg.w_local
    )

    train_ds = build_dataset(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)

    print(f"[train] 可训练参数: {sum(p.numel() for p in trainable)}")
    print(f"[train] 样本数: {len(train_ds)} device={device} "
          f"bottleneck={cfg.bottleneck} lambda_t={cfg.lambda_t} margin={cfg.margin}")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device) if cfg.mask_root else None

            enc = model.encode_anchors(anchors)   # 重算文本锚点
            outputs = model(images, enc)
            loss_dict = criterion(enc, outputs, labels, masks)

            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            if step % 20 == 0:
                print(
                    f"[epoch {epoch+1}/{cfg.epochs}] step {step}: "
                    f"total={loss_dict['total'].item():.4f} "
                    f"text={loss_dict['text'].item():.4f} "
                    f"global={loss_dict['global'].item():.4f} "
                    + (f" local={loss_dict['local'].item():.4f}" if cfg.mask_root else "")
                )

        print(f"[epoch {epoch+1}/{cfg.epochs}] avg loss = {total_loss / max(1, step+1):.4f}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {"adapter": model.text_adapter.state_dict(),
                 "fusion_weights": model.fusion_weights, "cfg": cfg},
                os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pt"),
            )

    # ---- 训练集上评估（异常检测指标）----
    img, pix, *_ = evaluate(model, train_loader, anchors, cfg, device)
    print("[eval] 图像级:", {k: round(v, 4) for k, v in img.items()})
    if cfg.mask_root:
        print("[eval] 像素级:", {k: round(v, 4) for k, v in pix.items()})
    print("[train] 完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--data_format", type=str, default="slice", choices=["slice", "volume"])
    parser.add_argument("--bottleneck", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--lambda_t", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default=None)
    main(parser.parse_args())
