"""我方方法（文本侧 Adapter + BiomedCLIP）在真实医学 AD 数据上的训练 + 评估。

当前用 PneumoniaMNIST(224，实际 28→224) 作真实、可复现的 normal/abnormal 基准；
brain MRI（BraTS/IXI）需凭证下载，拿到后只需替换 data + prompts。
"""

import sys

import torch
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.metrics import image_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import ThreeLevelPrompts

# 与数据匹配的三层提示（肺炎胸片）
PNEUMONIA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest x-ray"],
        "2": ["a normal chest x-ray with clear lungs"],
        "3": ["a normal chest x-ray with sharp lung markings"],
    },
    abnormal={
        "1": ["a chest x-ray with pneumonia"],
        "2": ["a chest x-ray with pulmonary opacities"],
        "3": ["a chest x-ray with hazy infiltrates"],
    },
)


@torch.no_grad()
def predict_loader(model, loader, anchors, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, model.encode_anchors(anchors))
        scores.append(out["cls_probs"].cpu())
        labels.append(batch["label"])
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(device=str(device), epochs=args.epochs, batch_size=args.batch_size,
                 bottleneck=args.bottleneck, lambda_t=args.lambda_t, margin=args.margin,
                 lr=args.lr)

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = PNEUMONIA_PROMPTS

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = TotalLoss(margin=cfg.margin, w_text=cfg.w_text,
                          w_global=cfg.w_global, w_local=0.0)  # 无掩码，仅全局+文本

    train_ds = SliceAnomalyDataset("data_pneumonia/train", mask_root=None)
    test_ds = SliceAnomalyDataset("data_pneumonia/test", mask_root=None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    print(f"[ours] 可训练参数={sum(p.numel() for p in trainable)} "
          f"train={len(train_ds)} test={len(test_ds)} device={device}")

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            enc = model.encode_anchors(anchors)
            out = model(images, enc)
            loss = criterion(enc, out, labels, None)
            optimizer.zero_grad()
            loss["total"].backward()
            optimizer.step()
            total += loss["total"].item()
        scores, trues = predict_loader(model, test_loader, anchors, device)
        m = image_metrics(scores, trues)
        print(f"[epoch {epoch+1}/{cfg.epochs}] loss={total/len(train_loader):.4f} "
              f"test AUROC={m['auroc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f}")

    scores, trues = predict_loader(model, test_loader, anchors, device)
    m = image_metrics(scores, trues)
    print("\n===== 我方方法（文本侧 Adapter + BiomedCLIP）=====")
    print("图像级: AUROC=%.4f  AP=%.4f  F1=%.4f  ACC=%.4f"
          % (m["auroc"], m["ap"], m["f1"], m["acc"]))
    print("融合权重:", [round(float(w), 3) for w in model.fusion_weights.softmax(dim=0)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bottleneck", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--lambda_t", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main()
