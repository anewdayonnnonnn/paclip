"""离线自测：用合成 brain-MRI 数据 + 模拟冻结骨干验证完整训练/推理/评估流程。

本环境无法下载 BiomedCLIP 权重与真实 brain MRI 数据，故用：
    - 合成数据：正常脑组织（平滑纹理） vs 异常（平滑纹理 + 高亮病灶团块 + 掩码）；
    - 模拟冻结骨干：固定随机文本词表 + 基于局部亮度的固定视觉特征，
      使得"病灶 patch 更亮"在特征空间中可区分，从而能真正训练文本 Adapter。

跑通后可得到真实的异常检测指标（图像级 AUROC/AP/F1、像素级 Dice/IoU/AUROC）。
在真机上把 transformers.CLIPModel 换成真 BiomedCLIP、数据换成真 brain MRI 即可。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 让脚本可被直接运行（当前目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transformers  # noqa: E402
from text_side_anomaly.config import Config  # noqa: E402
from text_side_anomaly.losses import TotalLoss  # noqa: E402
from text_side_anomaly.metrics import image_metrics, pixel_metrics  # noqa: E402
from text_side_anomaly.model import TextSideAnomalyModel  # noqa: E402
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS  # noqa: E402


# --------------------------------------------------------------------------- #
# 1) 合成 brain-MRI 数据
# --------------------------------------------------------------------------- #
def _smooth_field(rng, size, k=21):
    """低通滤波随机场，模拟脑组织平滑纹理。"""
    x = rng.normal(0, 1, size).astype(np.float32)
    # 用平均池近似高斯滤波
    x = torch.tensor(x)[None, None]
    x = F.avg_pool2d(x, k, stride=1, padding=k // 2)
    return x[0, 0].numpy()


def make_synthetic_sample(rng, abnormal):
    """返回 (image 3x224x224, label, mask 14x14)。"""
    H = W = 224
    # 平滑脑组织背景
    tissue = _smooth_field(rng, (H, W), 25)
    tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min() + 1e-6) * 0.35 + 0.05

    # 脑形椭圆区域（略亮于背景）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, rx, ry = H / 2, W / 2, 90, 70
    brain = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1.0
    img = tissue.copy()
    img[brain] = img[brain] * 0.7 + 0.12

    mask14 = np.zeros((14, 14), dtype=np.float32)
    if abnormal:
        # 高亮病灶团块（随机位置，位于脑内）
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, 0.5)
        lx = cx + r * rx * np.cos(angle)
        ly = cy + r * ry * np.sin(angle)
        rad = rng.uniform(14, 22)
        d2 = (xx - lx) ** 2 + (yy - ly) ** 2
        lesion = d2 <= rad ** 2
        img[lesion] = float(rng.uniform(0.75, 1.0))
        # 下采样到 14x14 网格
        m = torch.tensor(lesion[None, None].astype(np.float32))
        m = F.avg_pool2d(m, 16)
        mask14 = (m[0, 0].numpy() > 0.25).astype(np.float32)

    # 灰度 → 3 通道 + CLIP 归一化
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = np.stack([img, img, img], axis=0)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return torch.from_numpy(x).float(), int(abnormal), torch.from_numpy(mask14)


class SyntheticBrainMRIDataset(Dataset):
    def __init__(self, n_normal, n_abnormal, seed=0):
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_normal):
            self.samples.append(make_synthetic_sample(rng, False))
        for _ in range(n_abnormal):
            self.samples.append(make_synthetic_sample(rng, True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, mask = self.samples[idx]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long), "mask": mask}


# --------------------------------------------------------------------------- #
# 2) 模拟冻结骨干（替换 transformers.CLIPModel）
# --------------------------------------------------------------------------- #
class MockTextModel(nn.Module):
    """固定随机词表：不同 prompt → 不同的（冻结）文本特征。"""

    def __init__(self, vocab=512, d=768):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.emb.weight.requires_grad = False
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class MockVisionModel(nn.Module):
    """固定局部亮度特征：patch 特征 ∝ 该 patch 平均亮度，病灶(高亮)可被区分。"""

    def __init__(self, d=768):
        super().__init__()
        self.register_buffer("W", torch.randn(1, d) * 0.1)

    def forward(self, pixel_values=None):
        B = pixel_values.shape[0]
        x = pixel_values.mean(dim=1, keepdim=True)          # (B,1,224,224) 亮度
        x = F.avg_pool2d(x, 16)                             # (B,1,14,14)
        x = x.reshape(B, 196, 1)                            # (B,196,1)
        patch = x * self.W                                 # (B,196,768)
        cls = patch.mean(dim=1, keepdim=True)              # (B,1,768)
        feats = torch.cat([cls, patch], dim=1)             # (B,197,768)
        return SimpleNamespace(last_hidden_state=feats)


class MockCLIP(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=projection_dim)
        self.text_model = MockTextModel()
        self.vision_model = MockVisionModel()
        self.text_projection = nn.Linear(768, projection_dim)
        self.visual_projection = nn.Linear(768, projection_dim)
        for p in self.text_projection.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False


class MockTokenizer:
    def __call__(self, texts, padding=None, truncation=None, max_length=None, return_tensors=None):
        max_length = max_length or 64
        n = len(texts)
        ids = torch.zeros(n, max_length, dtype=torch.long)
        mask = torch.zeros(n, max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            L = min(len(t), max_length)
            for j, ch in enumerate(t[:L]):
                ids[i, j] = (ord(ch) % 500) + 1
            mask[i, :L] = 1
        return {"input_ids": ids, "attention_mask": mask}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    transformers.CLIPModel.from_pretrained = staticmethod(lambda *a, **k: MockCLIP())

    cfg = Config(device="cpu", bottleneck=128, lambda_t=0.05, margin=0.3, epochs=10, batch_size=32)
    device = torch.device("cpu")

    model = TextSideAnomalyModel(cfg).to(device)
    tokenizer = MockTokenizer()
    anchors = DEFAULT_BRAIN_MRI_PROMPTS

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = TotalLoss(margin=cfg.margin, w_text=cfg.w_text,
                          w_global=cfg.w_global, w_local=cfg.w_local)

    train_ds = SyntheticBrainMRIDataset(n_normal=200, n_abnormal=200, seed=0)
    val_ds = SyntheticBrainMRIDataset(n_normal=100, n_abnormal=100, seed=1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    print(f"[demo] 可训练参数: {sum(p.numel() for p in trainable)}")
    print(f"[demo] train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(cfg.epochs):
        model.train()
        total = text_sum = global_sum = local_sum = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            enc = model.encode_anchors(anchors, tokenizer)
            out = model(images, enc)
            loss = criterion(enc, out, labels, masks)
            optimizer.zero_grad()
            loss["total"].backward()
            optimizer.step()
            total += loss["total"].item()
            text_sum += loss["text"].item()
            global_sum += loss["global"].item()
            local_sum += loss["local"].item()
        n = len(train_loader)
        print(f"[epoch {epoch+1}/{cfg.epochs}] total={total/n:.4f} "
              f"text={text_sum/n:.4f} global={global_sum/n:.4f} local={local_sum/n:.4f}")

    # ---- 推理 + 评估 ----
    model.eval()
    scores, trues, maps, masks = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            enc = model.encode_anchors(anchors, tokenizer)
            out = model(images, enc)
            scores.append(out["cls_probs"].cpu())
            trues.append(batch["label"])
            maps.append(out["anomaly_map"].cpu())
            masks.append(batch["mask"])

    scores = torch.cat(scores).numpy()
    trues = torch.cat(trues).numpy()
    maps = torch.cat(maps).numpy()
    masks = torch.cat(masks).numpy()

    img = image_metrics(scores, trues)
    pix = pixel_metrics(maps, masks)

    print("\n===== 异常检测指标（合成数据 + 模拟骨干） =====")
    print("图像级：AUROC=%.4f  AP=%.4f  F1=%.4f  ACC=%.4f"
          % (img["auroc"], img["ap"], img["f1"], img["acc"]))
    print("像素级：Dice=%.4f  IoU=%.4f  pixelAUROC=%.4f"
          % (pix["dice"], pix["iou"], pix["pixel_auroc"]))
    print("\n说明：以上是离线自测（合成数据+模拟骨干）验证流程可用；")
    print("真机跑：python -m text_side_anomaly.train --data_root <brain_mri> --mask_root <masks>")


if __name__ == "__main__":
    main()
