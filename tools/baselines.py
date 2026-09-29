"""AD SOTA 基线：PaDiM 与 PatchCore（one-class，自实现，避免 anomalib 的 Lightning 开销）。

用 timm 预训练骨干（经 hf-mirror 下载），在 PneumoniaMNIST 上：
    - 训练只用 normal 图像；
    - 测试 normal + abnormal，输出图像级异常分数，算 AUROC。
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader

from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.metrics import image_metrics


# --------------------------------------------------------------------------- #
# PaDiM
# --------------------------------------------------------------------------- #
class PaDiM:
    def __init__(self, device, d_reduce=100):
        self.device = device
        self.d_reduce = d_reduce
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, features_only=True, out_indices=(1, 2, 3)
        ).eval().to(device)
        self.proj = torch.randn(448, d_reduce, device=device) * (448 ** -0.5)
        self.means = None
        self.inv_cov = None

    @torch.no_grad()
    def _extract(self, x):  # x:(B,3,224,224) -> (B, H*W, d)
        feats = self.backbone(x)
        target = feats[0].shape[-2:]
        resized = [F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(resized, dim=1)                 # (B,448,56,56)
        emb = emb.permute(0, 2, 3, 1)                   # (B,56,56,448)
        B, H, W, C = emb.shape
        emb = (emb.reshape(B, H * W, C)) @ self.proj    # (B, H*W, d)
        return emb

    @torch.no_grad()
    def fit(self, loader):
        embs = []
        for batch in loader:
            x = batch["image"].to(self.device)
            embs.append(self._extract(x))
        embs = torch.cat(embs, dim=0)                   # (N, L, d)
        N, L, d = embs.shape
        embs = embs.permute(1, 0, 2)                    # (L, N, d)
        self.means = embs.mean(dim=1)                   # (L, d)
        c = embs - self.means.unsqueeze(1)
        cov = torch.einsum("lnd,lnk->ldk", c, c) / max(1, N - 1)
        cov = cov + 0.01 * torch.eye(d, device=self.device)
        self.inv_cov = torch.linalg.inv(cov)            # (L, d, d)

    @torch.no_grad()
    def score(self, x):  # x:(B,3,224,224) -> (B,)
        emb = self._extract(x)                          # (B, L, d)
        diff = emb - self.means.unsqueeze(0)            # (B, L, d)
        m = torch.einsum("bld,lde,ble->bl", diff, self.inv_cov, diff)  # (B, L)
        return m.amax(dim=1)                            # (B,)


# --------------------------------------------------------------------------- #
# PatchCore（记忆库随机子采样 + kNN）
# --------------------------------------------------------------------------- #
class PatchCore:
    def __init__(self, device, n_bank=10000):
        self.device = device
        self.n_bank = n_bank
        self.backbone = timm.create_model(
            "wide_resnet50_2", pretrained=True, features_only=True, out_indices=(2, 3)
        ).eval().to(device)
        self.memory = None

    @torch.no_grad()
    def _extract(self, x):  # x:(B,3,224,224) -> (B, L, 3072)
        f2, f3 = self.backbone(x)                       # (B,1024,28,28),(B,2048,14,14)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        emb = torch.cat([f2, f3], dim=1)                # (B,3072,28,28)
        emb = emb.permute(0, 2, 3, 1).reshape(emb.shape[0], -1, emb.shape[1])
        return emb                                      # (B, 784, 3072)

    @torch.no_grad()
    def fit(self, loader):
        banks = []
        for batch in loader:
            x = batch["image"].to(self.device)
            e = self._extract(x)
            banks.append(e.reshape(-1, e.shape[-1]))
        bank = torch.cat(banks, dim=0)                  # (M, dim)
        # 随机子采样记忆库
        if bank.shape[0] > self.n_bank:
            idx = torch.randperm(bank.shape[0])[: self.n_bank]
            bank = bank[idx]
        self.memory = F.normalize(bank, dim=1)

    @torch.no_grad()
    def score(self, x):  # x:(B,3,224,224) -> (B,)
        emb = self._extract(x)                          # (B, 784, 3072)
        emb = F.normalize(emb, dim=-1)
        B, L, _ = emb.shape
        out = []
        for i in range(B):
            # 每个 patch 到记忆库最近邻距离
            sim = emb[i] @ self.memory.t()              # (L, M)
            dist = 1.0 - sim.amax(dim=1)                # (L,)
            out.append(dist.amax())
        return torch.stack(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SliceAnomalyDataset("data_pneumonia/train", mask_root=None)
    # one-class：只取 normal
    train_normal = [s for s in train_ds.samples] if hasattr(train_ds, "samples") else None
    from torch.utils.data import Subset

    normal_idx = [i for i, (p, l) in enumerate(train_ds.paths) if l == 0]
    train_loader = DataLoader(Subset(train_ds, normal_idx), batch_size=64, shuffle=True, num_workers=0)

    test_ds = SliceAnomalyDataset("data_pneumonia/test", mask_root=None)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    results = {}
    for name in ["PaDiM", "PatchCore"]:
        print(f"\n===== {name} =====")
        model = PaDiM(device) if name == "PaDiM" else PatchCore(device)
        print("  拟合正常特征...")
        model.fit(train_loader)

        scores, trues = [], []
        for batch in test_loader:
            x = batch["image"].to(device)
            s = model.score(x)
            scores.append(s.cpu())
            trues.append(batch["label"])
        scores = torch.cat(scores).numpy()
        trues = torch.cat(trues).numpy()
        m = image_metrics(scores, trues)
        results[name] = m["auroc"]
        print(f"  {name} AUROC={m['auroc']:.4f}  AP={m['ap']:.4f}  F1={m['f1']:.4f}")

    print("\n===== 汇总 =====")
    for k, v in results.items():
        print(f"  {k}: AUROC={v:.4f}")

    import json
    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存到 baseline_results.json")


if __name__ == "__main__":
    main()
