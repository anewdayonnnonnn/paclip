"""对照实验：只把文本分离损失改成有界区间，其余完全不动。

原版:  L = relu(cos - m)                 无下界 → cos 一路压到 -0.99
改后:  L = relu(cos - m_hi) + relu(m_lo - cos)   把 cos 约束在 [m_lo, m_hi]

不修改 text_side_anomaly/losses.py，用 monkeypatch 替换。
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")

from text_side_anomaly import losses as L
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

M_HI, M_LO = 0.3, -0.1


def bounded_text_separation_loss(anchors, margin=0.3):
    """有界版：把 cos 约束在 [M_LO, M_HI] 区间内，避免坍缩到对跖点。"""
    losses = []
    for lvl in anchors:
        cos = F.cosine_similarity(anchors[lvl]["abnormal"], anchors[lvl]["normal"], dim=0)
        losses.append(F.relu(cos - M_HI) + F.relu(M_LO - cos))
    if not losses:
        return torch.zeros((), device="cpu")
    return torch.stack(losses).mean()


L.text_separation_loss = bounded_text_separation_loss   # TotalLoss.forward 按模块全局查找

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(device=str(device), epochs=20, batch_size=8, lr=1e-4)
    print(f"[fixed] 损失改为有界: cos 目标区间 [{M_LO}, {M_HI}]")

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = DEFAULT_BRAIN_MRI_PROMPTS
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = L.TotalLoss(margin=cfg.margin, w_text=cfg.w_text,
                       w_global=cfg.w_global, w_local=cfg.w_local)

    ds = SliceAnomalyDataset("D:/brain_dl/brain_data/train", mask_root="D:/brain_dl/brain_masks",
                             image_size=cfg.image_size, grid=cfg.image_size // 16)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    print(f"[fixed] train={len(ds)}  可训练参数={sum(p.numel() for p in trainable)}")

    os.makedirs("D:/brain_dl/ckpt_fixed", exist_ok=True)
    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for step, b in enumerate(loader):
            enc = model.encode_anchors(anchors)
            out = model(b["image"].to(device), enc)
            d = crit(enc, out, b["label"].to(device), b["mask"].to(device))
            opt.zero_grad()
            d["total"].backward()
            opt.step()
            tot += d["total"].item()
            if step % 40 == 0:
                with torch.no_grad():
                    cs = [F.cosine_similarity(enc[l]["abnormal"], enc[l]["normal"], dim=0).item()
                          for l in enc]
                print(f"[epoch {ep+1}/20] step {step}: total={d['total'].item():.4f} "
                      f"text={d['text'].item():.4f} cos={['%+.3f' % c for c in cs]}", flush=True)
        print(f"[epoch {ep+1}/20] avg loss = {tot / max(1, step+1):.4f}", flush=True)
        torch.save({"adapter": model.text_adapter.state_dict(),
                    "fusion_weights": model.fusion_weights, "cfg": cfg},
                   f"D:/brain_dl/ckpt_fixed/checkpoint_epoch{ep+1}.pt")
    print("[fixed] 完成")


if __name__ == "__main__":
    main()
