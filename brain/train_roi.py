"""ROI 感知训练：L_local 只在脑组织内监督。

与 ckpt_ms/（多尺度、无 ROI）只差一个变量：L_local 是否被 ROI 掩码。
背景 patch 上的监督对定位没有意义，还会稀释正常锚点（让它去代表大片空气）。
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/新建文件夹")

from text_side_anomaly import losses as L
from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main(out_dir, use_roi=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(device=str(device), epochs=20, batch_size=8, lr=1e-4)
    cfg.ms_layers = [5, 8, 11]

    print(f"[roi] 多尺度 [5,8,11] + L_local ROI 掩码 = {use_roi}")

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = DEFAULT_BRAIN_MRI_PROMPTS
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = L.TotalLoss(margin=cfg.margin, w_text=cfg.w_text,
                       w_global=cfg.w_global, w_local=cfg.w_local)

    ds = SliceAnomalyDataset("D:/brain_dl/brain_data/train", mask_root="D:/brain_dl/brain_masks",
                             image_size=cfg.image_size, grid=cfg.image_size // 16)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    print(f"[roi] train={len(ds)}  可训练参数={sum(p.numel() for p in trainable)}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for ep in range(cfg.epochs):
        model.train()
        tot, step = 0.0, 0
        for step, b in enumerate(loader):
            enc = model.encode_anchors(anchors)
            out = model(b["image"].to(device), enc)
            d = crit(enc, out, b["label"].to(device), b["mask"].to(device),
                     b["roi"].to(device) if use_roi else None)
            opt.zero_grad()
            d["total"].backward()
            opt.step()
            tot += d["total"].item()
            if step % 40 == 0:
                print(f"[epoch {ep+1}/20] step {step}: total={d['total'].item():.4f} "
                      f"text={d['text'].item():.4f} global={d['global'].item():.4f} "
                      f"local={d['local'].item():.4f}", flush=True)
        print(f"[epoch {ep+1}/20] avg loss = {tot / max(1, step+1):.4f}", flush=True)
        torch.save({"adapter": model.text_adapter.state_dict(),
                    "fusion_weights": model.fusion_weights, "cfg": cfg},
                   f"{out_dir}/checkpoint_epoch{ep+1}.pt")
    print("[roi] 完成", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "D:/brain_dl/ckpt_roi",
         use_roi=(sys.argv[2] != "0" if len(sys.argv) > 2 else True))
