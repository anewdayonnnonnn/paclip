"""多尺度 patch 特征重训。

与产出 ckpt/ 的基线训练**只有一个变量之差**：cfg.ms_layers。
损失、超参、数据、轮数全不动（bottleneck=128 lambda_t=0.05 margin=0.3 20ep）。
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


def main(ms_layers, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(device=str(device), epochs=20, batch_size=8, lr=1e-4)
    cfg.ms_layers = list(ms_layers)

    tag = str(list(ms_layers)) if ms_layers else "单层(基线)"
    print(f"[ms] 图像侧 patch 特征: {tag}")

    model = TextSideAnomalyModel(cfg).to(device)
    anchors = DEFAULT_BRAIN_MRI_PROMPTS
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = L.TotalLoss(margin=cfg.margin, w_text=cfg.w_text,
                       w_global=cfg.w_global, w_local=cfg.w_local)

    ds = SliceAnomalyDataset("D:/brain_dl/brain_data/train", mask_root="D:/brain_dl/brain_masks",
                             image_size=cfg.image_size, grid=cfg.image_size // 16)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    print(f"[ms] train={len(ds)}  可训练参数={sum(p.numel() for p in trainable)}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        step = 0
        for step, b in enumerate(loader):
            enc = model.encode_anchors(anchors)
            out = model(b["image"].to(device), enc)
            d = crit(enc, out, b["label"].to(device), b["mask"].to(device))
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
    print("[ms] 完成", flush=True)


if __name__ == "__main__":
    ms = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] else []
    out = sys.argv[2] if len(sys.argv) > 2 else "D:/brain_dl/ckpt_ms"
    main(ms, out)
