"""生成异常检测热力图：把 patch 级异常图叠加到原图上。

训练我方方法（轻量 Adapter），然后对测试集正常/异常样本各取几张，
输出 224×224 的异常热力图（jet 叠加）。
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import ThreeLevelPrompts

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


def train_model(device, epochs=6, batch_size=64):
    cfg = Config(device=str(device), epochs=epochs, batch_size=batch_size, lr=1e-3)
    model = TextSideAnomalyModel(cfg).to(device)
    anchors = PNEUMONIA_PROMPTS
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = TotalLoss(margin=cfg.margin, w_text=cfg.w_text, w_global=cfg.w_global, w_local=0.0)

    ds = SliceAnomalyDataset("data_pneumonia/train", mask_root=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            enc = model.encode_anchors(anchors)
            out = model(images, enc)
            loss = crit(enc, out, labels, None)
            opt.zero_grad()
            loss["total"].backward()
            opt.step()
    return model, anchors


@torch.no_grad()
def anomaly_map_for(model, anchors, image_tensor, device):
    model.eval()
    enc = model.encode_anchors(anchors)
    out = model(image_tensor.unsqueeze(0).to(device), enc)
    m = out["anomaly_map"][0].cpu().numpy()          # (14,14)
    score = out["cls_probs"][0].item()
    return m, score


# 局部异常图 (pa - pn) 的固定尺度：|amap| 达到 AMAP_SCALE 即视为满热，<=0 全冷。
# 不要用逐图 min-max，否则正常图也会被拉出一块最红区域。
AMAP_SCALE = 0.3


def overlay_heatmap(gray_pil: Image.Image, amap: np.ndarray, score: float, out_path: str):
    """把 (14,14) 异常图用全局固定尺度映射到 [0,1]，jet 叠加到灰度图上。

    amap 是 (pa - pn) 余弦相似度差，>0 表示 patch 更靠近异常锚点。
    按图像级分数 score 做全局门控：正常图(score 低)整体保持冷色。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    h, w = gray_pil.size[1], gray_pil.size[0]

    # 固定尺度归一化（非逐图 min-max）
    local = np.clip(amap / AMAP_SCALE, 0.0, 1.0)
    up = Image.fromarray((local * 255).astype(np.uint8)).convert("L")
    up = up.resize((w, h), Image.BILINEAR)
    norm = np.asarray(up, dtype=np.float32) / 255.0
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)   # RGB 热力图

    # 全局门控：score<0.5 时热力逐渐熄灭，正常图保持冷色
    gate = float(np.clip((score - 0.5) * 2.0, 0.0, 1.0))
    alpha = 0.45 * gate

    gray = np.asarray(gray_pil.convert("RGB"), dtype=np.float32)
    blend = (1.0 - alpha) * gray + alpha * heat
    Image.fromarray(blend.astype(np.uint8)).save(out_path)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练中（约 2 分钟）...")
    model, anchors = train_model(device)

    os.makedirs("heatmaps", exist_ok=True)
    for cls, label in [("normal", 0), ("abnormal", 1)]:
        d = f"data_pneumonia/test/{cls}"
        files = sorted(os.listdir(d))[:3]
        for fn in files:
            path = os.path.join(d, fn)
            gray = Image.open(path).convert("L").resize((224, 224))
            img = np.asarray(gray, dtype=np.float32) / 255.0
            img = np.stack([img] * 3, axis=0)
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            x = (img - mean[:, None, None]) / std[:, None, None]
            amap, score = anomaly_map_for(model, anchors, torch.from_numpy(x).float(), device)
            out = f"heatmaps/{cls}_{fn}_score{score:.2f}.png"
            overlay_heatmap(gray, amap, score, out)
            print(f"  {cls} {fn} 异常分数={score:.3f} "
                  f"amap[min/mean/max]={amap.min():+.3f}/{amap.mean():+.3f}/{amap.max():+.3f} -> {out}")

    print("热力图已生成到 heatmaps/ 目录")


if __name__ == "__main__":
    main()
