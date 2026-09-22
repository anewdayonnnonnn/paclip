"""预处理：把胸腺瘤 ROI 3D 体积的病灶切片一次性提取成 .npz，供训练/评估快速读取。

避免训练时每个 epoch 重复解压 466 个 .nii.gz（约 90s/epoch 的纯 I/O）。
每个体积取肿瘤面积最大的 top-8 病灶切片，存：
    image   (224,224) 纵隔窗灰度 [0,1]
    mask    (14,14)   病灶 mask（分块 max 下采样，训练局部对齐用）
    mask224 (224,224) 病灶 mask（出图画 GT 轮廓用）
"""

import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image

from text_side_anomaly.thymoma_dataset import (
    _downsample_mask,
    _window_slice,
    scan_roi_volumes,
)

ROOT = "data/本院图像和ROI_能共享_2025-11-18整理/全部图像及ROI"
OUT = "thymoma_slices"
TOPK = 8


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    vols = scan_roi_volumes(ROOT)
    n_slices = 0
    for img_path, mask_path in vols:
        vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        msk = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)
        sums = msk.sum(axis=(0, 1))
        idxs = np.argsort(sums)[::-1][:TOPK]
        idxs = idxs[sums[idxs] > 0]

        base = os.path.basename(img_path).replace("-origin.nii.gz", "")
        for z in idxs:
            s = _window_slice(vol[:, :, z])
            m = msk[:, :, z]
            gray = Image.fromarray((s * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            mask14 = _downsample_mask(m, 14)
            mask224 = (np.asarray(
                Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
                dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

            np.savez_compressed(
                os.path.join(OUT, f"{base}_z{z}.npz"),
                image=np.asarray(gray, dtype=np.float32) / 255.0,
                mask=mask14,
                mask224=mask224,
            )
            n_slices += 1
    print(f"提取切片数: {n_slices}，输出到 {OUT}/")


if __name__ == "__main__":
    main()
