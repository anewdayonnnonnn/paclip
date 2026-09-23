"""brain MRI 数据加载：2D 切片 或 3D 体积（.nii.gz）+ 病灶掩码。

2D 切片结构：
    data_root/
        normal/      *.png|*.jpg   正常切片
        abnormal/    *.png|*.jpg   异常切片
    mask_root/（可选）
        abnormal/    *.png         与 abnormal 同名的病灶掩码（>0 为病灶）

3D 体积结构（如 BraTS）：
    data_root/
        normal/      *.nii.gz
        abnormal/    *.nii.gz
    mask_root/
        abnormal/    *.nii.gz      与 abnormal 同名的病灶掩码

训练时从体积中抽取 2D 轴向切片，缩放到 224×224、灰度转 3 通道、强度归一化。
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .roi import roi_from_gray

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_NII_EXT = {".nii", ".nii.gz"}


def _list_files(root: str, exts) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if any(f.lower().endswith(e) for e in exts)
    )


def _gray2clip(x: np.ndarray) -> torch.Tensor:
    """(H, W) → (3, H, W) CLIP 归一化张量。"""
    x = np.stack([x, x, x], axis=0).astype(np.float32)
    x = (x - np.array(_MEAN, dtype=np.float32).reshape(3, 1, 1)) / np.array(
        _STD, dtype=np.float32
    ).reshape(3, 1, 1)
    return torch.from_numpy(x)


def _normalize_slice(s: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(s, low), np.percentile(s, high)
    if hi - lo < 1e-6:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class SliceAnomalyDataset(Dataset):
    """2D brain MRI 切片数据集。"""

    def __init__(
        self,
        data_root: str,
        mask_root: Optional[str] = None,
        image_size: int = 224,
        grid: int = 14,
    ):
        self.image_size = image_size
        self.grid = grid
        self.paths: List[Tuple[str, int]] = []
        self.paths += [(p, 0) for p in _list_files(os.path.join(data_root, "normal"), _IMG_EXT)]
        self.paths += [(p, 1) for p in _list_files(os.path.join(data_root, "abnormal"), _IMG_EXT)]

        self.mask_paths: List[Optional[str]] = [None] * len(self.paths)
        if mask_root:
            mask_by_name = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in _list_files(os.path.join(mask_root, "abnormal"), _IMG_EXT)
            }
            for i, (p, label) in enumerate(self.paths):
                if label == 1:
                    self.mask_paths[i] = mask_by_name.get(os.path.splitext(os.path.basename(p))[0])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.paths[idx]
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        x = np.asarray(img, dtype=np.float32) / 255.0
        image = _gray2clip(x)

        mask = torch.zeros(self.grid, self.grid)
        if self.mask_paths[idx] is not None:
            m = Image.open(self.mask_paths[idx]).convert("L").resize(
                (self.grid, self.grid), Image.BILINEAR
            )
            mask = torch.from_numpy((np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32))

        return {"image": image, "label": torch.tensor(label, dtype=torch.long),
                "mask": mask, "roi": self._roi(x), "path": path}

    def _roi(self, x: np.ndarray) -> torch.Tensor:
        """由图像强度给出解剖 ROI（(grid, grid) 布尔），见 roi.py。"""
        return torch.from_numpy(roi_from_gray(x, grid=self.grid))


class VolumeAnomalyDataset(Dataset):
    """3D brain MRI 体积（.nii.gz）数据集，按轴向切片训练。

    Args:
        modality: 4D 体积时选用的通道索引；None 表示 3D 单模态。
        slice_strategy: "lesion"（异常体积优先采含病灶切片）/ "middle" / "random"。
    """

    def __init__(
        self,
        data_root: str,
        mask_root: Optional[str] = None,
        image_size: int = 224,
        grid: int = 14,
        modality: Optional[int] = None,
        slice_strategy: str = "lesion",
        normalize: bool = True,
    ):
        self.image_size = image_size
        self.grid = grid
        self.modality = modality
        self.slice_strategy = slice_strategy
        self.normalize = normalize

        self.paths: List[Tuple[str, int]] = []
        self.paths += [(p, 0) for p in _list_files(os.path.join(data_root, "normal"), _NII_EXT)]
        self.paths += [(p, 1) for p in _list_files(os.path.join(data_root, "abnormal"), _NII_EXT)]

        self.mask_paths: List[Optional[str]] = [None] * len(self.paths)
        if mask_root:
            mask_by_name = {
                os.path.basename(p).replace(".nii.gz", "").replace(".nii", ""): p
                for p in _list_files(os.path.join(mask_root, "abnormal"), _NII_EXT)
            }
            for i, (p, label) in enumerate(self.paths):
                if label == 1:
                    name = os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
                    self.mask_paths[i] = mask_by_name.get(name)

    def __len__(self) -> int:
        return len(self.paths)

    def _pick_slice(self, depth: int, mask_slice) -> int:
        if self.slice_strategy == "middle":
            return depth // 2
        if self.slice_strategy == "lesion" and mask_slice is not None:
            sums = mask_slice.reshape(depth, -1).sum(axis=1)
            idxs = np.nonzero(sums > 0)[0]
            if len(idxs):
                return int(np.random.choice(idxs))
        return int(np.random.randint(0, depth))

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        path, label = self.paths[idx]
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # (H, W, D[, C])
        depth = vol.shape[2]

        mask_slice = None
        if self.mask_paths[idx] is not None:
            mv = np.asarray(nib.load(self.mask_paths[idx]).dataobj, dtype=np.float32)
            mask_slice = (mv > 0).astype(np.float32)               # (H, W, D)

        z = self._pick_slice(depth, mask_slice)

        if vol.ndim == 4:
            m = self.modality if self.modality is not None else 0
            s = vol[:, :, z, m]
        else:
            s = vol[:, :, z]

        s = _normalize_slice(s) if self.normalize else s
        s = Image.fromarray((s * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        image = _gray2clip(np.asarray(s, dtype=np.float32) / 255.0)

        mask = torch.zeros(self.grid, self.grid)
        if mask_slice is not None:
            m = Image.fromarray((mask_slice[:, :, z] * 255).astype(np.uint8)).resize(
                (self.grid, self.grid), Image.BILINEAR
            )
            mask = torch.from_numpy((np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32))

        roi = roi_from_gray(np.asarray(s, dtype=np.float32) / 255.0, grid=self.grid)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long),
                "mask": mask, "roi": torch.from_numpy(roi), "path": path}
