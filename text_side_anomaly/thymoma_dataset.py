"""胸腺瘤 ROI 数据加载：3D CT 体积 + 手动勾画的肿瘤 ROI mask。

数据目录结构（本院 233 例胸腺瘤，全异常，均有 ROI）：
    root/<范围>/
        XXX000-origin.nii.gz   平扫图像
        XXX000.nii.gz          平扫 ROI（0/1）
        XXX020-origin.nii.gz   增强图像
        XXX020.nii.gz          增强 ROI（0/1）
    编号：前 3 位为患者编号，后 3 位 000=平扫、020=增强。

无正常（无病灶）病例 → 只用于训练局部对齐分支：
病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点（见 losses.local_alignment_loss）。
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import _gray2clip

_ORIGIN_SUFFIX = "-origin.nii.gz"


def _window_slice(s: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """CT HU 纵隔窗归一化到 [0,1]：lo=wl-ww/2，hi=wl+ww/2。

    胸腺瘤位于前纵隔，放射科标准纵隔窗 WW=400 / WL=40（文档建议 WW 350-400 / WL 40-50）。
    """
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _downsample_mask(m: np.ndarray, grid: int) -> np.ndarray:
    """二值 mask (H,W) → (grid,grid)，每 cell 取区域内最大值（保小病灶）。

    BILINEAR+阈值 会丢失面积小于一个 cell 的小肿瘤，改用分块 max 保真。
    """
    H, W = m.shape
    out = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        h0, h1 = int(i * H / grid), int((i + 1) * H / grid)
        for j in range(grid):
            w0, w1 = int(j * W / grid), int((j + 1) * W / grid)
            blk = m[h0:h1, w0:w1]
            out[i, j] = float(blk.max()) if blk.size else 0.0
    return out


def scan_roi_volumes(root: str) -> List[Tuple[str, str]]:
    """递归扫描 root 下所有 `*-origin.nii.gz`，配对同名（去 -origin）ROI mask。

    Returns:
        [(img_path, mask_path), ...]，按路径排序。
    """
    vols: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(_ORIGIN_SUFFIX):
                base = f[: -len(_ORIGIN_SUFFIX)]  # "003000"
                mask = os.path.join(dirpath, base + ".nii.gz")
                if os.path.isfile(mask):
                    vols.append((os.path.join(dirpath, f), mask))
    return sorted(vols)


def make_splits(
    root: str,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """按**患者编号**（文件名前 3 位）切 train/val/test，避免同病例切片泄漏。

    Returns:
        {"train": [(img, mask), ...], "val": [...], "test": [...]}
    """
    vols = scan_roi_volumes(root)
    cases = sorted({os.path.basename(img)[:3] for img, _ in vols})
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_cases = set(cases[:n_train])
    val_cases = set(cases[n_train : n_train + n_val])
    test_cases = set(cases[n_train + n_val :])

    splits: dict = {}
    for name, case_set in [
        ("train", train_cases),
        ("val", val_cases),
        ("test", test_cases),
    ]:
        splits[name] = [
            (img, m) for img, m in vols if os.path.basename(img)[:3] in case_set
        ]
    return splits


class ThymomaROIDataset(Dataset):
    """胸腺瘤 3D CT 体积 + ROI mask，按病灶切片采样（全异常）。

    每次取一个含病灶的轴向切片（优先取肿瘤面积较大的层），
    纵隔窗归一化 + 缩放 224×224；mask 下采样到 grid×grid（14×14）。
    """

    def __init__(
        self,
        root: str,
        volumes: Optional[List[Tuple[str, str]]] = None,
        image_size: int = 224,
        grid: int = 14,
        window: Tuple[float, float] = (40.0, 400.0),
        topk: int = 8,
    ):
        self.image_size = image_size
        self.grid = grid
        self.wl, self.ww = window
        self.topk = topk
        self.volumes = volumes if volumes is not None else scan_roi_volumes(root)

    def __len__(self) -> int:
        return len(self.volumes)

    def _pick_lesion_slice(self, msk: np.ndarray) -> int:
        """沿 z 找含病灶切片，从肿瘤面积最大的 topk 层里随机取一层。"""
        sums = msk.sum(axis=(0, 1))  # (D,)
        idxs = np.argsort(sums)[::-1][: self.topk]
        idxs = idxs[sums[idxs] > 0]
        if len(idxs) == 0:
            return msk.shape[2] // 2
        return int(np.random.choice(idxs))

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        img_path, mask_path = self.volumes[idx]
        vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)  # (H, W, D)
        msk = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)  # (H, W, D) 0/1

        z = self._pick_lesion_slice(msk)
        s = _window_slice(vol[:, :, z], self.wl, self.ww)
        m = msk[:, :, z]

        gray = Image.fromarray((s * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        image = _gray2clip(np.asarray(gray, dtype=np.float32) / 255.0)

        mask = torch.from_numpy(_downsample_mask(m, self.grid))

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(1, dtype=torch.long),
            "path": img_path,
        }


class ThymomaSliceDataset(Dataset):
    """从预提取的 .npz 病灶切片读取（快速训练/评估）。"""

    def __init__(self, npz_dir: str, files: Optional[List[str]] = None):
        self.files = (
            files
            if files is not None
            else sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        d = np.load(self.files[idx])
        x = d["image"].astype(np.float32)  # (224,224) [0,1]
        m = d["mask"].astype(np.float32)   # (14,14)
        return {
            "image": _gray2clip(x),
            "mask": torch.from_numpy(m),
            "label": torch.tensor(1, dtype=torch.long),
            "path": self.files[idx],
        }


def make_slice_splits(
    npz_dir: str,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """按病例号（文件名前 3 位）切分预提取切片，与 make_splits 同 seed 同病例集合。"""
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    cases = sorted({os.path.basename(f)[:3] for f in files})
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    sets = {
        "train": set(cases[:n_train]),
        "val": set(cases[n_train : n_train + n_val]),
        "test": set(cases[n_train + n_val :]),
    }
    return {
        name: [f for f in files if os.path.basename(f)[:3] in cset]
        for name, cset in sets.items()
    }
