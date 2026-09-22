"""异常检测通用指标。

图像级（分类）：AUROC、AP（平均精度）、F1、ACC。
像素级（定位）：Dice、IoU、像素 AUROC。
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """图像级异常分数 → 指标。

    Args:
        scores: (N,) 连续异常分数（越大越异常）。
        labels: (N,) 0/1 标签。

    Returns:
        {auroc, ap, f1, acc}
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "ap": float("nan"),
                "f1": float("nan"), "acc": float("nan")}

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    # 最优阈值下的 F1
    thresholds = np.unique(scores)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    acc = accuracy_score(labels, scores >= best_thr)
    return {"auroc": auroc, "ap": ap, "f1": best_f1, "acc": acc}


def pixel_metrics(maps: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
    """像素级异常图 → 定位指标。

    Args:
        maps: (N, H, W) 连续异常图（已上采样到掩码尺寸）。
        masks: (N, H, W) 0/1 病灶掩码。

    Returns:
        {dice, iou, pixel_auroc}
    """
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks)
    flat_m = maps.reshape(-1)
    flat_g = masks.reshape(-1)

    if len(np.unique(flat_g)) < 2:
        return {"dice": float("nan"), "iou": float("nan"), "pixel_auroc": float("nan")}

    pixel_auroc = roc_auc_score(flat_g, flat_m)

    # 最优阈值下的 Dice / IoU
    best_dice, best_iou, best_thr = 0.0, 0.0, 0.5
    for thr in np.unique(flat_m):
        pred = (flat_m >= thr)
        inter = (pred & (flat_g > 0)).sum()
        union_pred = pred.sum()
        union_gt = (flat_g > 0).sum()
        dice = 2 * inter / (union_pred + union_gt + 1e-6)
        iou = inter / (union_pred + union_gt - inter + 1e-6)
        if dice > best_dice:
            best_dice, best_iou, best_thr = dice, iou, thr
    return {"dice": best_dice, "iou": best_iou, "pixel_auroc": pixel_auroc}
