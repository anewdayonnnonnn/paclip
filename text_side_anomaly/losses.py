"""损失函数（brain MRI 异常检测）。

对应文档"文本侧损失函数"（m 为允许的最大相似度）：

    L_text = Σ_l  max(0, cos(t_a^l, t_n^l) − m)      # normal/abnormal 锚点分离

全局对齐（CLS 图像级判断）：
    L_global = CE( cls_logits, y )                   # normal vs abnormal

局部对齐（病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点）：
    L_local  = CE( patch_logits, mask )              # 逐像素
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def text_separation_loss(
    anchors: Dict[str, Dict[str, torch.Tensor]],
    margin: float = 0.3,
) -> torch.Tensor:
    """文本侧分离损失：惩罚每层 normal/abnormal 锚点相似度超过 margin m。"""
    losses = []
    for lvl in anchors:
        cos = F.cosine_similarity(anchors[lvl]["abnormal"], anchors[lvl]["normal"], dim=0)
        losses.append(F.relu(cos - margin))
    if not losses:
        return torch.zeros((), device="cpu")
    return torch.stack(losses).mean()


def global_alignment_loss(cls_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """全局对齐：CLS 对 normal/abnormal 的交叉熵。

    Args:
        cls_logits: (B, 2) 图像级 logits（[normal, abnormal]）。
        labels: (B,) 图像级标签，0=正常，1=异常。
    """
    return F.cross_entropy(cls_logits, labels.long())


def local_alignment_loss(
    patch_logits: torch.Tensor,
    masks: torch.Tensor,
    roi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """局部对齐：patch 对 normal/abnormal 的逐像素交叉熵。

    Args:
        patch_logits: (B, 2, h, w) patch 级 logits。
        masks: (B, h, w) 病灶掩码（0/1，已下采样到 patch 网格）。
        roi: (B, h, w) 解剖 ROI（布尔）。给了就只在 ROI 内算损失——背景 patch 上的
            监督对定位没有意义，还会稀释正常锚点（让它去代表大片空气）。
    """
    if roi is None:
        return F.cross_entropy(patch_logits, masks.long())

    b, c, h, w = patch_logits.shape
    logits = patch_logits.permute(0, 2, 3, 1).reshape(-1, c)   # (B*h*w, 2)
    target = masks.reshape(-1).long()
    sel = roi.reshape(-1) > 0
    if sel.sum() == 0:
        return F.cross_entropy(patch_logits, masks.long())
    return F.cross_entropy(logits[sel], target[sel])


class TotalLoss(nn.Module):
    """总损失 = w_text·文本分离 + w_global·全局对齐 + w_local·局部对齐。"""

    def __init__(
        self,
        margin: float = 0.3,
        w_text: float = 1.0,
        w_global: float = 1.0,
        w_local: float = 1.0,
    ):
        super().__init__()
        self.margin = margin
        self.w_text = w_text
        self.w_global = w_global
        self.w_local = w_local

    def forward(
        self,
        anchors: Dict[str, Dict[str, torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        roi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_text = text_separation_loss(anchors, self.margin)
        loss_global = global_alignment_loss(outputs["cls_logits"], labels)

        loss_dict = {"text": loss_text, "global": loss_global}

        if masks is not None:
            loss_local = local_alignment_loss(outputs["patch_logits"], masks, roi)
            loss_dict["local"] = loss_local
        else:
            loss_local = torch.zeros((), device=loss_text.device)

        total = (
            self.w_text * loss_text
            + self.w_global * loss_global
            + self.w_local * loss_local
        )
        loss_dict["total"] = total
        return loss_dict
