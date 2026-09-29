"""损失函数（brain MRI 异常检测）。

对应文档"文本侧损失函数"（m 为允许的最大相似度）：

    L_text = Σ_l  max(0, cos(t_a^l, t_n^l) − m)      # normal/abnormal 锚点分离

全局对齐（CLS 图像级判断）：
    L_global = CE( cls_logits, y )                   # normal vs abnormal

局部对齐（病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点）：
    L_local  = CE( patch_logits, mask )              # 逐像素

2026-09-28 新增（见 docx 改动 7 的机制定位）：
    L_text 加下界 m_lo，阻止 cos(t_a,t_n) 被推到 −1（对跖点）；
    L_div  惩罚三层差向量 d_l = t_a^l − t_n^l 的方向对齐。
    原因：anomaly_map_l = f_patch·d_l，融合是 softmax 凸组合，故三个 d_l 平行 ⟺
    融合图恒等于单层图 ⟺ 三层机制形同虚设。原版单边 hinge 会把 cos 推到 −1，
    于是 d_l ≈ 2·t_a^l，d_l 方向被锁成 t_a 的方向，跨层 t_a 相似度直接决定三层 map
    相同（实测训练后 cos(d)=0.96、三层 map 相关 0.994）。两项新损失默认关闭，
    以保持与旧实验的可比性。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def text_separation_loss(
    anchors: Dict[str, Dict[str, torch.Tensor]],
    margin: float = 0.3,
    margin_lo: Optional[float] = None,
) -> torch.Tensor:
    """文本侧分离损失：惩罚每层 normal/abnormal 锚点相似度超出 [margin_lo, margin]。

    margin_lo=None 时退化为文档原版的**单边** hinge relu(cos − margin)：只罚"太像"、
    不罚"太不像"，无下界，优化器会把 cos 一路推到 −1。这是三层塌成一层的第一环
    （机制见 docx 改动 7 第 (3) 条）。给定下界即拆掉这条坍缩路径。
    """
    losses = []
    for lvl in anchors:
        cos = F.cosine_similarity(anchors[lvl]["abnormal"], anchors[lvl]["normal"], dim=0)
        upper = F.relu(cos - margin)
        if margin_lo is None:
            losses.append(upper)
        else:
            losses.append(upper + F.relu(margin_lo - cos))
    if not losses:
        return torch.zeros((), device="cpu")
    return torch.stack(losses).mean()


def d_diversity_loss(
    anchors: Dict[str, Dict[str, torch.Tensor]],
    margin_d: float = 0.7,
) -> torch.Tensor:
    """三层差向量 d_l = t_a^l − t_n^l 的方向多样性。

    anomaly_map 只由 d_l 决定，而融合是 softmax 凸组合，所以三个 d_l 平行 ⟺
    融合图恒等于任一单层图 ⟺ 融合权重成哑参数。**没有任何一项损失要求三层分歧**，
    不显式要求它就不会发生——实测训练后 cos(d) 均值 0.96、三层 map 相关 0.994。
    """
    ds = []
    for lvl in anchors:
        d = anchors[lvl]["abnormal"] - anchors[lvl]["normal"]
        ds.append(F.normalize(d, dim=0, eps=1e-6))
    if len(ds) < 2:
        return torch.zeros((), device=ds[0].device)
    losses = [
        F.relu(F.cosine_similarity(ds[i], ds[j], dim=0) - margin_d)
        for i in range(len(ds))
        for j in range(i + 1, len(ds))
    ]
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
    """总损失 = w_text·文本分离 + w_global·全局对齐 + w_local·局部对齐 + w_div·多层多样性。

    margin_lo / w_div 默认 None / 0.0，即旧行为，保证历史实验可比。
    """

    def __init__(
        self,
        margin: float = 0.3,
        w_text: float = 1.0,
        w_global: float = 1.0,
        w_local: float = 1.0,
        margin_lo: Optional[float] = None,
        margin_d: float = 0.7,
        w_div: float = 0.0,
    ):
        super().__init__()
        self.margin = margin
        self.w_text = w_text
        self.w_global = w_global
        self.w_local = w_local
        self.margin_lo = margin_lo
        self.margin_d = margin_d
        self.w_div = w_div

    def forward(
        self,
        anchors: Dict[str, Dict[str, torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        roi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_text = text_separation_loss(anchors, self.margin, self.margin_lo)
        loss_global = global_alignment_loss(outputs["cls_logits"], labels)

        loss_dict = {"text": loss_text, "global": loss_global}

        if masks is not None:
            loss_local = local_alignment_loss(outputs["patch_logits"], masks, roi)
            loss_dict["local"] = loss_local
        else:
            loss_local = torch.zeros((), device=loss_text.device)

        loss_div = d_diversity_loss(anchors, self.margin_d)
        loss_dict["div"] = loss_div

        total = (
            self.w_text * loss_text
            + self.w_global * loss_global
            + self.w_local * loss_local
            + self.w_div * loss_div
        )
        loss_dict["total"] = total
        return loss_dict
