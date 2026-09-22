"""主模型：冻结 BiomedCLIP(open_clip) + 残差文本 Adapter + 全局/局部对齐。

BiomedCLIP 用 open_clip 加载（transformers 的 CLIPModel 无法识别该权重）。

对应文档：
    1) 三层 normal/abnormal 提示 → 多属性文本锚点；
    2) 冻结文本编码器上加残差 Adapter（W_down/W_up + λ_t，作用于 512 维投影空间）；
    3) 全局对齐（CLS → 图像级判断）+ 局部对齐（patch → 病灶定位）；
    4) 文本侧 margin 损失（见 losses.py）。
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .prompts import ThreeLevelPrompts
from .text_adapter import ResidualTextAdapter


class TextSideAnomalyModel(nn.Module):
    """冻结 BiomedCLIP，仅训练文本残差 Adapter 与融合权重。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ---- 冻结的 BiomedCLIP ----
        import open_clip

        loaded = open_clip.create_model_from_pretrained(cfg.model_name)
        self.clip = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # open_clip tokenizer：list[str] -> (B, L) 长整型张量
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

        # BiomedCLIP 投影维度为 512
        self.projection_dim = getattr(self.clip, "embed_dim", cfg.text_hidden)

        # ---- 残差文本 Adapter（只训练 W_down/W_up + λ_t，作用于 512 维投影空间）----
        self.text_adapter = ResidualTextAdapter(
            d_model=self.projection_dim,
            bottleneck=cfg.bottleneck,
            lambda_t=cfg.lambda_t,
            lambda_t_learnable=cfg.lambda_t_learnable,
            dropout=cfg.adapter_dropout,
        )

        # ---- 三层融合权重（可学习，softmax 归一）----
        n_levels = len(cfg.levels)
        if cfg.fusion_learnable:
            self.fusion_weights = nn.Parameter(torch.zeros(n_levels))
        else:
            self.register_buffer("fusion_weights", torch.zeros(n_levels))

    # ------------------------------------------------------------------ #
    # 文本侧
    # ------------------------------------------------------------------ #
    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, L) → 适配后归一化文本嵌入 (B, 512)。"""
        t = self.clip.encode_text(tokens)          # (B, 512) 投影后，未归一化
        t = self.text_adapter(t)                   # 残差适配
        return F.normalize(t, dim=-1)

    def _encode_list(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.cfg.device)  # (N, L)
        t = self.encode_text_tokens(tokens)
        return t.mean(dim=0) if t.size(0) > 1 else t.squeeze(0)

    def encode_anchors(self, prompts: ThreeLevelPrompts) -> Dict[str, Dict[str, torch.Tensor]]:
        """三层提示词 → 每层 normal/abnormal 文本锚点（归一化）。"""
        anchors: Dict[str, Dict[str, torch.Tensor]] = {}
        for lvl in prompts.levels:
            anchors[lvl] = {
                "normal": F.normalize(self._encode_list(prompts.normal[lvl]), dim=0),
                "abnormal": F.normalize(self._encode_list(prompts.abnormal[lvl]), dim=0),
            }
        return anchors

    # ------------------------------------------------------------------ #
    # 图像侧
    # ------------------------------------------------------------------ #
    def encode_image(self, pixel_values) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # 只做一次视觉前向，同时得到 CLS 与 patch 特征
        feats = self.clip.visual.trunk.forward_features(pixel_values)  # (B, 1+N, 768)
        proj = F.normalize(self.clip.visual.head(feats), dim=-1)       # (B, 1+N, 512)
        f_cls = proj[:, 0]                                             # (B, 512)
        f_patch = proj[:, 1:]                                          # (B, N, 512)
        h = w = int(f_patch.size(1) ** 0.5)
        return f_cls, f_patch, (h, w)

    # ------------------------------------------------------------------ #
    # 前向：三层独立匹配后融合
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values, anchors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        f_cls, f_patch, (h, w) = self.encode_image(pixel_values)
        B = f_cls.size(0)
        levels = list(anchors.keys())
        L = len(levels)

        fw = F.softmax(self.fusion_weights, dim=0)          # (L,)

        cls_logits_list = []
        patch_logits_list = []
        anomaly_maps_list = []
        for lvl in levels:
            t_n = anchors[lvl]["normal"]
            t_a = anchors[lvl]["abnormal"]

            s_n = f_cls @ t_n
            s_a = f_cls @ t_a
            cls_logits_list.append(torch.stack([s_n, s_a], dim=1) / self.cfg.temperature)

            pn = f_patch @ t_n
            pa = f_patch @ t_a
            patch_logits_list.append(torch.stack([pn, pa], dim=1) / self.cfg.temperature)
            anomaly_maps_list.append((pa - pn).reshape(B, h, w))

        cls_logits = sum(fw[i] * cls_logits_list[i] for i in range(L))
        patch_logits = sum(
            fw[i] * patch_logits_list[i].reshape(B, 2, h, w) for i in range(L)
        )
        anomaly_map = sum(fw[i] * anomaly_maps_list[i] for i in range(L))

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_logits.softmax(dim=1)[:, 1],
            "patch_logits": patch_logits,
            "anomaly_map": anomaly_map,
            "anomaly_maps": torch.stack(anomaly_maps_list, dim=1),
            "cls_logits_per_level": torch.stack(cls_logits_list, dim=1),
            "fusion_weights": fw.detach(),
        }
