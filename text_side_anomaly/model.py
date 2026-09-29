"""主模型：冻结 BiomedCLIP(open_clip) + 残差文本 Adapter + 全局/局部对齐。

BiomedCLIP 用 open_clip 加载（transformers 的 CLIPModel 无法识别该权重）。

对应文档：
    1) 三层 normal/abnormal 提示 → 多属性文本锚点；
    2) 冻结文本编码器上加残差 Adapter（W_down/W_up + λ_t，作用于 512 维投影空间）；
    3) 全局对齐（CLS → 图像级判断）+ 局部对齐（patch → 病灶定位）；
    4) 文本侧 margin 损失（见 losses.py）。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .inlayer_adapter import (
    DEFAULT_LAYERS,
    DEFAULT_POSITIONS,
    OrganAdapterBank,
    adapter_param_report,
    install_inlayer_adapters,
)
from .prompts import ThreeLevelPrompts
from .text_adapter import ResidualTextAdapter


class TextSideAnomalyModel(nn.Module):
    """冻结 BiomedCLIP，仅训练文本残差 Adapter 与融合权重。

    `inlayer` 给定时，额外在文本塔**内部**插入按器官/病分的瓶颈适配器
    （见 `inlayer_adapter.py`）。默认 None = 与改动 7~16 的行为逐位一致，
    保证已有 prompt-set 与全部老 ckpt 不受影响。
    """

    def __init__(self, cfg: Config, inlayer: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg

        # ---- 冻结的 BiomedCLIP ----
        import open_clip

        loaded = open_clip.create_model_from_pretrained(cfg.model_name)
        self.clip = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # ---- 层内适配器（可选）---- #
        # **必须在上面那个冻结循环之后安装** —— 否则新插的模块会被一起冻掉。
        self.inlayer_bank: Optional[OrganAdapterBank] = None
        self.inlayer_hits: Dict[str, int] = {}
        if inlayer:
            bank = OrganAdapterBank(
                organs=inlayer["organs"],
                layers=inlayer.get("layers", DEFAULT_LAYERS),
                positions=inlayer.get("positions", DEFAULT_POSITIONS),
                d_model=inlayer.get("d_model", 768),      # 塔内 hidden，不是 512
                bottleneck=inlayer.get("bottleneck", 64),
                lambda_t=inlayer.get("lambda_t", 0.1),
            )
            bank.set_active(inlayer.get("organ") or bank.organ_names()[0])
            self.inlayer_hits = install_inlayer_adapters(self.clip, bank)
            self.inlayer_bank = bank
            print(adapter_param_report(bank))
            # 挂载点数 = 层数 × 位置数；hits 计数器要等前向才会填，构造时是空的
            n_mount = len(bank.layers) * len(bank.positions)
            print(f"[inlayer] 挂载 {n_mount} 个点：层{bank.layers} × 位置{list(bank.positions)}"
                  f"（hits 计数器在前向后填充）")

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
    # 层内适配器 / 加载兼容
    # ------------------------------------------------------------------ #
    @property
    def active_organ(self) -> Optional[str]:
        return None if self.inlayer_bank is None else self.inlayer_bank.active

    def set_active_organ(self, organ: Optional[str]) -> None:
        """切换器官：只有那套层内适配器可训，其余全部冻住。"""
        if self.inlayer_bank is not None:
            self.inlayer_bank.set_active(organ)

    def lock_backbone_eval(self) -> None:
        """把冻结主干锁在 eval，只让适配器留在 train。

        坑：`model.train()` 会**递归**把 `self.clip` 也置成 train，于是 BERT 自带的
        dropout(0.1) 在训练时是开的，而评估时是关的 —— 训练时的锚点和评估时的锚点
        不是同一个分布。层内适配器在塔内，这件事的影响被放大，训练循环里每步都要调。
        """
        self.clip.eval()
        if self.inlayer_bank is not None:
            self.inlayer_bank.train()

    def load_compat(self, path, map_location="cpu", verbose: bool = True) -> None:
        """宽松加载。

        层内适配器会新增 `inlayer_bank.*` 键，老 ckpt（thymoma_*.pt，改动 7~16 产的）
        没有它们，用 `strict=True` 会直接报 Missing key。这里用 `strict=False` 并给出
        提示：缺 inlayer_bank 属预期，缺别的键才要警惕。
        """
        sd = torch.load(path, map_location=map_location)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if verbose:
            inlayer_missing = [k for k in missing if k.startswith("inlayer_bank")]
            others = [k for k in missing if not k.startswith("inlayer_bank")]
            if inlayer_missing:
                print(f"[load] 缺 {len(inlayer_missing)} 个层内适配器键（老 ckpt 属预期，"
                      f"从零初始化开始）")
            if others:
                print(f"[load] ⚠ 缺 {len(others)} 个非适配器键：{others[:4]}"
                      f"{' …' if len(others) > 4 else ''}")
            if unexpected:
                print(f"[load] ⚠ 多 {len(unexpected)} 个键：{unexpected[:4]}"
                      f"{' …' if len(unexpected) > 4 else ''}")
            if not missing and not unexpected:
                print("[load] 严格一致")

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
        """只做一次视觉前向，得到 CLS 与 patch 特征。

        ms_layers 非空时走多尺度：patch 特征取多个 ViT block（含归一化后的中间层）
        投影后平均。CLS 仍取末层（forward_intermediates 的中间层不含 CLS token，
        正好让图像级判断这条路径与单层版完全一致，改动只作用于定位）。
        """
        trunk = self.clip.visual.trunk
        head = self.clip.visual.head
        ms_layers = getattr(self.cfg, "ms_layers", None)

        if ms_layers:
            final, inter = trunk.forward_intermediates(
                pixel_values, indices=list(ms_layers), norm=True, output_fmt="NLC"
            )
            f_cls = F.normalize(head(final[:, 0]), dim=-1)              # (B, 512)
            proj = torch.stack([head(t) for t in inter], dim=0)         # (L, B, N, 512)
            f_patch = F.normalize(proj.mean(dim=0), dim=-1)             # (B, N, 512)
        else:
            feats = trunk.forward_features(pixel_values)                # (B, 1+N, 768)
            proj = F.normalize(head(feats), dim=-1)                     # (B, 1+N, 512)
            f_cls = proj[:, 0]                                          # (B, 512)
            f_patch = proj[:, 1:]                                       # (B, N, 512)

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
            # 逐层 patch logits (B, L, 2, h, w)：给「每层各自也要能定位」那份损失用。
            # 融合 logits 是 softmax 凸组合，只监督融合结果的话，单层可以烂得任意，
            # 只要加权后是好的即可 —— docx 改动 11 实测正是这样，单层被多样性项
            # 推成「不一样地差」而无人约束。
            "patch_logits_per_level": torch.stack(
                [patch_logits_list[i].reshape(B, 2, h, w) for i in range(L)], dim=1
            ),
            "anomaly_map": anomaly_map,
            "anomaly_maps": torch.stack(anomaly_maps_list, dim=1),
            "cls_logits_per_level": torch.stack(cls_logits_list, dim=1),
            "fusion_weights": fw.detach(),
        }
