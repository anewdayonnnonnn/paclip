"""量层间差向量 d_l 的两两余弦，看"训练把三层抹平"在层内适配器上是否同样发生。

改动 14/16 记的是输出端 adapter 的测量（训练后 cos≈0.96，三个 d_l 平行）。层内那版
参数多 6 倍、自由度更大，必须单独量一次，否则"还是在过拟合支持集"这个机制解释只是推测。

用法（服务器）：
    python _diag_inlayer_cos.py fewshot_ckpt/*_s0.pt
"""
import itertools
import sys

import torch
import torch.nn.functional as F

from text_side_anomaly.config import Config
from text_side_anomaly.model import TextSideAnomalyModel

import thymoma_local as TL

ORGAN = "thymoma"
PROMPT = "sentence"


def build(mode: str, prompts):
    inlayer = None
    if mode == "inlayer":
        inlayer = {"organs": [ORGAN], "organ": ORGAN, "bottleneck": 64,
                   "layers": [8, 9, 10, 11], "positions": ("attn", "ffn")}
    cfg = Config(device="cuda", levels=prompts.levels)
    model = TextSideAnomalyModel(cfg, inlayer=inlayer).to("cuda")
    if mode == "inlayer":
        # 与 fewshot_run.build_model 完全一致：输出端压恒等并冻住
        with torch.no_grad():
            model.text_adapter.lambda_t.data.zero_()
        for p in model.text_adapter.parameters():
            p.requires_grad = False
    return model


@torch.no_grad()
def report(model, prompts, tag: str):
    model.eval()
    anchors = model.encode_anchors(prompts)
    ds = {}
    for lvl in prompts.levels:
        d = anchors[lvl]["abnormal"] - anchors[lvl]["normal"]
        ds[lvl] = F.normalize(d, dim=0)
    parts = [f"L{i}-L{j}={F.cosine_similarity(ds[i], ds[j], dim=0).item():+.4f}"
             for i, j in itertools.combinations(prompts.levels, 2)]
    print(f"{tag:<28} " + "  ".join(parts), flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    prompts = TL.PROMPT_SETS[PROMPT]

    # 训练前的参考（W_up 零初始化 → 恒等，等于冻结编码器上的原始几何）
    for mode in ("output", "inlayer"):
        m = build(mode, prompts)
        report(m, prompts, f"[训练前] {mode}")
        del m
        torch.cuda.empty_cache()

    for path in sys.argv[1:]:
        mode = "inlayer" if "inlayer" in path else "output"
        m = build(mode, prompts)
        sd = torch.load(path, map_location="cuda")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  ! 意外的键 {unexpected[:3]}")
        report(m, prompts, f"[训练后] {path.split('/')[-1]}")
        del m, sd
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
