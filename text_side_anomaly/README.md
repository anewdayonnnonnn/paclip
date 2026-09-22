# 文本侧改进方法：基于 BiomedCLIP 的 brain MRI 异常检测

按文档《文本侧改进方法》实现：在冻结的 **BiomedCLIP** 之上，把一对
`normal/abnormal` 提示扩展为**三层多属性文本锚点**，加**残差 Adapter**，
配合**全局 + 局部对齐**完成图像级异常判断与病灶定位。

## 方法（对应文档四部分）

### 1. 三层提示词 → 多属性文本锚点

```
Level 1（器官/组织）: "a normal brain MRI"        / "an abnormal brain MRI"
Level 2（结构/病灶）: "… with intact anatomy"     / "… with a focal lesion"
Level 3（纹理/密度）: "… homogeneous texture"     / "… irregular texture"
```

每层 `normal`/`abnormal` 各一组锚点 `t_n^l`、`t_a^l`，三层**各自独立**与图像
特征匹配，输出各自判别结果，最后**融合**（可学习权重）。见 `prompts.py`。

### 2. 残差文本 Adapter

冻结文本编码器，只训练 `W_down`、`W_up` 与残差比例 `λ_t`：

```
r = W_up( ReLU( W_down( LayerNorm(h) ) ) )
t = h + λ_t · r        # λ_t 初始 0.05~0.10；bottleneck ∈ {64,128,256}
```

见 `text_adapter.py`（`up` 零初始化，初始残差为 0，训练更稳）。

### 3. 全局 + 局部对齐

- **全局**：CLS → 三层 normal/abnormal 锚点，融合后 softmax → 图像级异常概率。
- **局部**：patch → 三层 normal/abnormal 锚点，融合后逐像素 softmax → 异常图；
  **病灶内 patch 对齐异常锚点、病灶外 patch 对齐正常锚点**。见 `model.py`。

### 4. 文本侧损失（margin m）

```
L_text = Σ_l  max(0, cos(t_a^l, t_n^l) − m)      # m 为允许的最大相似度
```

另有 `L_global`（CLS 交叉熵）、`L_local`（逐像素交叉熵）。见 `losses.py`。

## 目录结构

```
text_side_anomaly/
├── config.py          # 超参数（bottleneck/λ_t/margin/三层/温度）
├── prompts.py         # 三层 normal/abnormal 提示锚点
├── text_adapter.py    # 残差 Adapter（W_down/W_up + λ_t）
├── model.py           # 冻结 BiomedCLIP + 全局/局部对齐 + 三层融合
├── losses.py          # 文本分离 / 全局 / 局部 损失
├── metrics.py         # 异常检测通用指标（AUROC/AP/F1/Dice/IoU/pixelAUROC）
├── dataset.py         # 2D 切片 / 3D 体积(.nii.gz) 加载 + 掩码
├── train.py           # 训练 / 推理 / 评估
└── README.md
demo_synthetic.py      # 离线自测脚本（合成数据 + 模拟骨干）
```

## 依赖

```bash
pip install torch transformers pillow numpy scikit-learn
```

需要联网（或已缓存）下载 `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`。

## 数据格式

```
data/
  normal/     *.png|*.nii.gz    正常
  abnormal/   *.png|*.nii.gz    异常
masks/                           # 可选，训练局部对齐用
  abnormal/   *.png|*.nii.gz    病灶掩码（>0 为病灶，与 abnormal 同名）
```

## 训练 + 评估

```bash
python -m text_side_anomaly.train \
    --data_root data --mask_root masks \
    --data_format slice|volume \
    --bottleneck 128 --lambda_t 0.05 --margin 0.3 --epochs 20
```

训练结束后自动打印**图像级 AUROC / AP / F1 / ACC** 与（有掩码时）**像素级
Dice / IoU / pixelAUROC**。

## 推理

```python
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.train import build_tokenizer
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPTS

model = TextSideAnomalyModel(cfg).to(cfg.device)
tok = build_tokenizer(cfg)
enc = model.encode_anchors(DEFAULT_BRAIN_MRI_PROMPTS, tok)
out = model(image_tensor.unsqueeze(0).to(cfg.device), enc)
# out["cls_probs"] 图像级异常概率；out["anomaly_map"] (14,14) 异常热图
```

## 离线自测（本仓库可直接跑）

本环境无法下载 BiomedCLIP 权重与真实数据，`demo_synthetic.py` 用合成脑组织
数据 + 模拟冻结骨干跑通全流程并给出异常检测指标（见脚本顶部说明）。
