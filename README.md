# 文本侧改进的医学图像异常检测

用**冻结的 BiomedCLIP** 做无需病灶标注训练的医学图像异常检测：把「正常 / 异常」写成文本提示词，
在文本嵌入空间里构造锚点，异常图由 **patch 特征与「异常锚点 − 正常锚点」差向量的相似度**给出。
训练只动文本侧的一小块（残差 Adapter + 融合权重），图像塔全程冻结。

两个数据集支线：

| 支线 | 数据 | 规模 | 现状 |
|---|---|---|---|
| **脑 MRI** | BraTS2021 肿瘤切片 | train 1956 / test 820 | 主线，含多尺度特征与 ROI 门控 |
| **胸腺瘤 CT** | 纵隔 CT 病灶切片 | train 2863 / test 375 | 支线，做提示词与 Adapter 插法的对照 |

改动都记在 [`docs/文本测改进方法.docx`](docs/文本测改进方法.docx) 的「改动存档」章节，
纯文本副本在 [`docs/改动存档/`](docs/改动存档/)。

---

## 目录结构

```
text_side_anomaly/       核心包（模型 / 损失 / 数据 / 指标 / 可视化）
thymoma_local.py         ★ 胸腺瘤入口：训练 + 定位评估 + 出热力图
fewshot_run.py           ★ 小样本 × Adapter 插法四组对照（断点续跑）
brain/                   脑支线脚本（数据不在仓库里，见下）
tools/                   开发运维工具、诊断脚本、strip_ckpt.py
docs/                    方法文档 + 改动存档
results/                 指标表、结果 JSON、训练结果解包
```

`★` = 直接可运行的入口，留在根目录（它们按相对路径找 `thymoma_slices/`，移走就跑不起来）。

**以下只在本机、不进仓库**（`.gitignore` 挡住）：

```
ckpt/        21 个胸腺实验 ckpt。⚠ 已用 tools/strip_ckpt.py 剥掉冻结主干，
             748MB -> 0.5MB。传 --out / --eval-only 时要写全路径，例如
                 python thymoma_local.py --eval-only --out ckpt/thymoma_local.pt
fewshot_ckpt/  小样本四组对照的 ckpt（同样已剥离主干）
heatmaps/      36 个历史热力图目录（在患者影像上叠加，属患者衍生数据）
logs/          各次训练的原始日志
figures/       对比图
docs/备份/     每次追加改动时自动生成的 docx 备份
```

<details>
<summary>各文件一句话说明</summary>

**核心包 `text_side_anomaly/`**
| 文件 | 作用 |
|---|---|
| `model.py` | 主模型：冻结 BiomedCLIP + 残差文本 Adapter + 三层融合 |
| `text_adapter.py` | 输出端残差瓶颈 Adapter（512 维投影空间） |
| `inlayer_adapter.py` | **层内** Adapter：猴补丁 `forward` 插进文本塔第 8-11 层的 attn / 整层末 |
| `config.py` | 超参。⚠ 见「两个坑」 |
| `prompts.py` | 三层 normal/abnormal 提示词锚点 |
| `losses.py` | 文本分离损失（单边/有界）、多样性项、全局/局部对齐 |
| `dataset.py` | 切片/体积数据集，含解剖 ROI（强度阈值，无需标注） |
| `roi.py` | ROI 掩码与异常图门控 |
| `visualize.py` | 全局分位数标定 + 引导滤波上采样 + 配色 |
| `metrics.py` | 图像级 / 像素级指标 |
| `train.py` | 通用训练入口（`--seed` / `--ms-layers` 见下） |
| `thymoma_dataset.py` | 胸腺瘤切片数据集与划分 |

**`brain/`** — `eval_final.py`（五配置对比 + 出图）、`eval_seeds.py`（多种子最终评测）、
`train_ms.py` / `train_roi.py` / `train_fixed.py`、`prep_brain.py`、`make_heatmaps.py` 等。

**`tools/`** — `_remote.py`（SSH/SFTP 助手）、`_docx_append.py`（往 docx 追加改动存档）、
`_mk_csv.py`（从评测日志导 CSV）、`_smoke_inlayer.py`（层内 Adapter 不变量冒烟）、
早期的 SOTA 基线对比脚本等。

</details>

---

## 环境

```bash
pip install -r requirements.txt
```

BiomedCLIP 权重首次运行会自动走 HuggingFace 下载。国内机器建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 实测 32 MB/s
export HF_HOME=/path/to/hf_cache
```

---

## 怎么跑

### 脑 MRI

数据放 `D:/brain_dl/`（`brain_data/{train,test}/{normal,abnormal}` + `brain_masks/`），
脚本里写死了这个路径，换机器改 `BRAIN_DL` 常量。

```bash
# 单层基线（必须显式 --ms-layers none，否则默认是多尺度，见「两个坑」）
python -m text_side_anomaly.train \
    --data_root D:/brain_dl/brain_data/train \
    --mask_root D:/brain_dl/brain_masks \
    --ms-layers none --seed 0 --save_dir D:/brain_dl/ckpt_base_s0

# 多尺度
python -m text_side_anomaly.train ... --ms-layers 5,8,11 --seed 0

# 五种配置对比 + 出热力图
python brain/eval_final.py
```

### 胸腺瘤 CT

数据 `thymoma_slices/`（由 `tools/prepare_thymoma_slices.py` 从 3D 体积切出）。

```bash
python thymoma_local.py --seed 0 --out ckpt/thymoma_local.pt
python thymoma_local.py --eval-only --out ckpt/thymoma_local.pt      # 只评估 + 出图
python fewshot_run.py --groups A,B,C,D --seeds 0,1,2 --cooldown 0   # 四组对照
```

ckpt 是 `model.state_dict()` 全量保存，含冻结主干（748MB）。**跑完用
`tools/strip_ckpt.py --in-place ckpt/*.pt` 剥成 0.5MB** —— 主干逐位相同、随时能从 HF 重下，
不剥就是几十份 748MB 堆在盘上。加载走 `model.load_compat()`，它认得缺 `clip.*` 的情况。

---

## 主要结果

### 脑 MRI（test 820 张：异常 573 / 正常 247）

基线单层与多尺度是 **4 次独立训练**（原 ckpt + 3 个种子）的均值；后三行是单次结果。
完整表见 [`results/eval_seeds_metrics.csv`](results/eval_seeds_metrics.csv) 与
[`results/eval_final_metrics.csv`](results/eval_final_metrics.csv)。

| 配置 | imgAUROC | Dice | IoU | pxAUROC | 脑外红斑(异常/正常) |
|---|---|---|---|---|---|
| 基线单层 | 0.9714 | 0.5573 | 0.3863 | 0.9405 | 26.4% / 72.1% |
| 多尺度 | 0.9716 | 0.6013 | 0.4299 | 0.9565 | 15.7% / 53.3% |
| 多尺度 + ROI 门控 | 0.9722 | 0.6125 | 0.4415 | 0.9728 | **0% / 0%** |
| ROI 训练 | 0.9733 | 0.2351 | 0.1332 | 0.8486 | 88.6% / 99.3% |
| **ROI 训练 + ROI 门控** | **0.9733** | **0.7080** | **0.5480** | **0.9820** | **0% / 0%** |

- **多尺度 patch 特征**：+0.044 Dice —— 是种子极差（0.007）的 6 倍，增益真实
- **ROI 门控**消除脑外红斑（53.3% → 0%）；**ROI 训练必须与门控捆绑**，
  单用 ROI 训练反而把 Dice 打到 0.235（门控不给它兜底）

### 胸腺瘤 CT（test 375 张）

| 配置 | Dice（3 种子均值） | IoU | pxAUROC |
|---|---|---|---|
| 输出端 Adapter + 全监督 | **0.6274** | 0.4571 | 0.9795 |
| 层内 Adapter + 全监督 | 0.6188 | 0.4481 | 0.9786 |
| 输出端 Adapter + K=5 | 0.4875 | 0.3236 | 0.9473 |
| 层内 Adapter + K=5 | 0.4794 | 0.3165 | 0.9456 |

- **Adapter 插在层内并不更好**（参数多 6 倍、两种支持集下都略差）。原因不在「插在哪」：
  唯一监督在融合处，多出来的自由度只会去买「三层锚点平行」和记支持集。
  训练后层间 `cos(d)` 被抹到 0.95（层内）vs 0.92（输出端），层内更彻底
- K=5 的种子跨度 0.11，**单种子结论不可信**

---

## 两个坑（必读）

### 1. 评估口径必须与训练口径一致

`Config.ms_layers` 的**默认值已经从「单层」改成了 `[5, 8, 11]`**，而 `train.py` 早先没有
显式设它 —— 于是直接跑 `train.py` 训出来的是**多尺度**模型，不是基线。拿单层口径去评它，
会得出「重训比基线差 0.08」这种假结论（实测踩过）。

判别式很干净：同一个 ckpt 按两种口径评，**训练与评估一致的那边会明显更好**。

| ckpt | 单层口径 Dice | 多尺度口径 Dice |
|---|---|---|
| 基线 ckpt（单层训的） | **0.5535** | 0.4245 |
| 多尺度 ckpt | 0.4731 | **0.6039** |

现在 `train.py` 有 `--ms-layers`：`none` = 单层基线，`5,8,11` = 多尺度。要基线就**显式传 none**。

### 2. 不给 `--seed` 就没有可比性

`train.py` 原本不设随机种子。现在有 `--seed`，**凡是要对比的实验都必须给**。
脑的种子极差 0.007、胸腺 0.002 —— 给了种子才敢做单次对比。

---

## 数据

**不在仓库里**（含患者影像），各自单独获取与准备：

- `thymoma_slices/` — 胸腺瘤纵隔 CT，用 `tools/prepare_thymoma_slices.py` 从体积切出
- `D:/brain_dl/` — BraTS2021 脑 MRI 切片与掩膜，`brain/prep_brain.py` 生成

热力图（在患者 CT/MRI 上叠加热区与 GT 轮廓）同样不进仓库，见 `.gitignore`。
