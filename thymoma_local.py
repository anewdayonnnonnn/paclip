"""用胸腺瘤 ROI 数据训练定位分支（局部对齐），量化验证热力图定位。

无正常（无病灶）病例 → 不做全局二分类（w_global=0），只训练：
    - 文本分离损失（normal/abnormal 锚点拉开）
    - 局部对齐损失（病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点）

评估用 patch 级 AUROC / Dice / IoU（metrics.pixel_metrics），并出热力图叠加 GT 肿瘤轮廓。
"""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import _gray2clip
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.metrics import pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import ThreeLevelPrompts
from text_side_anomaly.thymoma_dataset import (
    ThymomaSliceDataset,
    case_ids,
    make_slice_splits,
    sample_support,
)

# 显示层（改动 4 在脑 MRI 线验证过的同一套）：全局分位数标定 + 引导滤波上采样。
# 纯出图用，绝不参与任何指标计算。
from text_side_anomaly.visualize import AmapCalibration, upsample_guided

ROOT = "data/本院图像和ROI_能共享_2025-11-18整理/全部图像及ROI"
AMAP_SCALE = 0.3  # 旧口径的固定尺度（同 generate_heatmaps.py）。现仅 --legacy-display 用
ORGAN = "thymoma"  # 层内适配器库的键（OrganAdapterBank）；换器官 = 换一个键

# 胸腺瘤 CT 三级提示（前纵隔肿块）
#
# 2026-09-27 改写。旧版三层是**同一个病灶的三个名字**（anterior mediastinal mass /
# thymoma / mediastinal tumor），L3 是 L1 的同义词，完全没有体现文档「粗粒度→细粒度」
# 的多属性设计；且六句共享 "a normal chest CT" / "a chest CT with ..." 模板前缀，
# 而 BiomedCLIP 的句向量由前缀主导，换尾巴属性几乎推不动余弦。
#
# 实测（冻结编码器，_diag_prompts.py）：
#   旧版  跨层 cos(abn,abn)=0.927  cos(norm,norm)=0.945  cos(d)=0.688
#   新版  跨层 cos(abn,abn)=0.836  cos(norm,norm)=0.874  cos(d)=0.230
# 其中 d_l = t_a^l − t_n^l，异常图 pa−pn 只由它决定 —— 三个 d_l 平行则三层 map 必然
# 相同、融合权重失效。基线实测：训练前 cos(d)=0.688，训练后升到 0.984，三层 map 相关
# 0.994（单边 hinge 把 t_a 推到 −t_n，d_l≈2t_a^l，跨层 t_a 相似度直接决定 map 相同）。
#
# 新写法三层各占一个属性维度，且**句式框架互不相同**：
#   L1 器官/组织级 —— 形容词 + 模态：        a normal chest CT / an abnormal chest CT
#   L2 结构/病灶级 —— 模态 + showing + 结构： chest CT showing ...
#   L3 纹理/密度级 —— 主语前置 + on 模态：     ... density with ... margins on chest CT
THYMOMA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": ["chest CT showing normal mediastinal structures"],
        "3": ["homogeneous mediastinal fat density with sharp margins on chest CT"],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": ["chest CT showing a soft tissue mass in the anterior mediastinum"],
        "3": ["ill-defined soft tissue density with irregular blurred margins on chest CT"],
    },
)

# 2026-09-28 加：短名词短语版。动机 —— 上面那版三层都用整句，而整句模板
# （"a chest CT with X"）会把三个差向量 d_l 压进同一个子空间：实测 cos(d)=0.230。
# 换成互不共享框架的短名词短语后 cos(d)=0.031（低一个数量级），三层天然近正交。
#
# 这条路的意义：② d_diversity_loss 存在的唯一理由就是"不显式要求分歧，三层就会塌成
# 一张图"。而实测②正是打垮单层的元凶（只开②，L1 的 Dice 0.6215→0.4742）。**如果提示词
# 本身就给出正交的 d，就不需要②了** —— 既拿到分歧，又不用付单层变差的代价。
#
# 保留方式：不覆盖上面那份，改用 --prompt-set 切换。因为 thymoma_local.pt 是用上面那版
# 训的，直接改掉会让标定/出图配错词表（改动 8 附录记过这个坑）。
THYMOMA_PROMPTS_SHORT = ThreeLevelPrompts(
    normal={
        "1": ["normal mediastinal tissue"],
        "2": ["normal thymus"],
        "3": ["homogeneous mediastinal fat"],
    },
    abnormal={
        "1": ["mediastinal mass"],
        "2": ["thymoma"],
        "3": ["ill-defined soft tissue"],
    },
)

# 2026-09-28 加：分层版（外部建议稿，global / attribute×4 / lesion 三个层级）。
# 我们的框架是 N 层通用（层数由 prompts.levels 决定、融合权重按层数建），而这份设计里
# 属性层有 4 个属性 —— **不能把 4 个属性平均成一层**：实测平均后 cos(d) 会从 0.128 涨到
# 0.479（正交分量互相抵消，剩下的共同分量正好撞上 L1/L3），比现用的 0.230 还差。
# 所以按 4 个属性各自成层装，共 6 层。
#
# 实测 6x6 cos(d)：均值 +0.128（优于现用 +0.230），最大 +0.785（差于现用 +0.481）。
# **已知缺陷：L1 全局 ↔ L2 结构 = +0.785** —— 'an abnormal chest CT' 与
# 'abnormal mediastinal contours' 近乎同义，这两层会出几乎一样的图。其余属性对都很好
# （L4 边缘与所有层近正交）。先按原稿跑，再决定要不要删掉冗余的那层。
THYMOMA_PROMPTS_TIERS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],                                              # 全局
        "2": ["chest CT with normal mediastinal structures"],                    # 属性·结构
        "3": ["chest CT with homogeneous mediastinal fat density"],              # 属性·密度纹理
        "4": ["chest CT with sharp and well-defined margins"],                   # 属性·边缘
        "5": ["chest CT with bilaterally symmetric structures"],                 # 属性·对称性
        "6": ["normal mediastinal tissue"],                                      # 病灶
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": ["chest CT with abnormal mediastinal contours"],
        "3": ["chest CT with heterogeneous soft tissue density"],
        "4": ["chest CT with ill-defined and blurred margins"],
        "5": ["chest CT with asymmetric mediastinal structures"],
        "6": [
            "a soft tissue mass in the anterior mediastinum",
            "irregular and blurred margins",
            "a lesion with a soft tissue mass",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：分层设计的**原版**（3 层）。上面 THYMOMA_PROMPTS_TIERS 把 4 个属性拆成了
# 4 层，那是误读 —— 原设计里 4 个属性同属 L2，作为一层的多条提示词（框架内 _mean_embed
# 会把同一层的多条做均值池化）。三层各自成层，层数仍是 3，与既有机制完全对齐。
#
# 训练前实测 cos(d)：均值 +0.479、最大 +0.557（L1-L2）。对照：现用整句 3 层 +0.230、
# 短名词短语 3 层 +0.031、6 层拆法 +0.128/+0.785。
# 注意：把 4 个属性池化后，它们各自近正交的分量（两两均值 −0.05）会互相抵消，剩下的
# 共同分量反而更贴近 L1/L3 —— 所以三层装法的 cos(d) 比 6 层拆法**更高**，不是更低。
THYMOMA_PROMPTS_TIERS3 = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": [                                                    # 属性层：4 个属性池化
            "chest CT with normal mediastinal structures",
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
        ],
        "3": ["normal mediastinal tissue"],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": [
            "chest CT with abnormal mediastinal contours",
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
        ],
        "3": [
            "a soft tissue mass in the anterior mediastinum",
            "irregular and blurred margins",
            "a lesion with a soft tissue mass",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：把 L2（属性）与 L3（病灶）合并成一层 → 2 层设计（L1 全局 + L2' 合并层）。
# 起因是"觉得 L2 和 L3 两个维度蛮接近"。但实测最接近的一对其实是 **L1-L2**（cos(d)=+0.556），
# L2-L3 只有 +0.478 —— 合并 L2+L3 反而把最冗余的 L1-L2 留在了场上。仍按需求跑。
#
# 裁剪（_mean_embed 是均值池化，条数比例即权重），两处删除：
#   1. 删属性"结构"那条（'chest CT with normal mediastinal structures' 等）。它是实测最冗余
#      的一条 —— 6 层分析里 L1(全局)-结构 = +0.785，是所有配对里最平行的。既然合并就是
#      为了去冗余，留着它等于把冗余又搬进来。删后与 L1 的分歧从 cos(d,L1)=+0.498
#      降到 +0.368，正是想要的。代价是离纯 L2 远了些（保留 L2 方向 0.771→0.690）。
#   2. 病灶留 3 条（位置形态 / 占位效应 / 信号密度），去掉 'a lesion with a soft tissue mass'
#      （与第 1 条重复）与 'irregular and blurred margins'（是片段不是病灶名，且"边缘"已由
#      属性层覆盖）。
# 实测：同层 cos(n,a)=+0.914，cos(d_L1-L2')=+0.368，保留 L2 方向 0.690。
THYMOMA_PROMPTS_MERGE23 = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": [
            # 属性（原 L2，删掉"结构"那条，见上）
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
            # 病灶（原 L3，1 条）
            "normal mediastinal tissue",
        ],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": [
            # 属性（原 L2，删掉"结构"那条）
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
            # 病灶（原 L3，3 条：位置形态 / 占位效应 / 信号密度）
            "a soft tissue mass in the anterior mediastinum",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：单层设计 —— 只有"属性池化"这一层，没有 L1 全局、没有 L3 病灶、没有融合。
#
# 依据：在 tiers3 设计内丢层是唯一让结果涨回来的操作（三层 0.6224 → L1+L2 0.6259 →
# 只留 L2 0.6265，改动 15 第 (3) 条）。但那个 0.6265 是**事后从三层模型里挑最好的层**
# （用测试集选的），不是训练时就只有这一层。本跑把这件事做干净：
# 训练时就是单一锚点对，`d_diversity_loss` 自动退化为 0（len(ds)<2），融合权重恒为 1。
#
# 要回答的问题：**"多属性提示词池化成单维锚点"这个做法本身是否有效、且一层就够？**
# 若是，课题的卖点可以从"三层融合"改成"属性池化锚点"——更简洁，且与基线的差距消失。
THYMOMA_PROMPTS_ATTRONLY = ThreeLevelPrompts(
    normal={
        "1": [
            "chest CT with normal mediastinal structures",
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
        ],
    },
    abnormal={
        "1": [
            "chest CT with abnormal mediastinal contours",
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
        ],
    },
)

PROMPT_SETS = {
    "sentence": THYMOMA_PROMPTS,        # 改动 7 的整句版，现默认，基线 ckpt 用它训的
    "short": THYMOMA_PROMPTS_SHORT,     # 短名词短语版，cos(d)=0.031
    "tiers": THYMOMA_PROMPTS_TIERS,     # 6 层版（我误把 4 个属性拆成 4 层，非原设计）
    "tiers3": THYMOMA_PROMPTS_TIERS3,   # 原设计：L1 全局 / L2 属性(4 条池化) / L3 病灶
    "merge23": THYMOMA_PROMPTS_MERGE23, # L2+L3 合并成一层 → 2 层
    "attronly": THYMOMA_PROMPTS_ATTRONLY,  # 单层：只有 4 条属性池化
}


def pre_tokenize(model, prompts, device):
    """预 tokenize 三层提示词（tokenize 结果固定，避免每 step 重复 tokenize）。"""
    return {
        lvl: {
            k: model.tokenizer(texts).to(device)
            for k, texts in [("normal", prompts.normal[lvl]), ("abnormal", prompts.abnormal[lvl])]
        }
        for lvl in prompts.levels
    }


def encode_anchors_cached(model, tok):
    """用预 tokenize 的 tokens 编码三层锚点（每 step 仍重算 adapter 前向）。"""
    anchors = {}
    for lvl, d in tok.items():
        t_n = model.encode_text_tokens(d["normal"])
        t_a = model.encode_text_tokens(d["abnormal"])
        anchors[lvl] = {
            "normal": F.normalize(t_n.mean(0) if t_n.size(0) > 1 else t_n.squeeze(0), dim=0),
            "abnormal": F.normalize(t_a.mean(0) if t_a.size(0) > 1 else t_a.squeeze(0), dim=0),
        }
    return anchors


@torch.no_grad()
def evaluate(model, anchors, loader, device):
    """测试集 patch 级定位指标（14×14）。"""
    model.eval()
    enc = model.encode_anchors(anchors)
    maps, masks = [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, enc)
        maps.append(out["anomaly_map"].cpu())
        masks.append(batch["mask"])
    maps = torch.cat(maps).numpy()
    masks = torch.cat(masks).numpy()
    return pixel_metrics(maps, masks)


# ---------------------------------------------------------------------- #
# 出图标定（改动 8 第 (4)a：把改动 4 的显示管线移植到胸腺瘤支线）
#
# **纯显示层**：这里出来的每个数都只进画家，绝不进 pixel_metrics。改动 2 第 5 条已实测
# 逐图 z-score / min-max 会把 Dice 从 0.5535 打到 0.4012 / 0.3730 —— 因为 amap 整体为负
# （本模型实测均值 −0.40）是**跨图系统性**的、本身带信息。所以标定必须是全局的（所有图
# 共用一把尺子），且只在 14×14 原生分辨率上做。
# ---------------------------------------------------------------------- #
def prepare_heatmap_dir(out_dir, clean=False):
    """出图目录卫生：文件名带分类分数 `_score{score:.2f}`，分数一变就写新文件名、旧图不被
    覆盖 —— 一个目录会攒下多次运行的图。改动 8 第 (3) 条因此把「旧 vs 旧」配成「旧 vs 新」，
    得出过完全相反的结论。

    目录非空就报错退出（不静默删除），要清空得显式加 --clean-heatmap-dir 或换个目录。
    放在 main 开头调，免得训完 15 个 epoch 才因为目录脏而丢掉出图。
    """
    os.makedirs(out_dir, exist_ok=True)
    stale = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not stale:
        return
    if not clean:
        raise SystemExit(
            f"[heatmap] {out_dir}/ 里已有 {len(stale)} 个 png（多半是上一次运行留下的；"
            f"文件名带 _score，分数变了就不覆盖）。确认可以清空就加 --clean-heatmap-dir，"
            f"或换一个 --heatmap-dir。"
        )
    for p in stale:
        os.remove(p)
    print(f"[heatmap] 已清空 {out_dir}/（删除 {len(stale)} 个 png）")


@torch.no_grad()
def collect_amaps(model, anchors, loader, device):
    """跑一遍参考集，收集 14×14 原生 amap（只用于拟合出图标定，不进指标）。"""
    model.eval()
    enc = model.encode_anchors(anchors)
    maps = []
    for batch in loader:
        out = model(batch["image"].to(device), enc)
        maps.append(out["anomaly_map"].cpu().numpy())
    return np.concatenate(maps, axis=0)


def fit_calibration(model, anchors, splits, device, split="test", batch_size=16,
                    lo_pct=50.0, hi_pct=99.5, max_slices=None):
    """在该模型自己的 amap 分布上拟合全局分位数标定。

    参考集默认 test：划分 seed=0 是钉死的，每次跑都是同一批 375 张切片，所以「模型换了」
    不会和「切片换了」混在一起（改动 8 第 (3) 条就是配错图被坑的）。

    胸腺瘤 233 例全有病灶、**没有正常病例** → fit_from_normals 不可用，走「无正常切片」
    分支 fit_from_stats，在含病灶的混合分布上取分位数（实测 P50 与非病灶 P50 几乎重合，
    天然满足「正常组织 = 色阶 0」）。用无病灶像素的分位数当上界会严重过饱和，故不可取。
    """
    if split == "all":
        files = splits["train"] + splits["val"] + splits["test"]
    else:
        files = splits[split]
    if max_slices:
        files = files[:max_slices]

    loader = DataLoader(ThymomaSliceDataset("thymoma_slices", files=files),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    calib = AmapCalibration.fit_from_stats(
        collect_amaps(model, anchors, loader, device), lo_pct=lo_pct, hi_pct=hi_pct)
    # 把这把尺子打进日志：红度不可跨 ckpt 比较（lo/hi 各自在本模型分布上拟合），
    # 出图前先能看见各自的分位数
    print(f"[display] 标定参考集={split} 切片={len(files)} "
          f"lo=P{lo_pct:g}={calib.lo:+.4f} hi=P{hi_pct:g}={calib.hi:+.4f} "
          f"（旧口径等效 lo=0.0 hi={AMAP_SCALE}）")
    return calib


def overlay_heatmap_with_gt(gray_pil, amap, score, gt_mask_pil, out_path,
                            calib=None, alpha=0.45):
    """热力图叠加到窗化灰度 CT，并把 GT 肿瘤轮廓画成绿色。

    calib 非 None = 改动 4 的新口径：**先在 14×14 原生分辨率上做全局分位数标定，再做
    引导滤波上采样**（以窗化 CT 为引导）。顺序不能反 —— 反了引导滤波会改掉数值范围，
    在原生分辨率上测出来的分位数就失准了。
    calib=None = 改动 8 之前的旧口径（固定尺度 AMAP_SCALE + 双线性），供 A/B 对照。

    alpha 固定 0.45（不跟 visualize.render 的 0.5），使新旧图的差别**只**来自标定与上采样，
    不掺第三个变量。同理出图挑的 6 张切片也不改，否则 A/B 就配不上对了。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    w, h = gray_pil.size

    if calib is None:
        # 旧口径：与改动 8 之前逐字节一致
        local = np.clip(amap / AMAP_SCALE, 0.0, 1.0)
        up = Image.fromarray((local * 255).astype(np.uint8)).convert("L")
        up = up.resize((w, h), Image.BILINEAR)
        norm = np.asarray(up, dtype=np.float32) / 255.0
    else:
        # 新口径：先在 14×14 标定（clip 是截断，不做逐图归一化），再引导滤波上采样
        gray01 = np.asarray(gray_pil.convert("L"), dtype=np.float32) / 255.0
        norm = upsample_guided(calib(amap), gray01, radius=4, eps=1e-3)

    heat = (cm.jet(np.clip(norm, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)

    # 胸腺瘤定位场景全局分支未训练（w_global=0），cls_probs 无意义，
    # 故不用 score 门控，直接用局部异常图出图。
    gray = np.asarray(gray_pil.convert("RGB"), dtype=np.float32)
    blend = ((1.0 - alpha) * gray + alpha * heat).astype(np.uint8)

    # GT 轮廓（绿色）。注意这里直接 filter 0/255 的 PIL 图，阈值 30 才有意义；
    # visualize.colorize 走的是数组路径（gt.astype(np.uint8) + FIND_EDGES），喂 0/1 的
    # mask 会静默画不出线 —— 所以本次移植没有把这段换成 colorize。
    edge = np.asarray(gt_mask_pil.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    blend[edge > 30] = [0, 255, 0]
    Image.fromarray(blend).save(out_path)


@torch.no_grad()
def save_heatmaps(model, anchors, files, device, out_dir, n=6, calib=None):
    """采样与切片挑选逻辑不变，只把标定对象透传给 overlay_heatmap_with_gt。

    目录卫生由 main() 开头的 prepare_heatmap_dir 负责，这里不重复检查。
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    enc = model.encode_anchors(anchors)

    # 跨病例采样：每个病例取一个切片，避免全采到同一病例
    picked, seen = [], set()
    for f in files:
        case = os.path.basename(f)[:3]
        if case not in seen:
            picked.append(f)
            seen.add(case)
        if len(picked) >= n:
            break

    for f in picked:
        d = np.load(f)
        x = d["image"].astype(np.float32)       # (224,224) [0,1]
        mask224 = d["mask224"].astype(np.float32)

        gray = Image.fromarray((x * 255).astype(np.uint8))
        img = _gray2clip(x).unsqueeze(0).to(device)
        out = model(img, enc)
        amap = out["anomaly_map"][0].cpu().numpy()
        score = out["cls_probs"][0].item()
        gt = Image.fromarray((mask224 * 255).astype(np.uint8))

        fn = os.path.basename(f).replace(".npz", "")
        overlay_heatmap_with_gt(gray, amap, score, gt,
                                f"{out_dir}/{fn}_score{score:.2f}.png", calib=calib)
        print(f"  {fn} amap[min/mean/max]={amap.min():+.3f}/{amap.mean():+.3f}/{amap.max():+.3f}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="加载 checkpoint，只评估+出图")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=None,
                        help="按优化步数对齐而不是 epoch（给了它就覆盖 --epochs）。小样本下必须"
                             "用：5 张时一个 epoch 只有 1 个 batch，15 epoch = 15 步，与全量的 "
                             "2685 步不可比 —— 对照实验按步数对齐才公平")

    # ---- 小样本（改动 17 第 ① 条）：支持集只从 train 划分抽，绝不碰 val/test ----
    parser.add_argument("--n-support", type=int, default=None,
                        help="从 train 划分按病例抽 n 张切片当支持集（默认 None=全量，旧行为逐字"
                             "不变）。开了它，--out / --heatmap-dir 用默认值时会自动加 _k{n} 后缀，"
                             "以免覆盖基线 ckpt 和 heatmaps_thymoma/")
    # ---- 层内适配器（改动 17 第 ② 条）----
    parser.add_argument("--inlayer", action="store_true",
                        help="适配器插进文本塔**内部**（Houlsby 式瓶颈残差，按器官/病灶分套），并把"
                             "输出端的 ResidualTextAdapter 压成恒等再冻住 —— 即「不要加在输出后面」")
    parser.add_argument("--inlayer-bottleneck", type=int, default=64,
                        help="层内适配器瓶颈维度（塔内 hidden=768）")
    parser.add_argument("--inlayer-layers", type=str, default="8,9,10,11",
                        help="插进哪些 Transformer 层（0 起，文本塔共 12 层）")
    parser.add_argument("--inlayer-positions", type=str, default="attn,ffn",
                        help="层内插入点：attn=attention 之后、ffn=FFN 之后（Houlsby 的两个点）")
    parser.add_argument("--seed", type=int, default=0,
                        help="模型初始化与 DataLoader 打乱用的种子。数据划分固定 seed=0 不随之变，"
                             "这样多个种子可比。")
    parser.add_argument("--out", type=str, default="thymoma_local.pt", help="checkpoint 输出路径")
    parser.add_argument("--heatmap-dir", type=str, default="heatmaps_thymoma",
                        help="热力图输出目录；多种子跑时换目录以免覆盖")
    parser.add_argument("--margin-lo", type=float, default=None,
                        help="文本分离损失下界；None = 单边 hinge（旧行为，docx 改动 7）")
    parser.add_argument("--margin-d", type=float, default=0.7,
                        help="三层差向量 d_l 的相似度上限（多样性项）")
    parser.add_argument("--w-div", type=float, default=0.0,
                        help="多样性项权重；0 = 关闭（旧行为）")
    parser.add_argument("--w-level", type=float, default=0.0,
                        help="逐层定位损失权重（docx 改动 11 的修正 a）；0 = 关闭（旧行为）。"
                             "给①+②配质量项，约束多样性别把单层推成'不一样地差'")

    # ---- 显示层（改动 8 第 (4)a）。默认走新口径：本次改动的目的就是让新提示词的图能被
    # 看懂，若默认退回旧口径，附录里那条下一步命令仍会产出不能用的灰图。显示层不进任何
    # 指标，所以「新 flag 默认旧行为」那条保护指标表可比性的约定在这里没有保护对象。----
    parser.add_argument("--legacy-display", action="store_true",
                        help="出图退回旧口径（固定尺度 AMAP_SCALE=0.3 + 双线性），供 A/B；"
                             "默认走全局分位数标定 + 引导滤波上采样")
    parser.add_argument("--calib-split", type=str, default="test",
                        choices=["test", "train", "all"],
                        help="拟合出图标定分位数的参考集；test=固定的那 375 张"
                             "（划分 seed=0 钉死，每次同一批）")
    parser.add_argument("--calib-lo-pct", type=float, default=50.0,
                        help="标定下界分位数；P50 → 色阶 0（正常组织不发光），"
                             "调小=更多像素为 0、整体更暗")
    parser.add_argument("--calib-hi-pct", type=float, default=99.5,
                        help="标定上界分位数；调小=更热、红区更大，99.9=更保守")
    parser.add_argument("--calib-max-slices", type=int, default=None,
                        help="参考集最多用多少张切片拟合（默认全部；只影响标定，不影响指标）")
    parser.add_argument("--clean-heatmap-dir", action="store_true",
                        help="出图前清空 --heatmap-dir 里的 png；不加则目录非空时报错退出")
    parser.add_argument("--prompt-set", type=str, default="sentence",
                        choices=sorted(PROMPT_SETS),
                        help="提示词版本；sentence=改动 7 的整句版（基线 ckpt 用它训的），"
                             "short=短名词短语版（cos(d)=0.031，配合 ① 关 ② 使用）")
    args = parser.parse_args()
    # 小样本下默认输出名自动加后缀。--out 的默认值会直接覆盖基线 ckpt，--heatmap-dir 的默认值
    # 会往 heatmaps_thymoma/ 里写同名 png（改动 8 附录记过这个坑）。只有"没显式传参"才改。
    if args.n_support:
        if args.out == "thymoma_local.pt":
            args.out = f"thymoma_local_k{args.n_support}.pt"
        if args.heatmap_dir == "heatmaps_thymoma":
            args.heatmap_dir = f"heatmaps_thymoma_k{args.n_support}"
    prepare_heatmap_dir(args.heatmap_dir, clean=args.clean_heatmap_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchors = PROMPT_SETS[args.prompt_set]
    # levels 必须跟提示词的层数一致：model 按 len(cfg.levels) 建融合权重，层数对不上会索引越界
    cfg = Config(device=str(device), epochs=args.epochs, batch_size=16, lr=1e-4,
                 levels=anchors.levels)

    inlayer = None
    if args.inlayer:
        inlayer = {
            "organs": [ORGAN], "organ": ORGAN,
            "bottleneck": args.inlayer_bottleneck,
            "layers": [int(x) for x in args.inlayer_layers.split(",") if x.strip()],
            "positions": tuple(p.strip() for p in args.inlayer_positions.split(",") if p.strip()),
        }
    model = TextSideAnomalyModel(cfg, inlayer=inlayer).to(device)
    if args.inlayer:
        # 「不要加在输出后面」：把输出端 adapter 压成恒等（λ_t=0）再冻住，塔外那一路彻底不参与，
        # 可训参数只剩层内适配器。λ_t 是可学习 Parameter，所以必须显式置零才是恒等。
        with torch.no_grad():
            model.text_adapter.lambda_t.data.zero_()
        for p in model.text_adapter.parameters():
            p.requires_grad = False
    print(f"[prompt] set={args.prompt_set} levels={anchors.levels} "
          f"adapter={'层内' if args.inlayer else '输出端'}")
    tok = pre_tokenize(model, anchors, device)
    ckpt = args.out

    if args.eval_only:
        if args.inlayer:
            # 老 ckpt（改动 7~16 产的）没有 inlayer_bank.* 键，严格加载会直接报 Missing key
            model.load_compat(ckpt, map_location=device)
        else:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"已加载 checkpoint -> {ckpt}")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # 无正常样本：只训练文本分离 + 局部对齐（+ 可选的三层多样性项）
        crit = TotalLoss(
            margin=cfg.margin, w_text=1.0, w_global=0.0, w_local=1.0,
            margin_lo=args.margin_lo, margin_d=args.margin_d, w_div=args.w_div,
            w_level=args.w_level,
        )
        print(f"[loss] margin={cfg.margin} margin_lo={args.margin_lo} "
              f"w_div={args.w_div} margin_d={args.margin_d} w_level={args.w_level}")

        splits = make_slice_splits("thymoma_slices", seed=0)
        # 支持集只从 train 划分抽，按病例抽（同病例切片高度相关，混抽会把有效样本数虚高），
        # 并与 test 病例断言不交 —— 划分是 seed=0 钉死的，所以这个断言每次都会真的过一遍
        support = sample_support(splits["train"], args.n_support, seed=args.seed)
        if args.n_support:
            leak = case_ids(support) & case_ids(splits["test"])
            assert not leak, f"支持集与 test 病例重叠，会泄漏：{leak}"
        train_ds = ThymomaSliceDataset("thymoma_slices", files=support)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        n_epochs = cfg.epochs
        align = ""
        if args.steps:
            # 向上取整：batch 数除不尽就多跑一个 epoch，宁可多不可少（按步数对齐是硬约束）
            n_epochs = max(1, (args.steps + len(train_loader) - 1) // len(train_loader))
            align = f"（--steps {args.steps} 向上取整）"
        print(f"[thymoma-local] train_slices={len(train_ds)}"
              f"（{len(case_ids(support))} 个病例）device={device} "
              f"可训练参数={sum(p.numel() for p in trainable)}")
        print(f"[thymoma-local] 优化 {n_epochs} epoch × {len(train_loader)} batch "
              f"= {n_epochs * len(train_loader)} 步{align}")

        for epoch in range(n_epochs):
            model.train()
            total = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                labels = batch["label"].to(device)
                enc = encode_anchors_cached(model, tok)
                out = model(images, enc)
                loss = crit(enc, out, labels, masks)
                opt.zero_grad()
                loss["total"].backward()
                opt.step()
                total += loss["total"].item()
            print(f"[epoch {epoch + 1}/{n_epochs}] loss={total / len(train_loader):.4f}")

        torch.save(model.state_dict(), ckpt)
        print(f"已保存 checkpoint -> {ckpt}")

    splits = make_slice_splits("thymoma_slices", seed=0)
    test_ds = ThymomaSliceDataset("thymoma_slices", files=splits["test"])
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    m = evaluate(model, anchors, test_loader, device)
    print("\n===== 定位评估（test，patch 级 14×14）=====")
    print(f"  pixel AUROC={m['pixel_auroc']:.4f}  Dice={m['dice']:.4f}  IoU={m['iou']:.4f}")

    # 出图标定：纯显示层，放在 evaluate 之后，不动任何评估口径
    if args.legacy_display:
        calib = None
        print("[display] 旧口径：amap/AMAP_SCALE(0.3) + 双线性（改动 8 之前）")
    else:
        calib = fit_calibration(model, anchors, splits, device, split=args.calib_split,
                                batch_size=cfg.batch_size, lo_pct=args.calib_lo_pct,
                                hi_pct=args.calib_hi_pct, max_slices=args.calib_max_slices)

    save_heatmaps(model, anchors, splits["test"], device, args.heatmap_dir, n=6, calib=calib)
    print(f"热力图已生成到 {args.heatmap_dir}/（绿色=GT 肿瘤轮廓，红色=预测热区）")


if __name__ == "__main__":
    main()
