"""把 BraTS2021 原始布局整理成 text_side_anomaly 需要的 normal/abnormal + masks 布局。

源:  /d/brain_dl/BraTS2021/{test/normal, test/tumor, test/annotation}
目标: /d/brain_dl/brain_data/{train,test}/{normal,abnormal}
      /d/brain_dl/brain_masks/abnormal/     （掩码与 abnormal 图片同名）

按**病例**划分 train/test，避免同一患者的切片两边都出现。
"""
import os
import random
import shutil
from collections import defaultdict

SRC = "D:/brain_dl/BraTS2021"
OUT = "D:/brain_dl/brain_data"
MASKS = "D:/brain_dl/brain_masks"
SEED = 0
TEST_RATIO = 0.3


def patient_of(name: str) -> str:
    """BraTS2021_01467_flair_13.png -> 01467"""
    return name.split("_")[1]


def main():
    for d in [f"{OUT}/train/normal", f"{OUT}/train/abnormal",
              f"{OUT}/test/normal", f"{OUT}/test/abnormal",
              f"{MASKS}/abnormal"]:
        os.makedirs(d, exist_ok=True)

    # 掩码索引：BraTS2021_01467_seg_13.png -> (01467, 13)
    seg = {}
    for f in os.listdir(f"{SRC}/test/annotation"):
        seg[f.replace("_seg_", "_flair_")] = f

    normal_files = sorted(os.listdir(f"{SRC}/test/normal"))
    tumor_files = sorted(os.listdir(f"{SRC}/test/tumor"))

    # 按病例聚合
    by_pat_n, by_pat_t = defaultdict(list), defaultdict(list)
    for f in normal_files:
        by_pat_n[patient_of(f)].append(f)
    for f in tumor_files:
        by_pat_t[patient_of(f)].append(f)

    # 两个类各自按病例划分，保证两类在两边都有
    rng = random.Random(SEED)

    def split(pats):
        pats = sorted(pats)
        rng.shuffle(pats)
        k = max(1, int(len(pats) * TEST_RATIO))
        return set(pats[:k]), set(pats[k:])

    te_n, tr_n = split(by_pat_n)
    te_t, tr_t = split(by_pat_t)

    n_copy = t_copy = m_copy = 0
    for pats, split_name, files_by_pat, is_tumor in [
        (tr_n, "train", by_pat_n, False), (te_n, "test", by_pat_n, False),
        (tr_t, "train", by_pat_t, True),  (te_t, "test", by_pat_t, True),
    ]:
        sub = "abnormal" if is_tumor else "normal"
        for p in pats:
            for f in files_by_pat[p]:
                shutil.copy2(f"{SRC}/test/{'tumor' if is_tumor else 'normal'}/{f}",
                             f"{OUT}/{split_name}/{sub}/{f}")
                if is_tumor:
                    n_copy += 1
                    if f in seg:
                        shutil.copy2(f"{SRC}/test/annotation/{seg[f]}", f"{MASKS}/abnormal/{f}")
                        m_copy += 1
                else:
                    t_copy += 1

    print(f"train: normal={len(os.listdir(f'{OUT}/train/normal'))} "
          f"abnormal={len(os.listdir(f'{OUT}/train/abnormal'))}")
    print(f"test : normal={len(os.listdir(f'{OUT}/test/normal'))} "
          f"abnormal={len(os.listdir(f'{OUT}/test/abnormal'))}")
    print(f"masks: {m_copy}  (abnormal 共 {n_copy}，缺 {n_copy - m_copy})")
    print(f"病例数: normal train/test = {len(tr_n)}/{len(te_n)}, "
          f"tumor train/test = {len(tr_t)}/{len(te_t)}")
    print(f"病例重叠检查: {set(tr_n) & set(te_n) or '无'} / {set(tr_t) & set(te_t) or '无'}")


if __name__ == "__main__":
    main()
