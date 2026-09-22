"""三层提示词扩展，形成多属性文本锚点（brain MRI）。

文档原文：
    将一对 normal/abnormal 提示扩展为三层或者多层提示词，形成多属性文本锚点。
    如原来是：
        a normal brain MRI
        an abnormal brain MRI
    现在可以在三个层级上进行扩展。

    三层提示词各自独立地与图像特征做匹配，输出各自的判别结果，最后再融合。

三层（从粗粒度到细粒度）示例：
    Level 1（器官/组织级）:
        normal:   "a normal brain MRI"
        abnormal: "an abnormal brain MRI"
    Level 2（结构/病灶级）:
        normal:   "a normal brain MRI with intact anatomy and clear ventricles"
        abnormal: "an abnormal brain MRI with a focal lesion"
    Level 3（纹理/密度级）:
        normal:   "a normal brain MRI with homogeneous tissue texture and sharp boundaries"
        abnormal: "an abnormal brain MRI with irregular texture and blurred boundaries"
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class ThreeLevelPrompts:
    """三层提示词模板，每层 normal / abnormal 各一组（可多条，做均值池化）。"""

    normal: Dict[str, List[str]]
    abnormal: Dict[str, List[str]]

    @property
    def levels(self) -> List[str]:
        return list(self.normal.keys())

    def all_texts(self) -> List[str]:
        texts: List[str] = []
        for lvl in self.levels:
            texts.extend(self.normal[lvl])
            texts.extend(self.abnormal[lvl])
        return texts


DEFAULT_BRAIN_MRI_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal brain MRI"],
        "2": ["a normal brain MRI with intact anatomy and clear ventricles"],
        "3": ["a normal brain MRI with homogeneous tissue texture and sharp boundaries"],
    },
    abnormal={
        "1": ["an abnormal brain MRI"],
        "2": ["an abnormal brain MRI with a focal lesion"],
        "3": ["an abnormal brain MRI with irregular texture and blurred boundaries"],
    },
)


def build_text_anchors(
    texts: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """tokenize 字符串列表，返回 input_ids / attention_mask（CPU 长整型）。"""
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
