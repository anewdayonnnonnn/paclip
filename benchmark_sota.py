"""AD SOTA 基线对比：PatchCore / PaDiM / (可选 FastFlow)。

用 anomalib 在 PneumoniaMNIST(224) 上跑 one-class AD 基线，输出图像级 AUROC。
数据目录结构：
    data_pneumonia/
        train/normal/*.png      （one-class 训练集：正常）
        test/normal/*.png       （测试：正常）
        test/abnormal/*.png     （测试：异常）
"""

import sys

import torch


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    datamodule = Folder(
        name="pneumonia",
        root="D:/ad_data/data_pneumonia",
        normal_dir="train/normal",
        abnormal_dir="test/abnormal",
        normal_test_dir="test/normal",
    )

    models = {
        "PatchCore": Patchcore(),
        "PaDiM": Padim(),
    }

    results_table = {}
    for name, model in models.items():
        print(f"\n===== 运行 {name} =====")
        engine = Engine(max_epochs=1, devices=1, accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        # 提取图像级 AUROC（anomalib 2.x 返回 metric dict / list）
        auroc = _extract_auroc(test_results)
        results_table[name] = auroc
        print(f"{name} 图像级 AUROC = {auroc}")

    print("\n===== AD SOTA 对比（PneumoniaMNIST，one-class）=====")
    for name, a in results_table.items():
        print(f"  {name}: AUROC={a}")


def _extract_auroc(test_results):
    """从 anomalib Engine.test 的返回里提取图像级 AUROC。"""
    # test_results 可能是 dict、list[dict]、或 MetricCollection
    if isinstance(test_results, dict):
        d = test_results
    elif isinstance(test_results, (list, tuple)) and test_results:
        d = test_results[0] if isinstance(test_results[0], dict) else test_results
    else:
        d = test_results

    # 常见键：image_AUROC / Image_AUROC / image_auroc
    if hasattr(d, "items"):
        for k, v in d.items():
            kl = str(k).lower()
            if "image" in kl and "auroc" in kl:
                return float(torch.tensor(v))
    # 兜底：直接打印
    print("  [debug] test_results 结构:", type(test_results), str(test_results)[:300])
    return float("nan")


if __name__ == "__main__":
    main()
