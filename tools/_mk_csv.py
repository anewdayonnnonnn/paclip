"""从评测日志解析出 CSV —— 不手抄数字，避免转录错误。

用法（服务器）：python _mk_csv.py
"""
import csv
import re

BRAIN_LOG = "/root/work/brain_dl/eval_seeds.log"
THY_LOGS = [f"/root/work/thymoma/thymoma_seed{i}.log" for i in range(3)]

# ---------- 脑 ----------
row_re = re.compile(
    r"^(.+?)\s*\|\s*(\[[^\]]*\])\s*\|\s*"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*"
    r"([\d.]+)%\s*/\s*([\d.]+)%\s*\|\s*([\d.]+)%")
mean_re = re.compile(
    r"^(\S+)\s+多种子均值\s*\|\s*\|\s*"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")

rows = []
for line in open(BRAIN_LOG, encoding="utf-8"):
    line = line.rstrip("\n")
    m = row_re.match(line)
    if m:
        rows.append([m.group(1).strip(), m.group(2), m.group(3), m.group(4),
                     m.group(5), m.group(6), m.group(7), m.group(8), m.group(9)])
        continue
    m = mean_re.match(line)
    if m:
        rows.append([f"{m.group(1)} 多种子均值", "", m.group(2), m.group(3),
                     m.group(4), m.group(5), "", "", ""])

out = "/root/work/brain_dl/eval_seeds_metrics.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["配置", "ms_layers", "imgAUROC", "Dice", "IoU", "pxAUROC",
                "脑外红斑_异常切片(%)", "脑外红斑_正常切片(%)", "红灯真病灶率(%)"])
    w.writerows(rows)
print(f"[脑] {len(rows)} 行 -> {out}")

# ---------- 胸腺 ----------
thy_re = re.compile(r"pixel AUROC=([\d.]+)\s+Dice=([\d.]+)\s+IoU=([\d.]+)")
trows = []
for i, p in enumerate(THY_LOGS):
    txt = open(p, encoding="utf-8", errors="replace").read()
    m = thy_re.search(txt)
    if m:
        trows.append([f"seed{i}", m.group(1), m.group(2), m.group(3)])
    else:
        print(f"[胸腺] seed{i} 没解析到指标")

if trows:
    a = [float(r[1]) for r in trows]
    d = [float(r[2]) for r in trows]
    i_ = [float(r[3]) for r in trows]
    trows.append(["均值", f"{sum(a)/len(a):.4f}", f"{sum(d)/len(d):.4f}",
                  f"{sum(i_)/len(i_):.4f}"])
    trows.append(["极差", f"{max(a)-min(a):.4f}", f"{max(d)-min(d):.4f}",
                  f"{max(i_)-min(i_):.4f}"])

tout = "/root/work/thymoma/thymoma_seeds_metrics.csv"
with open(tout, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["配置", "pixelAUROC", "Dice", "IoU"])
    w.writerows(trows)
print(f"[胸腺] {len(trows)} 行 -> {tout}")
