"""把 heatmaps_final/ 里的对比图拼成一张总览。

坑：PIL 默认字体不含中文字形，直接画中文会变方块 —— 走 visualize.load_font()。
"""
import os
import sys

from PIL import Image

sys.path.insert(0, "D:/新建文件夹")
from text_side_anomaly.visualize import draw_labels, load_font

SRC = sys.argv[1] if len(sys.argv) > 1 else "D:/brain_dl/heatmaps_final"
DST = sys.argv[2] if len(sys.argv) > 2 else "D:/brain_dl/总览_6例对比.png"
GAP = 8

files = sorted(f for f in os.listdir(SRC) if f.endswith(".png"))
# 异常切片在前，正常切片在后
files.sort(key=lambda f: (f.startswith("normal"), f))
ims = [Image.open(os.path.join(SRC, f)) for f in files]
PW, PH = ims[0].size
NCOL = 4
PW //= NCOL                                   # 单面板宽

# 第三参数：逗号分隔的列标题；传 "-" 表示不加（每张子图自己已经带表头了）
_raw = sys.argv[3] if len(sys.argv) > 3 else "-"
COL_LABELS = [] if _raw == "-" else _raw.split(",")
HDR = 24 if COL_LABELS else 0     # 列标题行高
ROW = 20      # 每例名字行高

W = NCOL * (PW + GAP)
H = HDR + len(ims) * (ROW + PH)
canvas = Image.new("RGB", (W, H), (255, 255, 255))
if COL_LABELS:
    draw_labels(canvas, COL_LABELS, panel_w=PW)

font = load_font(13)
from PIL import ImageDraw
dr = ImageDraw.Draw(canvas)
for i, (im, f) in enumerate(zip(ims, files)):
    y = HDR + i * (ROW + PH)
    dr.text((4, y + 2), f.replace(".png", ""), fill=(0, 0, 0), font=font)
    canvas.paste(im, (0, y + ROW))
canvas.save(DST)
print(f"拼图 -> {DST}  ({canvas.size[0]}x{canvas.size[1]}, {len(ims)} 例)")
