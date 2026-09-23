"""异常图的可视化：稳健标定 + 引导滤波上采样。

**纯显示层，不参与任何指标计算。**
前面已实测：逐图 z-score / min-max 会破坏跨图可比性（Dice 0.5535 → 0.4012 / 0.3730），
所以标定只能用在出图上，绝不能回灌到评估口径里。

两件事：
1. 标定 —— 把 amap 映射到 [0,1]。默认用「在一批正常切片上拟合出的全局分位数」，
   保证正常切片不会因为逐图拉伸而凭空发亮（医学场景这点很重要）。
2. 上采样 —— 14×14 双线性放大到 224 必然糊。改用引导滤波（以原图为引导），
   让异常边界贴着解剖结构走，这是 paclip 的做法。
"""

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# 引导滤波
# ---------------------------------------------------------------------- #
def _box_filter(x: np.ndarray, r: int) -> np.ndarray:
    """积分图实现的均值滤波（O(1) per pixel，与半径无关）。"""
    h, w = x.shape
    cs = np.pad(np.cumsum(np.cumsum(x, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h - 1), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w - 1), np.clip(xs + r + 1, 0, w)
    s = (cs[np.ix_(y1, x1)] - cs[np.ix_(y0, x1)]
         - cs[np.ix_(y1, x0)] + cs[np.ix_(y0, x0)])
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return (s / area).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """引导滤波（He et al.）：以 guide 的边缘为准，对 src 做保边平滑。

    Args:
        guide: (H, W) 引导图，值域 [0, 1]（这里用原灰度图）。
        src: (H, W) 待滤波图，与 guide 同尺寸。
        radius: 滤波窗口半径。
        eps: 正则项，越大越平滑。
    """
    mi, mp = _box_filter(guide, radius), _box_filter(src, radius)
    mip = _box_filter(guide * src, radius)
    mii = _box_filter(guide * guide, radius)
    cov_ip = mip - mi * mp
    var_i = mii - mi * mi
    a = cov_ip / (var_i + eps)
    b = mp - a * mi
    return _box_filter(a, radius) * guide + _box_filter(b, radius)


def upsample_guided(amap: np.ndarray, guide: np.ndarray,
                    radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """14×14 异常图 → 引导滤波上采样到 guide 的尺寸。

    先双线性放大（补空间连续性），再用原图作引导做保边平滑（把边界贴回解剖结构）。
    """
    from PIL import Image

    h, w = guide.shape
    up = np.asarray(
        Image.fromarray(amap.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )
    g = np.asarray(guide, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    g = (g - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(g)
    return guided_filter(g, up, radius=radius, eps=eps)


# ---------------------------------------------------------------------- #
# 标定
# ---------------------------------------------------------------------- #
class AmapCalibration:
    """异常图 → [0, 1] 的稳健标定。

    用分位数而不是 min/max：局部极值不会再把整张图的色阶吃光。
    fit() 在正常切片上拟合即可，部署时不需要病灶标注。
    """

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        self.lo, self.hi = lo, hi

    @classmethod
    def fit_from_normals(cls, amaps: np.ndarray,
                         lo_pct: float = 50.0, hi_pct: float = 99.5) -> "AmapCalibration":
        """在一批正常切片（或任意参考图）的 amap 上标定。

        取 P50 作下界（正常组织对应色阶 0）、P99.5 作上界，留出对异常的高动态范围。
        """
        flat = np.asarray(amaps, dtype=np.float64).reshape(-1)
        lo = float(np.percentile(flat, lo_pct))
        hi = float(np.percentile(flat, hi_pct))
        return cls(lo, hi if hi - lo > 1e-6 else lo + 1e-6)

    @classmethod
    def fit_from_stats(cls, amaps: np.ndarray) -> "AmapCalibration":
        """无正常切片时：在混合分布上用分位数标定，同样稳健。"""
        flat = np.asarray(amaps, dtype=np.float64).reshape(-1)
        return cls(float(np.percentile(flat, 50)), float(np.percentile(flat, 99.5)))

    def __call__(self, amap: np.ndarray) -> np.ndarray:
        """amap → [0, 1]（截断，不逐图归一化）。"""
        return np.clip((np.asarray(amap, dtype=np.float32) - self.lo)
                       / (self.hi - self.lo), 0.0, 1.0)


# ---------------------------------------------------------------------- #
# 上色
# ---------------------------------------------------------------------- #
_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def load_font(size: int = 14):
    """加载一个含中文字形的字体。

    坑：PIL 的 ImageFont.load_default() 不含中文字形，直接画中文会全变成方块。
    """
    from PIL import ImageFont

    for p in _CJK_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_labels(canvas_img, labels, panel_w: int, header: int = 22, gap: int = 8,
                font_size: int = 14):
    """在拼图顶部画面板标题（支持中文）。canvas 尺寸需为 (len(labels)*(panel_w+gap), H+header)。"""
    from PIL import ImageDraw

    dr = ImageDraw.Draw(canvas_img)
    font = load_font(font_size)
    for j, lab in enumerate(labels):
        dr.text((j * (panel_w + gap) + 4, 3), lab, fill=(0, 0, 0), font=font)
    return canvas_img


def colorize(norm01: np.ndarray, gray: np.ndarray, gt: Optional[np.ndarray] = None,
             alpha: float = 0.5) -> np.ndarray:
    """[0,1] 的异常图叠到灰度原图上，可选画出 GT 轮廓（绿）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    h, w = gray.shape
    heat = (cm.jet(np.clip(norm01, 0, 1))[..., :3] * 255).astype(np.float32)
    g = np.stack([gray] * 3, axis=-1).astype(np.float32) * 255.0
    out = ((1 - alpha) * g + alpha * heat).astype(np.uint8)

    if gt is not None:
        from PIL import Image, ImageFilter
        e = np.asarray(Image.fromarray(gt.astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
                       dtype=np.float32)
        out[e > 30] = [0, 255, 0]
    return out


def render(amap: np.ndarray, gray: np.ndarray, calib: AmapCalibration,
           gt: Optional[np.ndarray] = None, guided: bool = True,
           radius: int = 4, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """一站式出图。返回 (叠加图, 归一化后的异常图)。

    顺序很关键：**先在 14×14 原生分辨率上标定到 [0,1]**（分位数就是在这个分辨率上
    测出来的），再上采样。掉过来的话，引导滤波会改变数值范围，标定就失准了。
    """
    from PIL import Image

    g = np.asarray(gray, dtype=np.float32)
    if g.max() > 1.0:
        g = g / 255.0

    norm01_low = calib(amap)                                   # (h, w) 原生分辨率
    if guided:
        up = upsample_guided(norm01_low, g, radius=radius, eps=eps)
    else:
        up = np.asarray(
            Image.fromarray(norm01_low.astype(np.float32), mode="F").resize(
                (g.shape[1], g.shape[0]), Image.BILINEAR), dtype=np.float32)
    up = np.clip(up, 0.0, 1.0)
    return colorize(up, g, gt), up
