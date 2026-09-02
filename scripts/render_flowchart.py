# -*- coding: utf-8 -*-
"""Render the gold-system decision flowchart to a PNG using only Pillow.

No matplotlib/graphviz needed. Draws boxes + arrows for the 7-step decision flow
(load -> macro gate -> bull gate -> rhythm gate -> entry -> sizing -> hold -> exit)
and saves to reports/gold_system_flowchart.png. CJK is drawn with a fallback font
if a CJK TTF is found, else ASCII labels.
"""
import sys
sys.path.insert(0, r"C:\Users\张策\Documents\EA量化项目")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "gold_system_flowchart.png"
W, H = 1040, 980
BG = (250, 250, 250)
GREEN = (39, 124, 65)
RED = (176, 48, 48)
AMBER = (196, 128, 20)
BLUE = (40, 80, 160)
GRAY = (90, 90, 90)


def find_cjk_font():
    cands = [
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
    ]
    for c in cands:
        if Path(c).is_file():
            return c
    return None


CJK = find_cjk_font()


def font(sz):
    if CJK:
        return ImageFont.truetype(CJK, sz)
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def draw_box(d, xy, text, fill, edge, txtcolor=(255, 255, 255), size=16, w=170, h=48):
    x0, y0 = xy
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10, fill=fill, outline=edge, width=2)
    f = font(size)
    lines = text.split("\n") if "\n" in text else [text]
    line_h = size + 8
    total_text_h = len(lines) * line_h
    start_y = y0 + (h - total_text_h) / 2 + 2
    for li, ln in enumerate(lines):
        bbox = d.textbbox((0, 0), ln, font=f)
        tw = bbox[2] - bbox[0]
        d.text((x0 + (w - tw) / 2, start_y + li * line_h), ln, font=f, fill=txtcolor)
    return (x0 + w, y0 + h)


def arrow(d, p1, p2, label=None, color=GRAY, side="down"):
    d.line([p1, p2], fill=color, width=3)
    # arrowhead
    if side == "down":
        d.polygon([(p2[0] - 7, p2[1] - 10), (p2[0] + 7, p2[1] - 10), (p2[0], p2[1])], fill=color)
    else:
        d.polygon([(p2[0] - 7, p2[1] - 4), (p2[0] + 7, p2[1] - 4), (p2[0], p2[1] - 12)], fill=color)
    if label:
        f = font(14)
        d.text((p1[0] + 6, (p1[1] + p2[1]) / 2 - 8), label, font=f, fill=color)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
cx = W // 2

# Title
tf = font(24)
d.text((cx, 20), "伦敦金量化策略 · 决策流程图", font=tf, fill=(0, 0, 0), anchor="ma")
sf = font(15)
d.text((cx, 58), "(牛市单向版 + 行情节奏门 + 风险预算定仓)", font=sf, fill=GRAY, anchor="ma")

# Boxes: (label, fill, edge, y, color-for-text, w)
steps = []
y = 96
row_h = 66
col = 300


def box(text, fill, edge, x, y, txt=255, w=col):
    return draw_box(d, (x, y), text, fill, edge, (txt, txt, txt), size=14, w=w, h=54)


# 1 load
b = box("① 加载数据\n黄金日线 + 宏观(DXY/VIX/10Y)", BLUE, (20, 40, 80), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# 2 macro gate
b = box("② 宏观门\n宏观分偏空? → 降杠杆0.5x", AMBER, (90, 60, 10), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# 3 bull gate
b = box("③ 牛市门\nbull分 ≥ 0.55?", AMBER, (90, 60, 10), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# 4 rhythm gate
b = box("④ 节奏门\n行情=确认上升趋势?", AMBER, (90, 60, 10), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# side exit for gates
ex = cx + col + 30
d.line([(cx + col, y - row_h - 3), (ex, y - row_h - 3), (ex, y - 3)], fill=RED, width=3)
d.polygon([(ex - 8, y - 8), (ex + 8, y - 8), (ex, y - 3)], fill=RED)
d.text((ex + 6, y - 12), "空仓\n(不限时间)", font=font(14), fill=RED)

# 5 entry
b = box("⑤ 入场\n收阳 + 微观多头对齐", AMBER, (90, 60, 10), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# 6 sizing
b = box("⑥ 风险预算定仓\n手数=账户2%/止损距离", GREEN, (12, 60, 30), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# 7 hold
b = box("⑦ 持有做多(低杠杆)", GREEN, (12, 60, 30), cx - col // 2, y)
y += row_h + 6
arrow(d, (cx, y - row_h - 3), (cx, y - 3), "")

# exits block
exits = [
    ("止损 ATR×2.5", RED),
    ("止盈 2×止损", GREEN),
    ("时间止损>30根", BLUE),
    ("峰值回撤≥20%→熔断停机", RED),
]
yy = y + 2
for lbl, colr in exits:
    box(lbl, colr, (0, 0, 0), cx - 110, yy, w=220)
    arrow(d, (cx - 110, yy - row_h - 4), (cx - 110, yy - 3), "", color=colr)
    yy += row_h + 4

img.save(str(OUT))
print("saved:", OUT)
