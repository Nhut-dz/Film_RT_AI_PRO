"""Vẽ kết quả lên ảnh: nhãn dạng 'P 85.3%'."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .schemas import AnalysisResult

DEFECT_COLORS = {'BT': '#2a78d6', 'Debris': '#eb6834', 'ER': '#1baf7a', 'P': '#eda100', 'Ti': '#e87ba4', 'Other': '#6250d6'}
LAYOUT_COLORS = {'film_roi': '#0b84a5', 'weld_info': '#c0392b'}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)        # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(im: Image.Image, items) -> Image.Image:
    """items = [(xyxy, nhãn, màu)]. Vẽ box + nhãn có nền màu, nhãn đặt trên box (hoặc dưới nếu sát mép)."""
    d = ImageDraw.Draw(im)
    fs = max(12, int(max(im.size) / 70))
    font, w = _font(fs), max(2, fs // 6)
    for b, label, color in items:
        b = [float(v) for v in b]
        d.rectangle(b, outline=color, width=w)
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 6
        y = b[1] - th if b[1] - th >= 0 else b[3]
        d.rectangle((b[0], y, b[0] + tw, y + th), fill=color)
        d.text((b[0] + 3, y + 3 - tb[1]), label, fill='white', font=font)
    return im


def render(im: Image.Image, res: AnalysisResult) -> Image.Image:
    items = [(r.bbox, f'{name} {r.conf_pct}', LAYOUT_COLORS[name])
             for name, regs in (('film_roi', res.film_roi), ('weld_info', res.weld_info)) for r in regs]
    items += [(d.bbox, f'{d.cls} {d.conf_pct}', DEFECT_COLORS.get(d.cls, '#ff0000'))
              for d in sorted(res.defects, key=lambda d: d.conf)]          # box tin cậy cao vẽ sau, nằm trên
    return draw_boxes(im.copy(), items)
