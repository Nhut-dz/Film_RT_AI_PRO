"""Đọc ảnh và crop ngữ cảnh, giữ đúng cách xử lý của notebook."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .config import CropCfg


def load_image(data: bytes | str) -> Image.Image:
    """Mở ảnh và đưa về RGB 8 bit. Ảnh 16 bit (TIFF film số hoá) được co giãn min-max về 0..255
    thay vì bị cắt ngưỡng như khi convert trực tiếp."""
    im = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    im.load()
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
        im = Image.fromarray(a.astype(np.uint8), 'L')
    return im.convert('RGB')


def context_crop(im: Image.Image, box, cc: CropCfg) -> Image.Image:
    """Crop vuông quanh box, cạnh = context x cạnh dài của box (tối thiểu min_side), đệm đen ngoài ảnh, resize về out.
    Giống hệt hàm context_crop ở giai đoạn 4.1 của notebook."""
    bx1, by1, bx2, by2 = box
    side = max(cc.min_side, cc.context * max(bx2 - bx1, by2 - by1))
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    crop = im.crop((int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(cx + side / 2)), int(round(cy + side / 2))))
    return crop.resize((cc.out, cc.out), Image.BILINEAR)


def pct(s: float) -> str:
    return f'{100 * float(s):.1f}%'
