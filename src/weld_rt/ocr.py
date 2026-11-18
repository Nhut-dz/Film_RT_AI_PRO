"""OCR vùng thông tin mối hàn (weld_info), port từ giai đoạn 9.3 của notebook.

- Thử nhiều hướng xoay, chọn hướng đọc ngang được nhiều ký tự tin cậy nhất (chữ chì thường xoay 90°, nhiều cột).
- Trả NGUYÊN VĂN từng dòng kèm % tin cậy, không sửa ký tự (O/0, I/1, B/8 giữ như OCR đọc).
- Dòng có tâm nằm trong film_roi (chữ ASTM, ký hiệu IQI) bị tách sang excluded_lines.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .config import OcrCfg
from .preprocessing import pct

log = logging.getLogger(__name__)


class PaddleEngine:
    """PaddleOCR 3.x huấn luyện sẵn. Tắt chỉnh hướng tài liệu vì ta tự thử các hướng xoay."""
    name = 'PaddleOCR'

    def __init__(self, lang: str):
        from paddleocr import PaddleOCR
        self.e = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False, text_rec_score_thresh=0.0)

    def __call__(self, rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        r = self.e.predict(np.ascontiguousarray(rgb[:, :, ::-1]))[0]          # PaddleOCR nhận BGR
        return [(t, float(s), np.asarray(p, float)) for t, s, p in zip(r['rec_texts'], r['rec_scores'], r['rec_polys'])]


def load_engine(cfg: OcrCfg):
    if not cfg.enabled:
        return None
    try:
        return PaddleEngine(cfg.lang)
    except Exception as e:                                  # thiếu thư viện hoặc không tải được model
        log.warning('Không khởi tạo được PaddleOCR (%r): tắt OCR', e)
        return None


def unrotate(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Đưa điểm từ ảnh np.rot90(a, k) về toạ độ ảnh gốc a có kích thước w x h."""
    x, y = pts[:, 0], pts[:, 1]
    k %= 4
    if k == 0:
        return np.stack([x, y], 1)
    if k == 1:
        return np.stack([w - 1 - y, x], 1)
    if k == 2:
        return np.stack([w - 1 - x, h - 1 - y], 1)
    return np.stack([y, h - 1 - x], 1)


def overlap_ratio(a, b) -> float:
    """Tỉ lệ diện tích box a nằm trong box b."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def group_rows(lines: list[dict], overlap: float) -> list[dict]:
    """Gom các dòng OCR (trong ảnh đã xoay về hướng đọc) thành hàng theo trục y, mỗi hàng sắp từ trái sang phải."""
    rows = []
    for ln in sorted(lines, key=lambda l: l['_y'][0]):
        r = rows[-1] if rows else None
        h = min(r['y2'] - r['y1'], ln['_y'][1] - ln['_y'][0]) if r else 0
        if r and min(r['y2'], ln['_y'][1]) - max(r['y1'], ln['_y'][0]) >= overlap * h:
            r['items'].append(ln)
            r['y2'] = max(r['y2'], ln['_y'][1])
        else:
            rows.append(dict(y1=ln['_y'][0], y2=ln['_y'][1], items=[ln]))
    out = []
    for r in rows:
        items = sorted(r['items'], key=lambda l: l['_x'])
        c = min(l['conf'] for l in items)
        out.append(dict(text=' '.join(l['text'] for l in items), conf=c, conf_pct=pct(c), n_lines=len(items)))
    return out


def read_weld_info(engine, cfg: OcrCfg, conf_min: float, im: Image.Image, box, exclude_boxes=(), box_conf: float = 1.0) -> dict:
    """OCR một vùng weld_info.
    exclude_boxes = [(bbox film_roi, conf)]. Nếu một film_roi chứa phần lớn chính vùng weld_info: film_roi tin cậy hơn thì
    giữ (weld_info nhiều khả năng là ASTM/IQI), ngược lại bỏ film_roi đó khỏi phép loại."""
    W, H = im.size
    x1, y1, x2, y2 = box
    p = cfg.pad * max(x2 - x1, y2 - y1)
    cx1, cy1, cx2, cy2 = int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(W, x2 + p)), int(min(H, y2 + p))
    crop = im.crop((cx1, cy1, cx2, cy2))
    scale = max(1.0, cfg.min_side / max(1, min(crop.size)))
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
    arr = np.asarray(crop.convert('RGB'))
    h, w = arr.shape[:2]
    best = None
    for k in cfg.rotations:
        lines = engine(np.rot90(arr, k))
        # chỉ tính dòng nằm ngang: OCR đọc được cả dòng dọc, phải ép về hướng đọc ngang thì gom hàng mới đúng
        score = sum(s * len(t.strip()) for t, s, q in lines if s >= conf_min and np.ptp(q[:, 0]) >= np.ptp(q[:, 1]))
        if best is None or score > best[0]:
            best = (score, k, lines)
    _, k, lines = best
    excluders = [b for b, s in exclude_boxes if overlap_ratio(box, b) < 0.5 or s > box_conf]
    kept, excluded, dropped = [], [], 0
    for t, s, poly in lines:
        if not t.strip():
            continue
        if s < conf_min:
            dropped += 1
            continue
        full = unrotate(poly, k, w, h) / scale + [cx1, cy1]                  # toạ độ ảnh gốc
        c = full.mean(0)
        ln = dict(text=t, conf=round(s, 4), conf_pct=pct(s), bbox=[round(float(v), 1) for v in (*full.min(0), *full.max(0))],
                  _x=poly[:, 0].mean(), _y=(poly[:, 1].min(), poly[:, 1].max()))
        inside = any(b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3] for b in excluders)
        (excluded if inside else kept).append(ln)
    rows = group_rows(kept, cfg.row_overlap)

    def strip(L):
        return [{kk: v for kk, v in l.items() if not kk.startswith('_')} for l in L]

    return dict(rotation_ccw_deg=90 * k, text_raw='\n'.join(r['text'] for r in rows), rows=rows,
                lines=strip(kept), excluded_lines=strip(excluded), n_dropped_below_conf_min=dropped)
