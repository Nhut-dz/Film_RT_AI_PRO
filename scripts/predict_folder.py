"""Chạy pipeline cho cả thư mục ảnh (không cần server), xuất ảnh vẽ box + JSON + 1 file CSV tổng.

  python scripts/predict_folder.py --input D:/film_full --output outputs/run1 --detector YOLO26
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weld_rt.config import load_settings  # noqa: E402
from weld_rt.pipeline import Analyzer  # noqa: E402
from weld_rt.preprocessing import load_image  # noqa: E402
from weld_rt.visualize import render  # noqa: E402

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def flatten(res) -> list[dict]:
    """Một dòng CSV cho mỗi kết quả: vùng bố cục, dòng OCR, khuyết tật."""
    rows = []

    def add(kind, label, d, **kw):
        rows.append(dict(image=res.image, kind=kind, label=label, conf=d.conf, conf_pct=d.conf_pct,
                         x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3], **kw))
    for name, regs in (('film_roi', res.film_roi), ('weld_info', res.weld_info)):
        for i, r in enumerate(regs):
            add(name, name, r, region_id=i)
            if r.ocr:
                for ln in r.ocr.lines:
                    add('ocr_line', ln.text, ln, region_id=i)
    for d in res.defects:
        add('defect', d.cls, d, resnet_top=d.resnet_top, resnet_top_pct=d.resnet_top_pct)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--detector', default=None, help='YOLO26 | Faster R-CNN (mặc định theo config)')
    ap.add_argument('--conf-min', type=float, default=None)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    an = Analyzer(load_settings())
    rows = []
    for p in sorted(x for x in a.input.iterdir() if x.suffix.lower() in EXT):
        im = load_image(str(p))
        res = an.analyze(im, p.name, a.detector, a.conf_min)
        (a.output / f'{p.stem}.json').write_text(res.model_dump_json(indent=2), encoding='utf-8')
        render(im, res).save(a.output / f'{p.stem}_annotated.jpg', quality=90)
        rows += flatten(res)
        print(f'{p.name}: {len(res.defects)} khuyết tật, {res.timing_ms["total"]:.0f} ms')
    pd.DataFrame(rows).to_csv(a.output / 'results.csv', index=False, encoding='utf-8-sig')
    print('Kết quả ở', a.output)


if __name__ == '__main__':
    main()
