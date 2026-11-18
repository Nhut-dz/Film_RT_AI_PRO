"""Cấu trúc kết quả trả về (JSON). Toạ độ bbox = pixel ảnh gốc [x1, y1, x2, y2]."""
from __future__ import annotations

from pydantic import BaseModel


class Scored(BaseModel):
    conf: float           # 0..1
    conf_pct: str         # "85.3%"


class Defect(Scored):
    cls: str
    bbox: list[float]
    resnet_top: str | None = None
    resnet_top_pct: str | None = None
    resnet_probs: dict[str, float] | None = None   # softmax 6 class của ResNet50


class OcrLine(Scored):
    text: str             # nguyên văn OCR đọc được, không sửa
    bbox: list[float]


class OcrRow(Scored):
    text: str
    n_lines: int


class OcrResult(BaseModel):
    rotation_ccw_deg: int
    text_raw: str
    rows: list[OcrRow]
    lines: list[OcrLine]
    excluded_lines: list[OcrLine]        # chữ nằm trong film_roi (ASTM, IQI)
    n_dropped_below_conf_min: int


class Region(Scored):
    bbox: list[float]
    ocr: OcrResult | None = None


class AnalysisResult(BaseModel):
    image: str
    width: int
    height: int
    conf_min: float
    defect_detector: str
    roi_source: str                      # film_roi | full_image
    film_roi: list[Region]
    weld_info: list[Region]
    defects: list[Defect]
    notes: list[str]
    timing_ms: dict[str, float]
