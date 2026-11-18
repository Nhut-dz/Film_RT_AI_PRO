"""Đọc cấu hình từ configs/config.yaml (+ ghi đè từ weights/config_final.json của notebook nếu có)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class InferenceCfg(BaseModel):
    conf_min: float = Field(0.10, ge=0.0, le=1.0)
    max_det: int = 300
    nms_iou: float = 0.5
    default_detector: str = 'YOLO26'


class YoloCfg(BaseModel):
    weights: Path
    imgsz: int = 1280


class FrcnnCfg(BaseModel):
    weights: Path
    min_size: int = 720
    max_size: int = 1280
    anchor_sizes: list[list[int]] = [[16], [32], [64], [128], [256]]
    aspect_ratios: list[float] = [0.5, 1.0, 2.0]


class ResnetCfg(BaseModel):
    weights: Path
    img: int = 224
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]


class CropCfg(BaseModel):
    context: float = 2.0
    min_side: int = 64
    out: int = 224


class LayoutCfg(BaseModel):
    weights: Path
    imgsz: int = 1280
    names: list[str] = ['film_roi', 'weld_info']


class OcrCfg(BaseModel):
    enabled: bool = True
    lang: str = 'en'
    rotations: list[int] = [0, 3, 1, 2]
    pad: float = 0.04
    min_side: int = 96
    row_overlap: float = 0.5


class ApiCfg(BaseModel):
    max_upload_mb: int = 25
    allowed_types: list[str] = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']


class Settings(BaseModel):
    class_names: list[str]
    main_class_ids: list[int]
    device: str = 'auto'
    inference: InferenceCfg = InferenceCfg()
    yolo: YoloCfg
    frcnn: FrcnnCfg
    resnet: ResnetCfg
    crop: CropCfg = CropCfg()
    layout: LayoutCfg
    ocr: OcrCfg = OcrCfg()
    api: ApiCfg = ApiCfg()
    source: list[str] = []          # file cấu hình đã đọc, để hiển thị/kiểm tra

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def _apply_notebook_config(raw: dict, nb_cfg: dict) -> None:
    """Ghi đè tham số kiến trúc/tiền xử lý bằng CFG mà notebook đã dùng lúc train (config_final.json)."""
    cfg = nb_cfg.get('CFG', {})
    if 'yolo' in cfg:
        raw['yolo']['imgsz'] = cfg['yolo'].get('imgsz', raw['yolo']['imgsz'])
    if 'frcnn' in cfg:
        for k in ('min_size', 'max_size', 'anchor_sizes'):
            if k in cfg['frcnn']:
                raw['frcnn'][k] = cfg['frcnn'][k]
    if 'resnet' in cfg:
        raw['resnet']['img'] = cfg['resnet'].get('img', raw['resnet']['img'])
    if 'crop' in cfg:
        raw['crop'].update({k: v for k, v in cfg['crop'].items() if k in ('context', 'min_side', 'out')})
    if 'infer' in cfg:
        raw['inference']['conf_min'] = cfg['infer'].get('conf_min', raw['inference']['conf_min'])
    if 'layout' in cfg:
        raw['layout']['imgsz'] = cfg['layout'].get('imgsz', raw['layout']['imgsz'])
    if 'ocr' in cfg:
        raw['ocr'].update({k: v for k, v in cfg['ocr'].items() if k in OcrCfg.model_fields and k != 'enabled'})


def load_settings(path: str | Path | None = None) -> Settings:
    path = _resolve(Path(path or 'configs/config.yaml'))
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    sources = [str(path)]
    nb_json = ROOT / 'weights' / 'config_final.json'
    if nb_json.exists():
        _apply_notebook_config(raw, json.loads(nb_json.read_text(encoding='utf-8')))
        sources.append(str(nb_json))
    for sec in ('yolo', 'frcnn', 'resnet', 'layout'):
        raw[sec]['weights'] = _resolve(Path(raw[sec]['weights']))
    return Settings(**raw, source=sources)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
