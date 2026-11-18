"""Detector khuyết tật: YOLO26 và Faster R-CNN ResNet50-FPN v2, dựng lại đúng kiến trúc lúc train trong notebook.

Mọi detector có chung giao diện: predict(images, conf) -> list[(cls, score, xyxy)] (numpy, toạ độ pixel ảnh đầu vào).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..config import FrcnnCfg, YoloCfg

Detections = tuple[np.ndarray, np.ndarray, np.ndarray]   # (cls int, score float, xyxy float (N, 4))


class Detector(Protocol):
    name: str

    def predict(self, images: list[Image.Image], conf: float) -> list[Detections]: ...


def _require(path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Thiếu file weights: {path}. Xem weights/README.md')


class YoloDetector:
    name = 'YOLO26'

    def __init__(self, cfg: YoloCfg, device: torch.device, max_det: int = 300):
        from ultralytics import YOLO
        _require(cfg.weights)
        self.model, self.imgsz, self.max_det = YOLO(str(cfg.weights)), cfg.imgsz, max_det
        self.device = 0 if device.type == 'cuda' else 'cpu'

    def predict(self, images: list[Image.Image], conf: float) -> list[Detections]:
        res = self.model.predict(images, imgsz=self.imgsz, conf=conf, max_det=self.max_det, verbose=False, device=self.device)
        return [(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()) for r in res]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())


def build_frcnn(cfg: FrcnnCfg, num_classes: int) -> torch.nn.Module:
    """Kiến trúc giống build_frcnn() ở giai đoạn 6: anchor 16-256, head num_classes + nền."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=cfg.min_size, max_size=cfg.max_size)
    sizes = tuple(tuple(s) for s in cfg.anchor_sizes)
    m.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(tuple(cfg.aspect_ratios),) * len(sizes))
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes + 1)
    m.roi_heads.score_thresh, m.roi_heads.detections_per_img = 0.001, 300
    return m


def load_state(path, device: torch.device) -> dict:
    """Checkpoint của notebook: {'model': state_dict, 'epoch': ..., ...}. File do chính mình train nên weights_only=False."""
    _require(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck['model'] if isinstance(ck, dict) and 'model' in ck else ck


class FasterRCNNDetector:
    name = 'Faster R-CNN'

    def __init__(self, cfg: FrcnnCfg, num_classes: int, device: torch.device, max_det: int = 300):
        self.device = device
        self.model = build_frcnn(cfg, num_classes)
        self.model.load_state_dict(load_state(cfg.weights, device))   # strict: lệch kiến trúc là báo lỗi ngay
        self.model.roi_heads.detections_per_img = max_det
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, images: list[Image.Image], conf: float) -> list[Detections]:
        outs = self.model([TF.to_tensor(im.convert('RGB')).to(self.device) for im in images])
        res = []
        for o in outs:
            c, s, b = o['labels'].cpu().numpy() - 1, o['scores'].cpu().numpy(), o['boxes'].cpu().numpy()   # 0 là nền
            m = s >= conf
            res.append((c[m], s[m], b[m]))
        return res

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
