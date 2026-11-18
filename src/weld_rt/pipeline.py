"""Pipeline suy luận: ảnh film -> bố cục -> OCR weld_info -> khuyết tật trong film_roi -> softmax ResNet50.

Mọi model được load MỘT lần khi khởi tạo Analyzer (FastAPI khởi tạo lúc server start).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image

from .config import Settings
from .models import FasterRCNNDetector, LayoutDetector, ResNetClassifier, YoloDetector
from .ocr import load_engine, read_weld_info
from .preprocessing import pct
from .schemas import AnalysisResult, Defect, OcrResult, Region

log = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


class Analyzer:
    def __init__(self, cfg: Settings, load_ocr: bool = True):
        self.cfg, self.device = cfg, pick_device(cfg.device)
        nc, ic = cfg.num_classes, cfg.inference
        t0 = time.perf_counter()
        self.detectors = {}
        for cls in (YoloDetector, FasterRCNNDetector):
            try:
                det = (cls(cfg.yolo, self.device, ic.max_det) if cls is YoloDetector
                       else cls(cfg.frcnn, nc, self.device, ic.max_det))
                self.detectors[det.name] = det
            except FileNotFoundError as e:
                log.warning('%s', e)
        if not self.detectors:
            raise RuntimeError('Không có detector nào: chép yolo26_best.pt hoặc faster_rcnn_best.pt vào thư mục weights/')
        self.classifier = ResNetClassifier(cfg.resnet, cfg.crop, nc, self.device) if cfg.resnet.weights.exists() else None
        self.layout = LayoutDetector(cfg.layout, self.device, ic.nms_iou) if cfg.layout.weights.exists() else None
        self.ocr = load_engine(cfg.ocr) if (load_ocr and self.layout is not None) else None
        self._lock = threading.Lock()          # CPU: chạy tuần tự từng request để không tranh RAM/luồng
        self.load_seconds = time.perf_counter() - t0
        log.info('Đã load model trong %.1fs: %s', self.load_seconds, self.info())

    # ------------------------------------------------------------------ thông tin
    def info(self) -> dict:
        return dict(device=str(self.device), detectors=list(self.detectors), default_detector=self.default_detector,
                    classifier=self.classifier is not None, layout=self.layout is not None,
                    ocr=getattr(self.ocr, 'name', None), class_names=self.cfg.class_names,
                    conf_min=self.cfg.inference.conf_min, config_sources=self.cfg.source)

    @property
    def default_detector(self) -> str:
        d = self.cfg.inference.default_detector
        return d if d in self.detectors else next(iter(self.detectors))

    # ------------------------------------------------------------------ các bước
    def detect_in_rois(self, im: Image.Image, rois: list[list[float]], detector: str, conf: float):
        """Chạy detector trên từng vùng, đổi toạ độ về ảnh gốc, NMS giữa các vùng chồng nhau."""
        crops = [im.crop(tuple(int(round(v)) for v in r)) for r in rois]
        C, S, B = [], [], []
        for r, (c, s, b) in zip(rois, self.detectors[detector].predict(crops, conf)):
            off = np.array([int(round(r[0])), int(round(r[1]))] * 2, np.float32)
            C.append(c), S.append(s), B.append(b + off)
        C, S, B = np.concatenate(C), np.concatenate(S), np.concatenate(B).reshape(-1, 4)
        if len(rois) > 1 and len(S):
            k = ops.batched_nms(torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(S, dtype=torch.float32),
                                torch.as_tensor(C), self.cfg.inference.nms_iou).numpy()
            C, S, B = C[k], S[k], B[k]
        return C, S, B

    def _defects(self, im, C, S, B) -> list[Defect]:
        probs = self.classifier.softmax_boxes(im, B) if self.classifier is not None else [None] * len(S)
        names, out = self.cfg.class_names, []
        for c, s, b, p in sorted(zip(C, S, B, probs), key=lambda t: -t[1]):
            d = Defect(cls=names[int(c)], conf=round(float(s), 4), conf_pct=pct(s), bbox=[round(float(v), 1) for v in b])
            if p is not None:
                top = int(np.argmax(p))
                d.resnet_top, d.resnet_top_pct = names[top], pct(p[top])
                d.resnet_probs = {n: round(float(v), 4) for n, v in zip(names, p)}
            out.append(d)
        return out

    # ------------------------------------------------------------------ API chính
    def analyze(self, im: Image.Image, name: str = 'image', detector: str | None = None, conf_min: float | None = None) -> AnalysisResult:
        detector = detector or self.default_detector
        if detector not in self.detectors:
            raise ValueError(f'Detector không hợp lệ: {detector}. Có: {list(self.detectors)}')
        floor = self.cfg.inference.conf_min
        conf = max(floor, conf_min if conf_min is not None else floor)      # không bao giờ xuống dưới ngưỡng đã chốt
        W, H = im.size
        timing, notes = {}, []
        with self._lock:
            t = time.perf_counter()
            layout = self.layout.predict(im, conf) if self.layout else {'film_roi': [], 'weld_info': []}
            timing['layout'] = (time.perf_counter() - t) * 1000
            if self.layout is None:
                notes.append('Chưa có detector bố cục: phân tích khuyết tật trên toàn ảnh, không OCR.')
            film = [Region(**r) for r in layout['film_roi']]
            weld = [Region(**r) for r in layout['weld_info']]

            t = time.perf_counter()
            for wi in weld:
                if self.ocr is not None:
                    wi.ocr = OcrResult(**read_weld_info(self.ocr, self.cfg.ocr, conf, im, wi.bbox,
                                                        exclude_boxes=[(f.bbox, f.conf) for f in film], box_conf=wi.conf))
            if weld and self.ocr is None:
                notes.append('Chưa có PaddleOCR: không đọc chữ vùng weld_info.')
            timing['ocr'] = (time.perf_counter() - t) * 1000

            if self.layout is not None and not film:
                notes.append(f'Không tìm thấy film_roi >= {conf:.0%}: phân tích khuyết tật trên toàn ảnh.')
            t = time.perf_counter()
            C, S, B = self.detect_in_rois(im, [f.bbox for f in film] or [[0, 0, W, H]], detector, conf)
            timing['detector'] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            defects = self._defects(im, C, S, B)
            timing['resnet50'] = (time.perf_counter() - t) * 1000
        timing = {k: round(v, 1) for k, v in timing.items()}
        timing['total'] = round(sum(timing.values()), 1)
        return AnalysisResult(image=Path(name).name, width=W, height=H, conf_min=conf, defect_detector=detector,
                              roi_source='film_roi' if film else 'full_image', film_roi=film, weld_info=weld,
                              defects=defects, notes=notes, timing_ms=timing)
