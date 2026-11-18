"""Detector bố cục YOLO26 (film_roi, weld_info) của giai đoạn 9. Tuỳ chọn: không có weights thì pipeline bỏ qua."""
from __future__ import annotations

import torch
import torchvision.ops as ops
from PIL import Image

from ..config import LayoutCfg
from ..preprocessing import pct


class LayoutDetector:
    def __init__(self, cfg: LayoutCfg, device: torch.device, nms_iou: float):
        from ultralytics import YOLO
        self.model, self.cfg, self.nms_iou = YOLO(str(cfg.weights)), cfg, nms_iou
        self.device = 0 if device.type == 'cuda' else 'cpu'

    def predict(self, im: Image.Image, conf: float) -> dict[str, list[dict]]:
        r = self.model.predict(im, imgsz=self.cfg.imgsz, conf=conf, verbose=False, device=self.device)[0]
        C, S, B = r.boxes.cls.cpu(), r.boxes.conf.cpu(), r.boxes.xyxy.cpu()
        keep = ops.batched_nms(B.float(), S.float(), C, self.nms_iou)     # bỏ box trùng cùng class (không lọc theo %)
        out = {n: [] for n in self.cfg.names}
        for c, s, b in zip(C[keep].numpy().astype(int), S[keep].numpy(), B[keep].numpy()):
            out[self.cfg.names[c]].append(dict(bbox=[round(float(v), 1) for v in b], conf=round(float(s), 4), conf_pct=pct(s)))
        return out
