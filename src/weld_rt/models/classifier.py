"""ResNet50 phân loại crop ngữ cảnh 6 class (giai đoạn 7 của notebook)."""
from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from ..config import CropCfg, ResnetCfg
from ..preprocessing import context_crop
from .detectors import load_state


class ResNetClassifier:
    name = 'ResNet50'

    def __init__(self, cfg: ResnetCfg, crop: CropCfg, num_classes: int, device: torch.device):
        self.device, self.crop = device, crop
        m = torchvision.models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(load_state(cfg.weights, device))
        self.model = m.to(device).eval()
        # Giống tf_eval: crop xám -> 3 kênh -> chuẩn hoá ImageNet
        self.tf = T.Compose([T.Resize((cfg.img, cfg.img)), T.Grayscale(3), T.ToTensor(), T.Normalize(cfg.mean, cfg.std)])

    @torch.no_grad()
    def softmax_crops(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.model.fc.out_features), np.float32)
        x = torch.stack([self.tf(c) for c in crops]).to(self.device)
        return self.model(x).softmax(1).cpu().numpy()

    def softmax_boxes(self, im: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """Softmax cho từng box của ảnh: crop từ ảnh xám, giống crop lúc train."""
        gray = im.convert('L')
        return self.softmax_crops([context_crop(gray, b, self.crop) for b in boxes])

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
