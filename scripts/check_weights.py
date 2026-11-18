"""Kiểm tra các file trong weights/: đúng loại model, đúng số class, không bị đặt nhầm tên.

  python scripts/check_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weld_rt.config import load_settings  # noqa: E402


def kind_of(ck) -> str:
    if isinstance(ck, dict) and 'train_args' in ck:
        return 'yolo'
    sd = ck.get('model', ck) if isinstance(ck, dict) else {}
    keys = list(sd) if isinstance(sd, dict) else []
    if any(k.startswith('roi_heads.') for k in keys):
        return 'frcnn'
    if 'fc.weight' in keys:
        return 'resnet'
    return 'không rõ'


def main() -> int:
    cfg = load_settings()
    expect = {'yolo': cfg.yolo.weights, 'frcnn': cfg.frcnn.weights, 'resnet': cfg.resnet.weights, 'layout': cfg.layout.weights}
    bad = 0
    for role, path in expect.items():
        if not path.exists():
            print(f'[--] {role:7s} {path.name}: chưa có')
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        k = kind_of(ck)
        want = 'yolo' if role == 'layout' else role
        info = ''
        if k == 'yolo':
            names = getattr(ck.get('model'), 'names', {})
            info = f"class={list(names.values()) if isinstance(names, dict) else names}, epochs={ck['train_args'].get('epochs')}"
        elif k in ('frcnn', 'resnet'):
            sd = ck['model']
            n = sd['roi_heads.box_predictor.cls_score.weight'].shape[0] - 1 if k == 'frcnn' else sd['fc.weight'].shape[0]
            info = f'số class={n}, best epoch={ck.get("epoch", -1) + 1}'
        ok = k == want
        bad += not ok
        print(f'[{"OK" if ok else "SAI"}] {role:7s} {path.name}: là {k} ({info})' + ('' if ok else f'  <- cần file {want}, có thể bị đặt nhầm tên'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
