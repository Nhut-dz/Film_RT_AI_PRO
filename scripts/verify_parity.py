"""Kiểm chứng source mới cho ra ĐÚNG kết quả của notebook, không cần train lại.

Tái tạo tập test của notebook từ split_manifest.csv + dataset v9 gốc, chạy 2 detector bằng source mới
(conf 0.001 như notebook), tính lại mAP và so với det_test_summary_final.csv. Thêm --pred-conf để so từng box
với pred_conf_test_final.csv (mục 8.6).

Ví dụ:
  python scripts/verify_parity.py --dataset D:/data/v9 --manifest results/split_manifest.csv \
      --expected results/det_test_summary_final.csv --pred-conf results/pred_conf_test_final.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weld_rt.config import load_settings  # noqa: E402
from weld_rt.metrics import evaluate_detector, iou_matrix  # noqa: E402
from weld_rt.models import FasterRCNNDetector, ResNetClassifier, YoloDetector  # noqa: E402
from weld_rt.pipeline import pick_device  # noqa: E402
from weld_rt.preprocessing import load_image  # noqa: E402

# Map 19 class gốc -> 6 class (giai đoạn 4 của notebook): BT, Debris, ER, P, Ti giữ; 9 khuyết tật khác -> Other; còn lại bỏ
OLD2NEW = {0: 0, 3: 1, 4: 2, 14: 3, 15: 4, **{c: 5 for c in (1, 2, 5, 6, 7, 8, 9, 10, 16)}}


def find_split_dirs(root: Path) -> Path:
    """Chấp nhận cả thư mục v9 lồng nhau (v9/v9/test/...)."""
    hits = sorted(p.parent for p in root.rglob('data.yaml') if (p.parent / 'test' / 'images').is_dir())
    if not hits:
        raise SystemExit(f'Không thấy dataset (data.yaml + test/images) trong {root}')
    return hits[0]


def load_test_gt(root: Path, manifest: Path):
    m = pd.read_csv(manifest)
    t = m[m.new_split == 'test']
    gt, paths = {}, {}
    for r in t.itertuples():
        img = root / r.orig_split / 'images' / r.file
        W, H = load_image(str(img)).size
        rows = [l.split() for l in (root / r.orig_split / 'labels' / (img.stem + '.txt')).read_text(encoding='utf-8').splitlines() if l.strip()]
        rows = [[OLD2NEW[int(v[0])]] + [float(x) for x in v[1:5]] for v in rows if int(v[0]) in OLD2NEW]
        a = np.array(rows, float).reshape(-1, 5)
        b = np.stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                      (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H], 1) if len(a) else np.zeros((0, 4))
        gt[r.file] = (a[:, 0].astype(int), b)
        paths[r.file] = img
    return gt, paths


def compare_boxes(ours: dict, nb: pd.DataFrame, names: list[str], conf_min: float):
    """Ghép box của source mới với box notebook (cùng ảnh, cùng class, IoU lớn nhất)."""
    ious, dconf, missing, extra = [], [], 0, 0
    for img, g in nb.groupby('image'):
        pc, ps, pb = ours.get(img, (np.zeros(0, int), np.zeros(0), np.zeros((0, 4))))
        m = ps >= conf_min; pc, ps, pb = pc[m], ps[m], pb[m]
        used = np.zeros(len(ps), bool)
        for r in g.itertuples():
            cand = np.where((pc == names.index(r.cls)) & ~used)[0]
            if not len(cand):
                missing += 1; continue
            iou = iou_matrix(np.array([[r.x1, r.y1, r.x2, r.y2]]), pb[cand])[0]
            j = cand[int(np.argmax(iou))]; used[j] = True
            ious.append(iou.max()); dconf.append(abs(ps[j] - r.conf))
        extra += int((~used).sum())
    return dict(matched=len(ious), missing=missing, extra=extra,
                min_iou=float(np.min(ious)) if ious else np.nan, max_conf_diff=float(np.max(dconf)) if dconf else np.nan)


def check_resnet(cfg, dev, gt, paths, expected, tol) -> bool:
    """ResNet50 trên crop GT của test (mục 8.3): macro-F1 5 class chính và accuracy trên mọi box."""
    from sklearn.metrics import f1_score
    clf = ResNetClassifier(cfg.resnet, cfg.crop, cfg.num_classes, dev)
    y, p = [], []
    for f, (gc, gb) in gt.items():
        if len(gc):
            y += gc.tolist(); p += clf.softmax_boxes(load_image(str(paths[f])), gb).argmax(1).tolist()
    y, p = np.array(y), np.array(p)
    f1 = f1_score(y, p, labels=cfg.main_class_ids, average='macro', zero_division=0); acc = float((y == p).mean())
    line = f'ResNet50      macro-F1(5 class)={f1:.4f} accuracy={acc:.4f} ({len(y)} crop)'
    good = True
    if expected is not None:
        e = expected.loc['ResNet50']; diff = max(abs(f1 - e.macroF1_5cls), abs(acc - e.accuracy_all)); good = diff <= tol
        line += f' | notebook macro-F1={e.macroF1_5cls:.4f} accuracy={e.accuracy_all:.4f} | lệch {diff:.4f} -> {"KHỚP" if good else "KHÔNG KHỚP"}'
    print(line)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, type=Path, help='thư mục dataset v9 gốc')
    ap.add_argument('--manifest', required=True, type=Path, help='results/split_manifest.csv của notebook')
    ap.add_argument('--expected', type=Path, help='results/det_test_summary_final.csv')
    ap.add_argument('--pred-conf', type=Path, help='results/pred_conf_test_final.csv (so từng box)')
    ap.add_argument('--expected-cls', type=Path, help='results/cls_test_summary_final.csv (kiểm ResNet50)')
    ap.add_argument('--tol', type=float, default=0.01, help='sai lệch mAP cho phép')
    a = ap.parse_args()

    cfg = load_settings(); dev = pick_device(cfg.device)
    gt, paths = load_test_gt(find_split_dirs(a.dataset), a.manifest)
    print(f'Tập test: {len(gt)} ảnh, {sum((g[0] < 5).sum() for g in gt.values())} box 5 class chính')
    expected = pd.read_csv(a.expected).set_index('model') if a.expected else None
    nb_boxes = pd.read_csv(a.pred_conf) if a.pred_conf else None
    ok = True
    for det in (YoloDetector(cfg.yolo, dev), FasterRCNNDetector(cfg.frcnn, cfg.num_classes, dev)):
        t0 = time.perf_counter()
        preds = {f: det.predict([load_image(str(p))], 0.001)[0] for f, p in paths.items()}
        thr = float(expected.loc[det.name, 'conf_thr']) if expected is not None else 0.5
        _, s = evaluate_detector(gt, preds, cfg.class_names, cfg.main_class_ids, thr)
        line = f'{det.name:13s} mAP50={s["mAP50"]:.4f} mAP50-95={s["mAP50_95"]:.4f} P={s["P"]:.3f} R={s["R"]:.3f} ({time.perf_counter() - t0:.0f}s)'
        if expected is not None:
            e = expected.loc[det.name]
            diff = max(abs(s['mAP50'] - e.mAP50), abs(s['mAP50_95'] - e.mAP50_95))
            good = diff <= a.tol; ok &= good
            line += f' | notebook mAP50={e.mAP50:.4f} mAP50-95={e.mAP50_95:.4f} | lệch {diff:.4f} -> {"KHỚP" if good else "KHÔNG KHỚP"}'
        print(line)
        if nb_boxes is not None:
            print('   so từng box với notebook:', compare_boxes(preds, nb_boxes[nb_boxes.detector == det.name], cfg.class_names,
                                                           cfg.inference.conf_min))
    if cfg.resnet.weights.exists():
        ok &= check_resnet(cfg, dev, gt, paths, pd.read_csv(a.expected_cls).set_index('model') if a.expected_cls else None, a.tol)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
