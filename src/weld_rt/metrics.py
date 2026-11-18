"""Hàm đánh giá detection, port nguyên văn từ notebook (AP kiểu COCO 101 điểm, IoU 0.50:0.95, ghép tham lam).
Dùng để kiểm chứng source mới cho cùng mAP với notebook."""
from __future__ import annotations

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
EMPTY = (np.zeros(0, int), np.zeros(0), np.zeros((0, 4)))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_class(gt, preds, c, thr):
    confs, tps, npos = [], [], 0
    for img, (gc, gb) in gt.items():
        g = gb[gc == c]; npos += len(g)
        pc, ps, pb = preds.get(img, EMPTY)
        m = pc == c; s, b = ps[m], pb[m]
        order = np.argsort(-s); s, b = s[order], b[order]
        used = np.zeros(len(g), bool); ious = iou_matrix(b, g)
        for i in range(len(s)):
            tp = 0
            if len(g):
                cand = np.where(~used & (ious[i] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]; used[j] = True; tp = 1
            confs.append(s[i]); tps.append(tp)
    confs, tps = np.array(confs), np.array(tps)
    order = np.argsort(-confs, kind='stable')
    return confs[order], tps[order], npos


def ap_101(tp, npos):
    if npos == 0:
        return np.nan
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp); rec = ctp / npos; prec = ctp / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)]))


def prf_at(conf, tp, npos, thr):
    k = int((conf >= thr).sum()); t = int(tp[:k].sum())
    p = t / k if k else 0.0; r = t / npos if npos else np.nan
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_detector(gt, preds, class_names, main_ids, conf_thr):
    """Trả về (bảng theo class, tổng hợp 5 class chính) tại ngưỡng conf_thr cho P/R/F1."""
    rows = []
    for c, name in enumerate(class_names):
        per = {t: match_class(gt, preds, c, t) for t in IOU_THRS}
        conf, tp, npos = per[0.5]
        p, r, f = prf_at(conf, tp, npos, conf_thr)
        rows.append(dict(cls=name, gt=npos, AP50=ap_101(tp, npos),
                         AP50_95=np.nanmean([ap_101(per[t][1], npos) for t in IOU_THRS]) if npos else np.nan, P=p, R=r, F1=f))
    main = [rows[c] for c in main_ids]
    summary = dict(conf_thr=conf_thr, mAP50=np.nanmean([r['AP50'] for r in main]), mAP50_95=np.nanmean([r['AP50_95'] for r in main]),
                   P=np.mean([r['P'] for r in main]), R=np.nanmean([r['R'] for r in main]), F1=np.mean([r['F1'] for r in main]))
    return rows, summary
