import numpy as np
from PIL import Image

from weld_rt.config import CropCfg, OcrCfg, load_settings
from weld_rt.metrics import evaluate_detector
from weld_rt.ocr import read_weld_info, unrotate
from weld_rt.preprocessing import context_crop, load_image


def test_settings_load():
    cfg = load_settings()
    assert cfg.class_names == ['BT', 'Debris', 'ER', 'P', 'Ti', 'Other']
    assert cfg.inference.conf_min == 0.10


def test_context_crop_pads_black_outside_image():
    im = Image.new('L', (100, 100), 255)
    out = context_crop(im, (0, 0, 20, 20), CropCfg())
    assert out.size == (224, 224)
    a = np.asarray(out)
    assert a[0, 0] == 0 and a[-1, -1] == 255          # góc trên trái nằm ngoài ảnh -> đen


def test_load_image_16bit_is_rescaled():
    a = (np.arange(64 * 64).reshape(64, 64) * 10).astype(np.uint16)
    im = Image.fromarray(a)
    rgb = np.asarray(load_image_from(im))
    assert rgb.min() == 0 and rgb.max() == 255


def load_image_from(im):
    import io
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return load_image(buf.getvalue())


def test_unrotate_inverts_rot90():
    a = np.arange(12).reshape(3, 4)
    for k in range(4):
        r = np.rot90(a, k); yy, xx = np.mgrid[:r.shape[0], :r.shape[1]]
        o = unrotate(np.stack([xx.ravel(), yy.ravel()], 1), k, 4, 3).astype(int)
        assert (a[o[:, 1], o[:, 0]] == r.ravel()).all()


class FakeOcr:
    """Chỉ đọc được chữ khi ảnh đã xoay về hướng ngang (k=3); trả 2 dòng, 1 dòng nằm trong vùng film_roi."""
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        if w < h:                                  # còn dọc: không đọc được
            return []
        return [('B2000007PALU', 0.9, np.array([[5, 5], [w - 5, 5], [w - 5, 25], [5, 25]], float)),
                ('ASTM', 0.8, np.array([[5, h - 25], [w - 5, h - 25], [w - 5, h - 5], [5, h - 5]], float)),
                ('noise', 0.05, np.array([[5, 40], [20, 40], [20, 50], [5, 50]], float))]


def test_read_weld_info_rotation_exclusion_and_floor():
    im = Image.new('RGB', (400, 400))
    box = (10, 10, 110, 390)                       # vùng chữ dọc
    out = read_weld_info(FakeOcr(), OcrCfg(rotations=[0, 3]), 0.10, im, box,
                         exclude_boxes=[((0, 0, 400, 30), 0.9)], box_conf=0.8)
    assert out['rotation_ccw_deg'] == 270
    texts = [l['text'] for l in out['lines']]
    assert 'noise' not in texts and out['n_dropped_below_conf_min'] == 1     # < 10% bị bỏ
    assert len(out['lines']) + len(out['excluded_lines']) == 2


def test_metrics_perfect_prediction():
    gt = {'a': (np.array([0, 3]), np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float))}
    preds = {'a': (gt['a'][0], np.ones(2), gt['a'][1])}
    _, s = evaluate_detector(gt, preds, ['BT', 'Debris', 'ER', 'P', 'Ti', 'Other'], [0, 3], 0.5)
    assert abs(s['mAP50'] - 1) < 1e-9 and abs(s['mAP50_95'] - 1) < 1e-9
