import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from weld_rt.preprocessing import pct
from weld_rt.schemas import AnalysisResult, Defect


class FakeAnalyzer:
    load_seconds = 0.0
    default_detector = 'YOLO26'

    def info(self):
        return dict(device='cpu', detectors=['YOLO26'], default_detector='YOLO26', classifier=False, layout=False,
                    ocr=None, class_names=['BT', 'Debris', 'ER', 'P', 'Ti', 'Other'], conf_min=0.1, config_sources=[])

    def analyze(self, im, name, detector, conf_min):
        if detector not in (None, 'YOLO26'):
            raise ValueError('Detector không hợp lệ')
        W, H = im.size
        return AnalysisResult(image=name, width=W, height=H, conf_min=0.1, defect_detector='YOLO26', roi_source='full_image',
                              film_roi=[], weld_info=[], notes=[], timing_ms={'total': 1.0},
                              defects=[Defect(cls='P', conf=0.853, conf_pct=pct(0.853), bbox=[1, 1, 10, 10])])


app.state.analyzer = FakeAnalyzer()
client = TestClient(app)                    # không dùng "with": bỏ qua lifespan, không load model thật


def _png():
    buf = io.BytesIO(); Image.new('RGB', (64, 48)).save(buf, 'PNG'); return buf.getvalue()


def test_models_and_ui():
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/api/models').json()['detectors'] == ['YOLO26']
    assert 'Weld RT AI' in client.get('/').text


def test_predict_ok():
    r = client.post('/api/predict', files={'file': ('a.png', _png(), 'image/png')}, data={'detector': 'YOLO26'})
    assert r.status_code == 200
    body = r.json()
    assert body['defects'][0]['conf_pct'] == '85.3%' and body['width'] == 64
    Image.open(io.BytesIO(base64.b64decode(body['annotated_jpeg_base64'])))


def test_predict_rejects_bad_input():
    assert client.post('/api/predict', files={'file': ('a.txt', b'hello', 'text/plain')}).status_code == 415
    assert client.post('/api/predict', files={'file': ('a.png', b'notimage', 'image/png')}).status_code == 400
    assert client.post('/api/predict', files={'file': ('a.png', _png(), 'image/png')}, data={'detector': 'X'}).status_code == 422
    assert client.post('/api/predict', files={'file': ('a.png', _png(), 'image/png')}, data={'conf_min': '2'}).status_code == 422
