"""FastAPI demo. Chạy: uvicorn api.main:app --host 127.0.0.1 --port 8000  rồi mở http://127.0.0.1:8000"""
from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from weld_rt.config import get_settings
from weld_rt.pipeline import Analyzer
from weld_rt.preprocessing import load_image
from weld_rt.visualize import render

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
UI_DIR = Path(__file__).resolve().parents[1] / 'ui'


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.analyzer = Analyzer(get_settings())          # load model 1 lần khi khởi động
    yield


app = FastAPI(title='Weld RT AI', version='1.0.0', lifespan=lifespan,
              description='Phát hiện khuyết tật trên phim RT mối hàn (YOLO26, Faster R-CNN, ResNet50) + vùng film và OCR.')


def analyzer(request: Request) -> Analyzer:
    return request.app.state.analyzer


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/models')
def models(request: Request):
    a = analyzer(request)
    return {**a.info(), 'load_seconds': round(a.load_seconds, 1)}


@app.post('/api/predict')
def predict(request: Request, file: UploadFile = File(...), detector: str | None = Form(None),
            conf_min: float | None = Form(None), return_image: bool = Form(True)):
    """Nhận 1 ảnh, trả JSON: vùng film_roi / weld_info (+ OCR), khuyết tật (class, % tin cậy, bbox, softmax ResNet50)
    và ảnh đã vẽ box (base64 JPEG). Hàm đồng bộ: FastAPI tự chạy trong threadpool nên không chặn event loop."""
    cfg = get_settings().api
    if file.content_type not in cfg.allowed_types:
        raise HTTPException(415, f'Định dạng không hỗ trợ: {file.content_type}. Chấp nhận: {cfg.allowed_types}')
    data = file.file.read(cfg.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f'Ảnh vượt quá {cfg.max_upload_mb} MB')
    try:
        im = load_image(data)
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f'Không đọc được ảnh: {e}') from e
    if conf_min is not None and not 0 <= conf_min <= 1:
        raise HTTPException(422, 'conf_min phải trong khoảng 0..1')
    try:
        res = analyzer(request).analyze(im, file.filename or 'image', detector, conf_min)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out = res.model_dump()
    if return_image:
        buf = io.BytesIO()
        render(im, res).save(buf, 'JPEG', quality=90)
        out['annotated_jpeg_base64'] = base64.b64encode(buf.getvalue()).decode()
    return JSONResponse(out)


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(UI_DIR / 'index.html')


app.mount('/ui', StaticFiles(directory=UI_DIR), name='ui')
