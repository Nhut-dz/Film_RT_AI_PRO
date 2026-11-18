# Film RT AI PRO: demo phát hiện khuyết tật trên phim RT mối hàn

Bộ source suy luận + FastAPI + giao diện web, dùng **trực tiếp weights đã train** trong notebook `Weld_RT_AI.ipynb` (không train lại).

- **Phát hiện khuyết tật:** YOLO26 (mặc định) hoặc Faster R-CNN ResNet50-FPN v2. 6 class: `BT, Debris, ER, P, Ti, Other`.
- **Phân loại lại:** ResNet50 cho phân phối softmax trên từng box.
- **% tin cậy:** mọi kết quả đều hiển thị dạng `P 85.3%`. Ngưỡng tối thiểu 10%; box < 10% bị bỏ.
- **Mở rộng (giai đoạn 9):** detector bố cục `film_roi` / `weld_info` + PaddleOCR đọc thông tin mối hàn. Tự bật khi có `weights/layout_best.pt`.

## Kết quả kiểm chứng: source mới khớp notebook

`scripts/verify_parity.py` tái tạo đúng tập test của notebook (56 ảnh, 77 box) và chạy lại bằng source này trên CPU:

| Model | Notebook | Source mới | Lệch |
| --- | --- | --- | --- |
| YOLO26 mAP50 / mAP50-95 | 0.6166 / 0.3475 | 0.6166 / 0.3475 | 0.0000 |
| Faster R-CNN mAP50 / mAP50-95 | 0.4757 / 0.2835 | 0.4757 / 0.2835 | 0.0000 |
| ResNet50 macro-F1 / accuracy | 0.8572 / 0.8208 | 0.8572 / 0.8208 | 0.0000 |

Tốc độ đo trên CPU 4 nhân: YOLO26 khoảng 0,5–0,8 s/ảnh, Faster R-CNN khoảng 3–3,5 s/ảnh.

## Cấu trúc

```
configs/config.yaml        cấu hình (class, ngưỡng, đường dẫn weights, tham số kiến trúc)
weights/                   weights đã train (không đưa lên git, xem weights/README.md)
src/weld_rt/
  config.py                đọc config.yaml (+ ghi đè từ weights/config_final.json)
  schemas.py               cấu trúc JSON kết quả (Pydantic)
  preprocessing.py         đọc ảnh (cả TIFF 16 bit), crop ngữ cảnh cho ResNet50
  models/detectors.py      YOLO26, Faster R-CNN (dựng lại đúng kiến trúc lúc train)
  models/classifier.py     ResNet50
  models/layout.py         detector bố cục film_roi / weld_info
  ocr.py                   PaddleOCR: thử 4 hướng xoay, trả nguyên văn, tách chữ ASTM/IQI
  pipeline.py              Analyzer: bố cục -> OCR -> khuyết tật trong film_roi -> ResNet50
  visualize.py             vẽ box "P 85.3%"
  metrics.py               hàm đánh giá mAP (port từ notebook)
api/main.py                FastAPI: /api/health, /api/models, /api/predict, phục vụ UI
ui/                        giao diện Vue 3 (1 trang, không cần build, Vue đã đóng gói sẵn)
scripts/check_weights.py   kiểm tra weights đúng loại, không nhầm tên
scripts/verify_parity.py   kiểm chứng mAP/F1 khớp notebook
scripts/predict_folder.py  chạy cả thư mục ảnh, xuất JSON/CSV/ảnh vẽ box
tests/                     pytest
```

## Cài đặt (Windows, chạy CPU)

Cần Python 3.10 trở lên.

```bat
cd Film_RT_AI_PRO
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Trên Linux/macOS thay dòng `activate` bằng `source .venv/bin/activate`.

Nếu `pip` báo `UnicodeDecodeError ... cp1252`, chạy `set PYTHONUTF8=1` (cmd) hoặc `$env:PYTHONUTF8=1` (PowerShell) rồi cài lại. Các file requirements đã để ASCII nên bình thường không gặp lỗi này.

OCR (chỉ cần khi đã có `weights/layout_best.pt`): `pip install -r requirements-ocr.txt`. Lần chạy đầu PaddleOCR cần Internet để tải model.

## Chép weights

Chép 3 file vào `weights/` với đúng tên `yolo26_best.pt`, `faster_rcnn_best.pt`, `resnet50_best.pt` (bảng chi tiết ở `weights/README.md`), rồi kiểm tra:

```bat
python scripts\check_weights.py
```

Nếu báo `SAI` thì hai file đang bị đặt nhầm tên.

## Chạy demo

```bat
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. Model được load một lần khi server khởi động.

Trên giao diện:
1. Chọn hoặc kéo thả ảnh film, chọn detector, bấm **Nhận diện**.
2. Kéo thanh ngưỡng (từ 10% trở lên) để lọc box ngay, không cần chạy lại.
3. Rê chuột lên box hoặc dòng trong bảng để xem chi tiết; cột ResNet50 hiện class có xác suất cao nhất, rê chuột để xem đủ 6 class.
4. Tải JSON kết quả hoặc ảnh đã vẽ box.

Tài liệu API tự sinh: http://127.0.0.1:8000/docs

### Gọi API trực tiếp

```bash
curl -F "file=@film.jpg;type=image/jpeg" -F "detector=YOLO26" http://127.0.0.1:8000/api/predict
```

Kết quả JSON gồm `defects` (class, `conf`, `conf_pct`, `bbox` [x1, y1, x2, y2] theo pixel ảnh gốc, softmax ResNet50), `film_roi`, `weld_info` (kèm `ocr.rows`, `ocr.lines`, `ocr.excluded_lines`), `notes`, `timing_ms` và `annotated_jpeg_base64`.

## Kiểm chứng và test

```bat
pytest
python scripts\verify_parity.py --dataset D:\data\v9 --manifest results\split_manifest.csv ^
    --expected results\det_test_summary_final.csv --expected-cls results\cls_test_summary_final.csv
```

`--dataset` là thư mục dataset v9 gốc; các file `results\...` lấy từ output của notebook.

## Lưu ý

- % là điểm tin cậy của model, chưa hiệu chỉnh xác suất (calibration).
- Ngưỡng 10% chỉ dùng cho hiển thị/xuất; mAP vẫn tính với conf 0.001 như notebook.
- Chưa có `layout_best.pt`: detector khuyết tật chạy trên toàn ảnh, không OCR (giao diện có ghi chú).
- Nếu có `weights/config_final.json` từ notebook thì tham số kiến trúc lấy từ file đó, bảo đảm khớp lúc train.
