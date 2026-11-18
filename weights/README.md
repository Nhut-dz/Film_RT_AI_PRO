# Weights đã train (không lưu trên git)

Chép các file kết quả train từ notebook (Kaggle/Colab `outputs/full/`) vào thư mục này, **đổi tên** như sau:

| File trong thư mục này | Lấy từ notebook | Bắt buộc |
| --- | --- | --- |
| `yolo26_best.pt` | `checkpoints/yolo26/train_final/weights/best.pt` | Có (hoặc Faster R-CNN) |
| `faster_rcnn_best.pt` | `checkpoints/faster_rcnn/train_final/best.pt` | Tuỳ chọn |
| `resnet50_best.pt` | `checkpoints/resnet50/train_final/best.pt` | Tuỳ chọn (softmax ResNet50) |
| `layout_best.pt` | `checkpoints/layout_yolo26/train_final/weights/best.pt` | Tuỳ chọn (vùng film + OCR) |
| `config_final.json` | `results/config_final.json` | Tuỳ chọn (ghi đè tham số trong configs/config.yaml) |

Lưu ý: trình duyệt có thể lưu file `.pt` thành `.zip` (vì `.pt` là định dạng zip). **Không giải nén**, chỉ đổi đuôi về `.pt`.
Kiểm tra đúng file: `python scripts/check_weights.py`.
