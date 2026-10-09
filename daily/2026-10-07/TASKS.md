# Task ngày 07/10/2026

**Người thực hiện:** Huy.

## Yêu cầu

Tìm hiểu framework deploy model trên các nền tảng; chọn model pretrained, convert ONNX và TFLite chưa quantize, chạy trên PC hoặc Android, đánh giá sai khác output và tốc độ, báo cáo kết quả.

## Kết quả thực hiện

- [x] Khảo sát ONNX Runtime, TFLite/LiteRT, TensorRT, OpenVINO và Core ML.
- [x] Chọn Keras MobileNetV2 pretrained ImageNet, giữ FP32.
- [x] Export ONNX và TFLite độc lập từ model gốc.
- [x] Dùng cùng input và lưu output đầy đủ.
- [x] Tính MAE, RMSE, max error, relative L2, cosine, SNR, agreement và allclose.
- [x] Đo latency CPU, lưu 200 mẫu/runtime.
- [x] Viết báo cáo và kèm code.

**Giới hạn:** hai ảnh không có ground truth; không báo cáo accuracy. PC đáp ứng phạm vi chạy của task; Android được thực hiện vào ngày tiếp theo.
