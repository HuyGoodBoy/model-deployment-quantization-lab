# Kết quả MobileNetV2 trên Android

Thời điểm đo (UTC+7): 2026-10-08T09:47:55.692000+07:00.

Thiết bị báo trong runtime: Google Pixel 8a; Android 14 (API 34); ABI ['arm64-v8a'].

CPU, FP32, batch=1, 1 thread, warm-up=30, runs=200.
Thứ tự runtime: ['onnx', 'tflite']. Phạm vi timer: Java inference API including input/output copies and tensor creation; excludes model/file load, preprocessing, file write and metric calculation.

## Sai khác output so với TensorFlow gốc trên PC

So sánh 2 ảnh × 1.000 lớp; không có nhãn thật nên chưa tính accuracy.

| Metric | ONNX | TFLite |
|---|---:|---:|
| MAE | 1.64218e-09 | 9.9646e-10 |
| RMSE | 3.01082e-08 | 1.60368e-08 |
| Max abs | 1.2517e-06 | 6.55651e-07 |
| Relative L2 | 1.09917e-06 | 5.85464e-07 |
| Cosine | 0.99999999999948008 | 0.99999999999993294 |
| SNR dB | 119.179 | 124.65 |
| Top-1 agreement | 1 | 1 |
| Top-5 overlap | 1 | 1 |
| Allclose | True | True |

## Tốc độ trên Android

| Runtime | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|
| onnx | 36.004 | 35.284 | 40.406 |
| tflite | 22.391 | 22.092 | 24.640 |

## Cách diễn giải và giới hạn

- Đối chiếu model thiết bị/Android với cấu hình test Firebase; ghi rõ physical device hay emulator.
- Chỉ hai ảnh mẫu; agreement không phải accuracy. Dữ liệu latency là các lần lặp của ảnh đầu tiên.
- ONNX Runtime Android 1.20.0/PC 1.20.1 và TFLite Android 2.15.0/PC 2.15.1 khác phiên bản; timer Java khác timer Python.
- Hai runtime chạy nối tiếp trong một test process; thứ tự, nhiệt độ và tải nền có thể ảnh hưởng kết quả.
- Một thread intra-op không có nghĩa toàn bộ hệ thống hoặc runtime chỉ có một thread.
- Xem metadata runtime và nhiệt độ/thermal status trước và sau đo trong android_run.json.
- Các số liệu PC ngày 07/10/2026 được giữ riêng; không suy ra mọi điện thoại từ một thiết bị.
