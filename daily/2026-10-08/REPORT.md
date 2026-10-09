# Báo cáo ngày 08/10/2026

**Người thực hiện:** Huy.

## 1. Công việc hoàn thành

Thử quantization ONNX Runtime/TFLite với ResNet50 và DistilBERT; đánh giá cả output, accuracy/F1, dung lượng và latency. Bổ sung ONNX CUDA trên RTX 3050 Laptop. Thực nghiệm Android riêng dùng MobileNetV2 FP32 trên Pixel 8a.

## 2. Phương pháp

ResNet50 dùng Imagenette, DistilBERT dùng SST-2 tiếng Anh; mỗi model có 100 mẫu calibration và 100 mẫu evaluation không giao nhau. TensorFlow FP32 là reference. Benchmark batch 1, CPU 1 thread, 30 warm-up, 200 lượt; cặp CPU/CUDA dùng cùng graph và ONNX Runtime 1.20.2. GPU timer gồm truyền input/output.

## 3. Kết quả chính

| Thực nghiệm | Median latency | Accuracy | Nhận xét |
|---|---|---|---|
| ResNet50 ONNX FP32 CPU → CUDA | 83,232 → 5,903 ms | 73% | Speedup cùng cặp 14,10× |
| ResNet50 ONNX FP16 CUDA | 3,665 ms | 73% | File 48,74 MiB, SNR 53,407 dB |
| DistilBERT ONNX FP32 CPU → CUDA | 70,281 → 4,833 ms | 91% | Speedup cùng cặp 14,54× |
| DistilBERT ONNX FP16 CUDA | 11,168 ms | 89% | Chậm hơn FP32 CUDA trong lượt đo |
| DistilBERT TFLite FP16 weights CPU | 344,053 ms | 91% | Giảm file khoảng 2× |
| DistilBERT TFLite calibrated mixed INT8 CPU | 268,074 ms | 52% | Chưa đạt chất lượng triển khai |

Số liệu, SNR, MAE/RMSE, cosine, agreement, P95 và provider audit đầy đủ nằm trong [báo cáo CPU/GPU](../../quantization/bao-cao-ngay-2026-10-08-cpu-gpu.md). Bản [CPU chi tiết](../../quantization/bao-cao-quantization-2026-10-08.md) ghi riêng phiên ONNX Runtime 1.20.1.

Android MobileNetV2: ONNX Runtime median 35,284 ms, TFLite XNNPACK 22,092 ms; SNR 119,179 và 124,650 dB. Cả hai đạt allclose và top-1 agreement 2/2 ảnh. [Báo cáo Android](../../results/android/bao-cao-2026-10-08.md).

## 4. Nhận xét và giới hạn

Giảm dung lượng không bảo đảm tăng tốc. INT8 CUDA có CPU fallback; static QDQ không chứng minh toàn bộ phép tính GPU là integer. Evaluation chỉ 100 mẫu/model và timing lặp một input. Kết quả không đại diện mọi thiết bị hoặc toàn dataset. Verifier GPU chưa chạy xác nhận trong lần bàn giao này; ZIP CPU không gồm phần GPU bổ sung.

## 5. Hướng tiếp theo

Reproduce môi trường của Đức, đối chiếu dynamic/static giữa các công cụ, làm rõ DistilBERT mixed INT8 và mở rộng M-LSD. Nội dung được theo dõi tại ngày [09/10/2026](../2026-10-09/REPORT.md).

## 6. Mã nguồn và bằng chứng

- [Quantization CPU/GPU](../../quantization/)
- [Android FP32](../../android/)
- [Môi trường](ENVIRONMENT.md)
