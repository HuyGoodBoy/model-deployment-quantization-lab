# Môi trường thực nghiệm ngày 07/10/2026

| Hạng mục | Cấu hình |
|---|---|
| OS / Python | Windows x64 / Python 3.11.9 |
| CPU | Intel Core i7-12700H |
| Runtime | TensorFlow 2.15.1, ONNX Runtime 1.20.1 |
| Converter | tf2onnx 1.16.1, ONNX 1.16.2, opset 13 |
| Input | Batch 1, NHWC FP32 `[1,224,224,3]` |
| Benchmark | CPU intra-op 1 thread, warm-up 30, runs 200 |

Dependency: [requirements](../../requirements.txt), [lock file](../../requirements-lock.txt). Phiên bản và giao thức chi tiết: [báo cáo PC](../../results/report.md). Thông tin CPU được đối chiếu với inventory phần cứng ngày 08/10 trên cùng máy; không phải phép đo phần cứng mới.
