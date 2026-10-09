# Báo cáo ngày 07/10/2026

**Người thực hiện:** Huy. **Model:** MobileNetV2 pretrained ImageNet. **Thiết bị:** CPU Windows.

## 1. Nội dung hoàn thành

Khảo sát runtime triển khai model; export MobileNetV2 FP32 sang ONNX/TFLite; so output với TensorFlow gốc và đo inference CPU trên cùng input.

## 2. Phương pháp

Hai ảnh RGB, resize bilinear 224×224, chuẩn hóa [-1,1], NHWC FP32 batch 1. Output gồm 1.000 xác suất softmax. Benchmark 1 thread intra-op, 30 warm-up, 200 lượt trong process riêng; timer loại tải model và preprocessing.

## 3. Kết quả

| Runtime | Median ms | P95 ms | Max error so TensorFlow | SNR dB | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 24,831 | 72,665 | 0 | ∞ | Reference |
| ONNX FP32 | 9,112 | 11,690 | 6,55651e-7 | 124,190 | 100% |
| TFLite FP32 | 53,418 | 62,264 | 9,53674e-7 | 121,621 | 100% |

Các số liệu lấy từ [báo cáo chi tiết PC](../../results/report.md); [summary JSON](../../results/summary.json) và raw output (`results/outputs.npz`, cục bộ, tạo lại bằng pipeline) lưu dữ liệu đối chiếu.

## 4. Nhận xét và giới hạn

ONNX và TFLite đạt tolerance output trên hai ảnh đã dùng. ONNX có latency thấp nhất trong phiên đo này. Không có nhãn thật nên agreement không phải accuracy; không suy rộng thứ hạng runtime sang mọi thiết bị.

## 5. Hướng tiếp theo

Chạy cùng model/input trên Android thật, thu output và latency riêng. Công việc đã triển khai ở ngày [08/10/2026](../2026-10-08/REPORT.md).

## 6. Mã nguồn và bằng chứng

- [Pipeline](../../lab/)
- [Môi trường](ENVIRONMENT.md)
- [Báo cáo chi tiết](../../results/report.md)
