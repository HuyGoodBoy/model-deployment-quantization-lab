# Task ngày 08/10/2026

**Người thực hiện:** Huy.

## Yêu cầu quantization

Tìm hiểu quantization với ONNX Runtime và TFLite; dùng mô hình khác, đánh giá và bàn giao báo cáo cùng code. Mở rộng thêm model text và GPU trên máy.

- [x] Chọn ResNet50 và DistilBERT SST-2, giữ pretrained weights.
- [x] Chuẩn bị calibration/evaluation có nhãn, tách riêng 100/100 mẫu mỗi model.
- [x] Export và chạy 8 biến thể/model trên CPU.
- [x] Đo output metrics, accuracy/F1, dung lượng và latency.
- [x] Đo ONNX FP32, dynamic INT8, static QDQ và FP16 với CPU/CUDA đối chứng.
- [x] Lưu profiling để xác minh provider/operator và CPU fallback.
- [x] Kiểm chứng dữ liệu CPU, 10 unit test của module đã pass.
- [ ] Chạy verifier GPU xác nhận toàn bộ evidence bổ sung.
- [x] Viết báo cáo Markdown CPU/GPU; ZIP CPU đã được tạo.
- [ ] Đóng gói và kiểm chứng ZIP gồm phần GPU bổ sung.

## Bổ sung Android từ pipeline FP32

- [x] Build ứng dụng và instrumentation test MobileNetV2.
- [x] Chạy trên Pixel 8a thật qua Firebase Test Lab.
- [x] Thu output, latency, kiểm checksum và tính metric so với TensorFlow gốc.
- [x] Viết báo cáo Android riêng.

**Chưa thực hiện:** quantized ResNet50/DistilBERT trên Android, GPU delegate điện thoại, RAM/VRAM đỉnh và năng lượng.
