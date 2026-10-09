# Convert và quantize M-LSD Tiny 512×512

**Ngày:** 09/10/2026. **Trạng thái:** đã khảo sát nguồn, chưa tải checkpoint/convert/benchmark.

Nguồn: [lhwcv/mlsd_pytorch](https://github.com/lhwcv/mlsd_pytorch), [navervision/mlsd](https://github.com/navervision/mlsd). Revision và checkpoint hash phải được cố định trước khi chạy.

## Pipeline dự kiến

PyTorch checkpoint → FP32 reference → TFLite FP32 → FP16 weights/dynamic range/static INT8 → audit/evaluate/benchmark → báo cáo.

- Sai khác conversion so với chính checkpoint PyTorch, trên cùng input.
- So tốc độ với TFLite chính thức ghi rõ checkpoint khác, input RGBA và decoding trong graph.
- Đo riêng inference, decoding và post-processing; không so raw feature graph với pipeline có top-k như cùng phạm vi.
- Static INT8 dùng ảnh thật calibration riêng với evaluation; nhãn/metric hình học tùy dữ liệu có thể thu được, không gọi agreement là accuracy.
- Lưu failure/unsupported nếu converter hoặc runtime không hỗ trợ, không thay bằng số đo giả.

## Bố trí khi có thực nghiệm

`src/` chứa export/quantize/evaluate/benchmark; `results/<run-id>/` chứa evidence; `artifacts/` chứa model cục bộ. Dependency riêng của bài được ghi ở đây khi đã lựa chọn và kiểm khả dụng, không thay môi trường quantization cũ.

- [Task/tiêu chí](../../TASKS.md)
- [Báo cáo](../../REPORT.md)
- [Post-processing Android](../04-box-postprocess/README.md)
