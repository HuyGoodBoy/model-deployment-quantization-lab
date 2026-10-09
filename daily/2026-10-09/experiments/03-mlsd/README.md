# Convert và quantize M-LSD Tiny 512×512

**Ngày:** 09/10/2026. **Trạng thái:** đã tải source/checkpoint có hash, chạy PyTorch reference và chuyển cùng weights sang TFLite.

Nguồn: [lhwcv/mlsd_pytorch](https://github.com/lhwcv/mlsd_pytorch), [navervision/mlsd](https://github.com/navervision/mlsd). Revision và checkpoint hash phải được cố định trước khi chạy.

## Pipeline thực hiện

PyTorch checkpoint → FP32 reference → bridge Keras cùng weights → TFLite FP32 → FP16 weights/dynamic range/static INT8 → audit/evaluate/benchmark → báo cáo. Thêm decoded_fp32 cùng contract với official để so tốc độ.

- Sai khác conversion so với chính checkpoint PyTorch, trên cùng input.
- So tốc độ với TFLite chính thức ghi rõ checkpoint khác, input RGBA và decoding trong graph.
- Đo riêng inference, decoding và post-processing; không so raw feature graph với pipeline có top-k như cùng phạm vi.
- Static INT8 dùng ảnh thật calibration riêng với evaluation; nhãn/metric hình học tùy dữ liệu có thể thu được, không gọi agreement là accuracy.
- Lưu failure/unsupported nếu converter hoặc runtime không hỗ trợ, không thay bằng số đo giả.

## Code và evidence

`src/` chứa export/quantize/evaluate/benchmark; `results/<run-id>/` chứa evidence; `artifacts/` chứa model cục bộ. Dependency riêng của bài được ghi ở đây khi đã lựa chọn và kiểm khả dụng, không thay môi trường quantization cũ.

- [prepare_torch.py](src/prepare_torch.py): khóa nguồn, dữ liệu 32/10 ảnh không trùng, Torch reference/state.
- [port_keras.py](src/port_keras.py): mapping conv/depthwise/BN/padding/resize, kiểm bridge FP32, converter các variant.
- [measure.py](src/measure.py): raw output metrics, metrics center/displacement riêng, latency và line CSV.
- [sources.lock.json](sources.lock.json), [results/run-01](results/run-01/), [cách chạy](../../RUNNING.md).

Source liên quan Apache 2.0: [lhwcv license](LICENSE-MLSD-PYTORCH.txt), [NAVER license](LICENSE-MLSD-NAVER.txt). Bản bridge là adaptation, không phải source upstream nguyên bản. Calibration Imagenette chưa có nhãn line/box; không báo accuracy detection.

- [Task/tiêu chí](../../TASKS.md)
- [Báo cáo](../../REPORT.md)
- [Post-processing Android](../04-box-postprocess/README.md)
