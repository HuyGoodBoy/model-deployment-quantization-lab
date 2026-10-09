# Môi trường và nguồn đối chiếu ngày 09/10/2026

**Trạng thái:** đã tạo môi trường đối chiếu riêng và chạy thực nghiệm CPU; chưa reproduce đầy đủ pipeline converter hoặc phần cứng Mac của Đức.

## 1. Baseline Huy ngày 08/10

- Windows x64, Python 3.11.9, Intel i7-12700H, RAM 16 GB.
- CPU: TensorFlow 2.15.1, ONNX Runtime 1.20.1, Transformers 4.38.2.
- GPU đối chứng: ONNX Runtime GPU 1.20.2, RTX 3050 Laptop 4 GB.
- Benchmark: 1 thread, batch 1, 30 warm-up, 200 lượt.

Nguồn: [môi trường ngày 08/10](../2026-10-08/ENVIRONMENT.md), [inventory GPU](../../quantization/results/gpu/environment.json).

## 2. Môi trường Đức theo repo

- Repo: [TruongDuke/Quantization, 4-quantization](https://github.com/TruongDuke/Quantization/tree/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization).
- Revision đã đọc: `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`.
- Báo cáo ghi Mac M1 Pro, CPU 4 thread; README hướng dẫn Python 3.11.
- Dependency khai báo: torch 2.13.0, torchvision 0.28.0, NumPy 2.4.6, Pillow 12.3.0, ONNX 1.23.2, onnxscript 0.7.2, ONNX Runtime 1.30.0, litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0.
- Benchmark trong code: batch 1, 10 warm-up, 100 lượt, median; CPUExecutionProvider và LiteRT Interpreter 4 thread.

Đây là phiên bản khai báo trong [requirements.txt](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/requirements.txt). Thử cài toàn bộ trên Windows thất bại: litert-torch 0.9.4 yêu cầu litert-converter 0.4.*, pip không tìm được Windows distribution phù hợp. Không kết luận các phiên bản khai báo không tồn tại. Docker daemon chưa chạy và chưa có Linux/WSL hoạt động trong phiên này.

## 3. Nội dung cần lưu khi chạy phiên mới

Mỗi phiên lưu dependency thực cài, model/input/output hash và đủ latency samples. Cài cùng dependency trên Windows không biến máy Windows thành Mac M1 Pro; phải phân biệt đối chiếu software với đối chiếu phần cứng.

## 4. Môi trường và giao thức thực chạy hôm nay

| Thành phần | `.venv` baseline | `.venv-day09` đối chiếu |
| --- | --- | --- |
| Python | 3.11.9 | 3.11.9 |
| NumPy | 1.26.4 | 2.4.6 |
| TensorFlow / Keras | 2.15.1 / 2.15 | Không cài |
| Transformers | 4.38.2 | Không cài |
| ONNX Runtime | 1.20.1 | 1.30.0 |
| ONNX | 1.16.2 | 1.23.2 |
| LiteRT Interpreter | TF Lite 2.15.1 | ai-edge-litert 2.2.0 |
| PyTorch / torchvision | Không cài | 2.13.0 / 0.28.0 |
| Pillow | 10.4.0 | 12.3.0 |

Inventory: [environment.json](experiments/01-reproduce/results/run-01/environment.json). Lock thực cài: [requirements-runtime-lock.txt](requirements-runtime-lock.txt). `pip check` đã thành công. Baseline không nâng phiên bản; `.venv-day09` chỉ là runtime/software subset của bộ Đức khai báo.

- Matrix runtime, DistilBERT và TFLite M-LSD: CPU 4 thread, batch 1, 30 warm-up, 200 mẫu latency. ORT CPUExecutionProvider, inter-op=1, default sequential execution/graph optimization.
- PyTorch M-LSD reference: 1 intra-op/1 inter-op, 30/200. Latency khác thread/scope, không so trực tiếp với TFLite 4 thread.
- Process riêng chạy tuần tự; conversion xong trước benchmark. Timer API đồng bộ gồm copy/q-deq I/O, loại model load và preprocess. TFLite dùng delegate mặc định, XNNPACK ghi trong log; chưa audit mọi partition.
- Java geometry: desktop JVM JDK 21.0.12.1, compile `--release 8`, chưa Android. Hôm nay dùng CPU, không benchmark GPU.
- Phiên `run-01`; [execution.json](experiments/01-reproduce/results/run-01/execution.json) lưu command/exit code. Chưa lặp nhiều session hoặc profile nhiệt độ/power mode.

## 5. Nguồn M-LSD và dữ liệu

PyTorch revision `2312205254e66911703decf775f626995d260f17`, NAVER `453cafa09467d0272760578d35c1fda38e8895a5`. [sources.lock.json](experiments/03-mlsd/sources.lock.json) chứa URL/SHA256 source/checkpoint/demo/official model. Copy cùng weights qua bridge Keras, không retrain.

Calibration 32 ảnh train Imagenette, evaluation 10 ảnh validation, seed 20261009, không trùng hash. Tập này chưa có nhãn line/box, chưa đúng hoàn toàn miền deployment. Input RGBA 512×512 alpha=1; normalization nằm trong bridge TFLite. Raw output 256×256×9; graph decoded trả points/scores/vmap.

Nguồn Apache 2.0: [PyTorch license](experiments/03-mlsd/LICENSE-MLSD-PYTORCH.txt), [NAVER license](experiments/03-mlsd/LICENSE-MLSD-NAVER.txt). Giữ source attribution và source lock khi bàn giao.
