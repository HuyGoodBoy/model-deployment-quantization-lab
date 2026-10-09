# Môi trường và nguồn đối chiếu ngày 09/10/2026

**Trạng thái:** đã đối chứng CPU Windows, bổ sung Linux Docker/WSL2 chạy source Đức đủ 6/6 ResNet18 và 12 cấu hình model Huy. Phần cứng khác vẫn reproduce được; so latency phải ghi riêng OS/ISA/backend.

## 1. Baseline Huy ngày 08/10

- Windows x64, Python 3.11.9, Intel i7-12700H, RAM 16 GB.
- CPU: TensorFlow 2.15.1, ONNX Runtime 1.20.1, Transformers 4.38.2.
- GPU đối chứng: ONNX Runtime GPU 1.20.2, RTX 3050 Laptop 4 GB.
- Benchmark: 1 thread, batch 1, 30 warm-up, 200 lượt.

Nguồn: [inventory CPU baseline ngày 08/10](../../quantization/results/cv/benchmark_tensorflow_fp32.json), [inventory GPU](../../quantization/results/gpu/environment.json).

## 2. Môi trường Đức theo repo

- Repo: [TruongDuke/Quantization, 4-quantization](https://github.com/TruongDuke/Quantization/tree/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization).
- Revision đã đọc: `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`.
- Báo cáo ghi Mac M1 Pro, CPU 4 thread; README hướng dẫn Python 3.11.
- Dependency khai báo: torch 2.13.0, torchvision 0.28.0, NumPy 2.4.6, Pillow 12.3.0, ONNX 1.23.2, onnxscript 0.7.2, ONNX Runtime 1.30.0, litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0.
- Benchmark trong code: batch 1, 10 warm-up, 100 lượt, median; CPUExecutionProvider và LiteRT Interpreter 4 thread.

Đây là phiên bản khai báo trong [requirements.txt](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/requirements.txt). Cài toàn bộ trên Windows thất bại vì litert-converter 0.4.* không có Windows distribution phù hợp. Phiên bổ sung đã khởi động được Docker/WSL2, cài Linux litert-converter 0.4.0 và toàn bộ mười release versions Đức khai báo; pip check thành công. Đây là khác biệt platform hỗ trợ package, không phải yêu cầu phải dùng Mac.

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

Inventory: [environment.json](experiments/01-reproduce/results/run-01/environment.json). Lock thực cài: [requirements-runtime-lock.txt](../../environments/runtime/requirements-lock.txt). `pip check` đã thành công. Baseline không nâng phiên bản; `.venv-day09` chỉ là runtime/software subset của bộ Đức khai báo.

- Matrix runtime, DistilBERT và TFLite M-LSD: CPU 4 thread, batch 1, 30 warm-up, 200 mẫu latency. ORT CPUExecutionProvider, inter-op=1, default sequential execution/graph optimization.
- PyTorch M-LSD reference: 1 intra-op/1 inter-op, 30/200. Latency khác thread/scope, không so trực tiếp với TFLite 4 thread.
- Process riêng chạy tuần tự; conversion xong trước benchmark. Timer API đồng bộ gồm copy/q-deq I/O, loại model load và preprocess. TFLite dùng delegate mặc định, XNNPACK ghi trong log; chưa audit mọi partition.
- Java geometry: desktop JVM JDK 21.0.12.1, compile `--release 8`, chưa Android. Hôm nay dùng CPU, không benchmark GPU.
- Phiên `run-01`; [execution.json](experiments/01-reproduce/results/run-01/execution.json) lưu command/exit code. Chưa lặp nhiều session hoặc profile nhiệt độ/power mode.

## 5. Nguồn M-LSD và dữ liệu

PyTorch revision `2312205254e66911703decf775f626995d260f17`, NAVER `453cafa09467d0272760578d35c1fda38e8895a5`. [sources.lock.json](experiments/03-mlsd/sources.lock.json) chứa URL/SHA256 source/checkpoint/demo/official model. Copy cùng weights qua bridge Keras, không retrain.

Calibration 32 ảnh train Imagenette, evaluation 10 ảnh validation, seed 20261009, không trùng hash. Tập này chưa có nhãn line/box, chưa đúng hoàn toàn miền deployment. Input RGBA 512×512 alpha=1; normalization nằm trong bridge TFLite. Raw output 256×256×9; graph decoded trả points/scores/vmap.

Nguồn Apache 2.0: [PyTorch license](experiments/03-mlsd/LICENSE-MLSD-PYTORCH.txt), [NAVER license](experiments/03-mlsd/LICENSE-MLSD-NAVER.txt). Giữ source attribution và source lock khi bàn giao.

## 6. Phiên reproduce source Đức trên Linux (bổ sung)

- Linux 6.18.33.2 WSL2 x86_64, Debian bookworm, Python 3.11.17; cùng CPU vật lý i7-12700H của Huy. Docker engine nhìn thấy 20 logical CPUs và khoảng 7.57 GiB RAM. Không có CPU quota riêng cho container; inference đặt 4 thread theo source Đức.
- Image Python ghim digest trong [Dockerfile](../../environments/peer-linux/Dockerfile); [lock Linux thực cài](../../environments/peer-linux/requirements-lock.txt) ghi cả dependency gián tiếp của phiên Huy.
- Mười release dependencies khớp repo Đức; Torch/torchvision dùng CPU build. Không có full transitive lock và artifact hash máy Mac Đức để khẳng định hai môi trường giống tuyệt đối.
- `run-04-peer-linux`: q1/q2/q3 nguyên script, q4 giữ nguyên hàm với collector/tách variant, 6/6 biến thể có output/latency. ResNet18: batch 1, 4 thread, 10 warm-up/100 lượt. Source mount read-only, hash trước/sau không đổi.
- `run-04-huy-linux`: 12 cấu hình ResNet50/DistilBERT Huy, giữ model/input hash, 4 thread/30 warm-up/200 lượt để so Windows run-01. Các phép đo chạy tuần tự, conversion kết thúc trước benchmark.
- ONNX profile và tie audit chạy riêng sau benchmark. TFLite static reproduce được range max 14.584794 so reference max 32.523399 và 62 mẫu hòa top-1; 32 vẫn đoán đúng, 30 đoán sai trong nhóm hòa.
- [Bảng đối chiếu và phân tích](PEER-REPRODUCTION.md), [cách chạy](experiments/01-reproduce/PEER-RUNNING.md). Không yêu cầu cùng Mac để reproduce; hardware/OS/backend là điều kiện phải báo khi so tốc độ, không phải lý do bỏ chạy source.

## Chuyển công cụ cài đặt và cấu trúc Git

Setup hiện dùng `uv venv` / `uv pip sync`; đã sync và kiểm baseline, runtime mới
và GPU bằng uv, giữ nguyên phiên bản đã đo. Lock được gom vào `environments/`;
lock runtime thay local wheel URL bằng pin cùng phiên bản và bổ sung đúng
onnxscript/onnx-ir đã cài khi reproduce. Không chạy benchmark mới cho thay đổi này.
Dockerfile mới dùng uv; metadata image/run cũ không được sửa thành image mới.
