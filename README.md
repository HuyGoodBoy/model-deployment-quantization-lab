# Triển khai và lượng tử hóa mô hình

**Người thực hiện:** Huy. Thực nghiệm ONNX Runtime, TFLite/LiteRT, CPU, GPU và Android.

## Báo cáo

| Ngày | Nội dung | Kết quả | Code |
|---|---|---|---|
| 07/10/2026 | MobileNetV2 ONNX/TFLite FP32 | [Báo cáo](daily/2026-10-07/REPORT.md) | [lab](lab/) |
| 08/10/2026 | Quantization ResNet50/DistilBERT, GPU và Android FP32 | [Báo cáo](daily/2026-10-08/REPORT.md) | [quantization](quantization/) · [android](android/) |
| 09/10/2026 | Reproduce Đức, DistilBERT INT8, M-LSD và detect box | [Báo cáo](daily/2026-10-09/REPORT.md) · [Đối chiếu reproduce](daily/2026-10-09/PEER-REPRODUCTION.md) | [experiments](daily/2026-10-09/experiments/) |

Ngày 09 đã reproduce 6 biến thể ResNet18 từ source Đức và đo 12 cấu hình model Huy
trên Linux. M-LSD đã convert/quantize; detect box mới có prototype Java desktop,
chưa đo trên Android. Báo cáo ghi rõ môi trường, protocol và giới hạn từng phép đo.

## Cấu trúc

```text
environments/          Dependency lock theo môi trường
scripts/               Setup bằng uv và entry point MobileNetV2
daily/YYYY-MM-DD/       Báo cáo, cấu hình môi trường và thực nghiệm theo ngày
  experiments/         Source và kết quả từng thực nghiệm
  tools/               Điều phối, kiểm chứng và tạo báo cáo
quantization/          Pipeline ResNet50/DistilBERT và ONNX CUDA
lab/                   Pipeline MobileNetV2 FP32
android/               App và instrumentation test
results/               Kết quả MobileNetV2 PC và Android
```

## Tái lập

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), dùng Python 3.11
x64, chạy từ root repo trên Windows:

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
```

| Profile | Môi trường | Dependency lock |
|---|---|---|
| `baseline` | `.venv/`: TensorFlow 2.15.1, ORT 1.20.1 | [Lock](environments/baseline/requirements-lock.txt) |
| `runtime` | `.venv-day09/`: PyTorch 2.13, ORT 1.30, LiteRT 2.2 | [Lock](environments/runtime/requirements-lock.txt) |
| `gpu` | `quantization/.venv-gpu/`: ORT GPU 1.20.2 và CUDA/cuDNN | [Lock](environments/gpu/requirements-lock.txt) |
| `mobilenet` | `.venv-mobilenet/`: dependency thực nghiệm ngày 07 | [Lock](environments/mobilenet/requirements-lock.txt) |
| `peer-linux` | Docker: reproduce source Đức trên Linux CPU | [Dockerfile](environments/peer-linux/Dockerfile) · [Lock](environments/peer-linux/requirements-lock.txt) |

Bốn lock Windows dùng Python 3.11 x64; Linux có lock riêng. Setup dùng `uv venv`
và `uv pip sync --strict`; thêm `--check` để kiểm dependency mà không cài lại.

Lệnh thực nghiệm: [ngày 09](daily/2026-10-09/RUNNING.md),
[reproduce Linux](daily/2026-10-09/experiments/01-reproduce/PEER-RUNNING.md),
[quantization](quantization/README.md), [GPU](quantization/README-gpu.md),
[Android](android/README.md). Chạy benchmark tuần tự bằng interpreter của từng profile.

Git lưu code, báo cáo, lock, source manifest và JSON/CSV kết quả.
Model, dataset, tensor thô, profiler trace, venv, cache, credentials và học liệu
cục bộ được ignore. Cần tải/tạo lại artifact trước khi chạy toàn bộ verifier.
