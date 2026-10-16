# Triển khai và lượng tử hóa mô hình

**Người thực hiện:** Huy. Thực nghiệm ONNX Runtime, TFLite/LiteRT, CPU, GPU và Android.

## Báo cáo

| Ngày | Nội dung | Kết quả | Code |
|---|---|---|---|
| 07/10/2026 | MobileNetV2 ONNX/TFLite FP32 | [Báo cáo](daily/2026-10-07/REPORT.md) | [lab](lab/) |
| 08/10/2026 | Quantization ResNet50/DistilBERT, GPU và Android FP32 | [Báo cáo](daily/2026-10-08/REPORT.md) | [quantization](quantization/) · [android](android/) |
| 09/10/2026 | Reproduce Đức, DistilBERT INT8, M-LSD và detect box | [Báo cáo](daily/2026-10-09/REPORT.md) · [Đối chiếu reproduce](daily/2026-10-09/PEER-REPRODUCTION.md) | [experiments](daily/2026-10-09/experiments/) |
| 10/10/2026 | Đọc source, nghiên cứu thuật toán M-LSD và quantization | [Báo cáo](daily/2026-10-10/REPORT.md) | [Ghi chép nghiên cứu](daily/2026-10-10/RESEARCH.md) |
| 12/10/2026 | M-LSD: Hough và ghép box Java, đối chiếu Python, thử danh sách kề | [Báo cáo](daily/2026-10-12/REPORT.md) | [Code và kết quả](daily/2026-10-12/experiments/mlsd-postprocess/) |
| 13/10/2026 | Task PTQ M-LSD: tách head và đánh giá trên ảnh có nhãn; chạy bổ sung 14/10 | [Báo cáo](daily/2026-10-13/REPORT.md) | [mlsd-ptq](daily/2026-10-13/experiments/mlsd-ptq/) |
| 14/10/2026 | QAT M-LSD và DistilBERT: pilot 8 bước/model, task tuần đang làm | [Tiến độ](daily/2026-10-14/REPORT.md) | [qat](daily/2026-10-14/experiments/qat/) |
| 15/10/2026 | QAT 128 bước/model; calibration M-LSD tăng chất lượng trên 32 ảnh; sửa làm tròn export giảm lệch tensor, chưa đạt gate 2% | [Báo cáo](daily/2026-10-15/REPORT.md) · [Research](daily/2026-10-15/RESEARCH.md) | [qat](daily/2026-10-15/experiments/qat/) |
| 16/10/2026 | Chốt 54 ảnh M-LSD và 772 câu SST-2; export integer DistilBERT, đối chiếu CPU/GPU và kiểm chứng cuối tuần | [Báo cáo ngày](daily/2026-10-16/REPORT.md) · [Tổng kết tuần](daily/2026-10-16/WEEKLY-REPORT.md) | [Code và kết quả](daily/2026-10-16/experiments/finalize/) |

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
Model, dataset, tensor thô, profiler trace, venv, cache, credentials và tài liệu
nội bộ được ignore. Cần tải/tạo lại artifact trước khi chạy toàn bộ verifier.

## Học liệu HTML

Tải file và mở bằng trình duyệt để xem nội dung tương tác:

- [Ngày 07: triển khai model](docs/2026-10-07/hoc-deploy-model.html)
- [Ngày 08: quantization](docs/2026-10-08/hoc-quantization.html)
- [Ngày 09: reproduce, DistilBERT và M-LSD/post-processing](docs/2026-10-09/hoc-ngay-2026-10-09.html)
- [Ngày 12: học thuật toán bằng hình tương tác](docs/2026-10-12/hoc-thuat-toan.html)

HTML ngày 09 có 4 ô tự giải thích phương pháp/thuật toán theo yêu cầu mentor. Mở bằng Chrome/Edge, viết câu trả lời,
bấm **Save vào HTML** rồi chọn file đang mở ở lần lưu đầu để cấp quyền ghi.
Câu trả lời được nhúng vào file; có thể tải bản HTML đã điền nếu trình duyệt không hỗ trợ ghi trực tiếp.
