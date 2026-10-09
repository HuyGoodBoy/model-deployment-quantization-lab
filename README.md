# Triển khai và lượng tử hóa mô hình

**Người thực hiện:** Huy. Thực nghiệm ONNX Runtime, TFLite/LiteRT, CPU, GPU và Android.

## Báo cáo theo ngày

| Ngày | Nội dung | Tài liệu |
|---|---|---|
| 07/10/2026 | MobileNetV2 ONNX/TFLite FP32 | [Code và cách chạy](daily/2026-10-07/README.md) · [Báo cáo](daily/2026-10-07/REPORT.md) |
| 08/10/2026 | Quantization ResNet50/DistilBERT, GPU và Android FP32 | [Code và cách chạy](daily/2026-10-08/README.md) · [Báo cáo](daily/2026-10-08/REPORT.md) |
| **09/10/2026** | **Reproduce Đức, DistilBERT INT8, M-LSD và detect box** | **[Mục lục code](daily/2026-10-09/README.md)** · **[Báo cáo](daily/2026-10-09/REPORT.md)** |

Ngày 09 đã reproduce 6 biến thể ResNet18 từ source Đức và đo 12 cấu hình model Huy
trên Linux. [Bảng đối chiếu](daily/2026-10-09/PEER-REPRODUCTION.md) ghi rõ model,
phần cứng, phần mềm, sai khác output và latency. M-LSD đã convert/quantize; box
mới là prototype Java desktop, chưa đo trên Android.

## Cấu trúc

```text
environments/          Dependency và lock theo môi trường; quản lý bằng uv
scripts/               Setup chung và entry point MobileNetV2
daily/YYYY-MM-DD/       Task, báo cáo, cấu hình và code mới theo ngày
  experiments/         Mỗi task có src/, results/ và artifact cục bộ riêng
  tools/               Điều phối, audit, kiểm chứng và tạo báo cáo
quantization/          Pipeline baseline ResNet50/DistilBERT và ONNX CUDA
lab/                   Các module MobileNetV2 FP32
android/               App/instrumentation MobileNetV2 trên Firebase Test Lab
results/               JSON/CSV/báo cáo MobileNetV2 PC và Android
_template/             Mẫu ngày làm việc mới
```

Code baseline giữ vị trí để tái sử dụng và bảo toàn đường dẫn trong evidence.
[Quy tắc tổ chức theo ngày](daily/README.md).

## Cài bằng uv

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), rồi chạy tại root
repo với Python 3.11 trên Windows x64:

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
```

[Các profile và lock](environments/README.md) · [Cách chạy hôm nay](daily/2026-10-09/RUNNING.md)
· [Reproduce Linux](daily/2026-10-09/experiments/01-reproduce/PEER-RUNNING.md).
Mỗi môi trường dùng interpreter riêng; benchmark chạy tuần tự.

## Bàn giao qua Git

Repository chứa code, báo cáo Markdown, dependency lock, source manifest và
JSON/CSV kết quả có raw latency samples. Model, dataset, tensor NPY/NPZ, profiler
trace, venv, cache, credentials, học liệu HTML trong `docs/` và báo cáo chat giữ
cục bộ, được ignore. Tải/tạo lại artifact theo source lock trước khi chạy verifier
đầy đủ; một bản clone không kèm raw tensor cũ.

Các bảng chỉ báo số đã đo và ghi rõ giới hạn tập dữ liệu, protocol, provider/delegate.
Đổi công cụ cài sang uv giữ nguyên package versions, không được tính là số đo mới.
