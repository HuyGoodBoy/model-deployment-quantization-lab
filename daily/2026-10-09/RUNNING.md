# Chạy lại thực nghiệm 09/10/2026

**Bổ sung reproduce source Đức:** xem [PEER-RUNNING.md](experiments/01-reproduce/PEER-RUNNING.md) để chạy ResNet18 trong Linux Docker, [bảng kết quả](PEER-REPRODUCTION.md) gồm 6/6 variant source Đức và 12 cấu hình model Huy. Các mục dưới đây mô tả matrix Windows và M-LSD ban đầu; không dùng chúng thay cho chạy source Đức.

Các lệnh chạy từ **root repo**, PowerShell, Python 3.11 x64. Chạy benchmark tuần tự, đóng ứng dụng nặng, giữ cùng power mode. Không chạy conversion và benchmark đồng thời.

## 1. Cài môi trường bằng uv

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
python scripts/setup.py --profile baseline --check
python scripts/setup.py --profile runtime --check
.\.venv-day09\Scripts\python.exe daily/2026-10-09/tools/environment_probe.py
```

Dependency và lock tập trung ở [environments](../../environments/).
`baseline` giữ TensorFlow 2.15/NumPy 1.26 trong `.venv`; `runtime` giữ ORT/LiteRT
mới và NumPy 2.4 trong `.venv-day09`. Lock runtime dùng pin phiên bản, không có
đường dẫn wheel tuyệt đối của máy Huy. Không cài hai profile vào cùng venv.

Reproduce source Đức dùng [peer-linux](../../environments/peer-linux/requirements.txt)
và Docker; runtime Windows không có đủ converter. Các helper cài đặt/bảo trì cũ
không thuộc code nộp Git.

## 2. Chuẩn bị baseline Huy

Nếu chưa có `quantization/artifacts`, canonical inputs và reference outputs, chạy pipeline cũ theo [quantization/README.md](../../quantization/README.md). Với `.venv` ở root, dùng interpreter như sau:

```powershell
.\.venv\Scripts\python.exe quantization/run_lab.py --task all --stages download
.\.venv\Scripts\python.exe quantization/run_lab.py --task all --stages prepare
.\.venv\Scripts\python.exe quantization/run_lab.py --task all --stages convert
.\.venv\Scripts\python.exe quantization/run_lab.py --task all --stages evaluate
```

Ngày 09/10 giữ nguyên artifact/input của baseline, không convert lại để so runtime mới/cũ. Nếu regenerate, đối chiếu hash với JSON trước khi so kết quả.

## 3. M-LSD reference và gate FP32

```powershell
.\.venv-day09\Scripts\python.exe daily/2026-10-09/experiments/03-mlsd/src/prepare_torch.py
.\.venv\Scripts\python.exe daily/2026-10-09/experiments/03-mlsd/src/port_keras.py --variant fp32
```

Downloader khóa revision và SHA trong `sources.lock.json`. Source Tiny PyTorch được import từ cache riêng; checkpoint load với `weights_only=True`. Bước M-LSD này không dùng repo Đức. Reproduce ResNet18 là pipeline riêng theo PEER-RUNNING.md, thực thi source ghim revision và không chỉnh source gốc.

Gate FP32 lưu cả `allclose_1e4`, relative L2 và max error từng channel. Điều kiện chấp nhận bridge được ghi công khai trong JSON; không thay reference bằng output của model official. Nếu fail gate, dừng quantization và đọc mapping/architecture.

## 4. DistilBERT đối chứng converter

```powershell
.\.venv\Scripts\python.exe daily/2026-10-09/experiments/02-distilbert-int8/src/run.py convert --variant strict
.\.venv\Scripts\python.exe daily/2026-10-09/experiments/02-distilbert-int8/src/run.py convert --variant mixed_mask1e4
.\.venv\Scripts\python.exe daily/2026-10-09/experiments/02-distilbert-int8/src/run.py convert --variant mixed_mask1e2
```

Không sửa source Transformers đã cài; sentinel được đổi trong process bằng bản sao hàm. Với hai bản đổi mask, gate FP32 so với reference trước conversion. Không đổi file baseline.

## 5. Matrix conversion/benchmark

```powershell
.\.venv\Scripts\python.exe daily/2026-10-09/tools/run_matrix.py
```

Runner convert M-LSD FP16/dynamic/static/decoded trước, rồi lần lượt đo DistilBERT, runtime cũ/mới và M-LSD. Ghi log/exit code từng job, tiếp tục các job độc lập nếu có lỗi. Nếu rerun cùng `run-01`, file kết quả bị thay; để lưu nhiều phiên, gọi từng CLI với `--run-id run-02`. Runner mặc định là một phiên cố định.

Ví dụ gọi riêng một phép đo:

```powershell
.\.venv-day09\Scripts\python.exe daily/2026-10-09/experiments/01-reproduce/src/runtime_compare.py --task text --variant onnx_static --runtime new --threads 4 --run-id run-02
```

## 6. Java post-processing trên desktop

```powershell
New-Item -ItemType Directory -Force daily/2026-10-09/experiments/04-box-postprocess/artifacts/classes
New-Item -ItemType Directory -Force daily/2026-10-09/experiments/04-box-postprocess/results/run-01
javac --release 8 -d daily/2026-10-09/experiments/04-box-postprocess/artifacts/classes daily/2026-10-09/experiments/04-box-postprocess/src/BoxPostProcessor.java
java -cp daily/2026-10-09/experiments/04-box-postprocess/artifacts/classes BoxPostProcessor
java -cp daily/2026-10-09/experiments/04-box-postprocess/artifacts/classes BoxPostProcessor daily/2026-10-09/experiments/03-mlsd/results/run-01/lines_fp32_old_t4.csv daily/2026-10-09/experiments/04-box-postprocess/results/run-01/demo_boxes.json
```

Đây là phép đo JVM desktop, không phải Android. Prototype giả định input đã merge; CSV demo chưa qua bước này nên không chứng minh tương đương NAVER. Đọc [ALGORITHM.md](experiments/04-box-postprocess/ALGORITHM.md).

## 7. Báo cáo và kiểm chứng

Tạo báo cáo từ kết quả đã lưu, không chạy lại benchmark:

```powershell
python daily/2026-10-09/tools/build_report.py
python daily/2026-10-09/tools/build_peer_report.py
```

Kiểm chứng đầy đủ khi đã có raw output, model và dữ liệu cục bộ:

```powershell
.\.venv\Scripts\python.exe daily/2026-10-09/tools/verify_results.py
```

`post_checks.py` là runner đo bổ sung thread/delegate/Java, chỉ chạy khi cần làm
lại các đối chứng sau matrix; không chạy để chuẩn bị Git. Cần compile Java ở
bước 6 trước khi chạy runner này. Không chạy benchmark song song.

## 8. Bàn giao qua Git

Giữ code, README/RUNNING/ENVIRONMENT, báo cáo, lock, source manifest và JSON/CSV
kết quả trong repository. `.gitignore` loại model, dataset, tensor NPY/NPZ,
profiler trace, venv/cache, credentials, docs HTML và báo cáo chat. File bị bỏ
khỏi Git vẫn giữ trên máy; clone mới phải tải/tạo lại theo pipeline trước khi
chạy verifier có yêu cầu raw artifact. Không chỉnh hash/command trong evidence
lịch sử chỉ để khớp tên thư mục mới.

Source Đức đã chạy đủ 6/6 biến thể trên Linux. Phần tự đọc/viết lại thuật toán
của Huy vẫn cần hoàn thành; tài liệu giải thích không thay thế phần này.
