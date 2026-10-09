# Chạy lại thực nghiệm 09/10/2026

Các lệnh chạy từ **root repo**, PowerShell, Python 3.11 x64. Chạy benchmark tuần tự, đóng ứng dụng nặng, giữ cùng power mode. Không chạy conversion và benchmark đồng thời.

## 1. Hai môi trường độc lập

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r quantization/requirements-lock.txt
py -3.11 -m venv .venv-day09
.\.venv-day09\Scripts\python.exe -m pip install -r daily/2026-10-09/requirements-runtime-lock.txt
.\.venv-day09\Scripts\python.exe -m pip check
.\.venv-day09\Scripts\python.exe daily/2026-10-09/environment_probe.py
```

`requirements-peer.txt` ghi nguyên phiên bản Đức khai báo để đối chiếu, **không phải bộ đã cài thành công trên Windows**. `requirements-runtime-lock.txt` là bộ thực cài và đã kiểm dependency. Không gộp TensorFlow 2.15 và NumPy 2.4 vào cùng venv.

Nếu tải wheel chậm, helper `fetch_wheel.py PACKAGE VERSION` tải wheel Windows Python 3.11 theo URL PyPI và kiểm SHA256. Không dùng helper cho platform khác. Lần chạy đầu cần Internet; trọng số, dataset và wheel không lưu trong Git.

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

Downloader khóa revision và SHA trong `sources.lock.json`. Source Tiny PyTorch được import từ cache riêng; checkpoint load với `weights_only=True`. Repo Đức chỉ được đọc, không được import/thực thi hoặc sửa trong pipeline này.

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
.\.venv\Scripts\python.exe daily/2026-10-09/run_matrix.py
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

## 7. Báo cáo, kiểm chứng và ZIP

```powershell
.\.venv\Scripts\python.exe daily/2026-10-09/post_checks.py
.\.venv\Scripts\python.exe daily/2026-10-09/build_report.py
.\.venv\Scripts\python.exe daily/2026-10-09/verify_results.py
.\.venv\Scripts\python.exe daily/2026-10-09/package_submission.py
```

Sau matrix, `post_checks.py` chạy hai đối chứng thread, hai audit delegate và Java smoke test. Cần compile Java ở bước 6 trước khi gọi script này. Chạy tuần tự trước verifier.

MD/JSON/CSV được commit; chat report, model, dataset, venv, log và cache được ignore. ZIP cục bộ chứa code và raw evidence đã tạo, không chứa model/credential; model/data tải lại bằng source lock. Phần tự đọc thuật toán của Huy và reproduce đầy đủ pipeline Đức vẫn phải được hoàn thành riêng, không được đánh dấu thay bằng tài liệu AI.
