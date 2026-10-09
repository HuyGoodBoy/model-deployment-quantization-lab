# Môi trường Python — uv

Dependency được quản lý bằng **uv**. `uv pip` là giao diện requirements của uv,
không gọi chương trình pip. Giữ các lock đã đo để không đổi phiên bản runtime
khi tái lập kết quả; không gộp TensorFlow/NumPy 1.x và LiteRT/NumPy 2.x.

| Profile | Phạm vi | Venv trên Windows |
|---|---|---|
| `baseline` | TensorFlow 2.15.1, ORT 1.20.1, ResNet50/DistilBERT; dùng được cho MobileNetV2 | `.venv/` |
| `runtime` | PyTorch 2.13, ORT 1.30, LiteRT 2.2; đối chứng ngày 09 | `.venv-day09/` |
| `gpu` | ORT GPU 1.20.2 và CUDA/cuDNN wheels | `quantization/.venv-gpu/` |
| `mobilenet` | Bộ dependency riêng của thực nghiệm ngày 07 | `.venv-mobilenet/` |
| `peer-linux` | Reproduce ResNet18 Đức với converter Linux, CPU | Docker |

Mỗi profile có `requirements.txt` ghi dependency trực tiếp và
`requirements-lock.txt` ghi phiên bản toàn bộ dependency đã cài. Bốn profile
Windows dùng Python 3.11 x64; các lock này không phải lock đa nền tảng.
Lock Linux ghi môi trường Huy thực đo, không đại diện toàn bộ dependency Mac của Đức.

## Cài đặt

Cài [uv theo tài liệu chính thức](https://docs.astral.sh/uv/getting-started/installation/),
sau đó chạy tại root repo:

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
# Chỉ khi cần chạy NVIDIA GPU:
python scripts/setup.py --profile gpu
```

Script tạo venv bằng `uv venv` nếu chưa có, cài đúng lock bằng `uv pip sync`,
kiểm dependency với `--strict`. Cache lưu tại `.cache/uv/`. Lần đầu cần mạng;
không cài package vào Python hệ thống.

Kiểm tra mà không cài lại hoặc chạy model:

```powershell
python scripts/setup.py --profile baseline --check
python scripts/setup.py --profile runtime --dry-run --offline
```

Các lệnh uv tương đương, ví dụ với baseline:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe environments/baseline/requirements-lock.txt --strict
uv pip check --python .venv/Scripts/python.exe
```

Chỉ dùng `uv venv` khi chưa có venv. Không cần activate; các lệnh chạy model dùng
interpreter cụ thể để tránh chọn nhầm môi trường. Runtime lock đã bỏ đường dẫn
wheel tuyệt đối `file:///D:/...`, thay bằng pin cùng phiên bản có thể tải lại.

## Linux reproduce

```powershell
docker build -t huy-peer-day09:cpu environments/peer-linux
```

Dockerfile dùng uv ghim phiên bản và Python image ghim digest, cài lock riêng với
`--torch-backend cpu`. [Cách chạy source Đức](../daily/2026-10-09/experiments/01-reproduce/PEER-RUNNING.md).
Việc đổi công cụ cài không được coi là một phép benchmark mới; image đã đo trước
đây vẫn được ghi trong metadata lịch sử.

Nguồn: [uv environments](https://docs.astral.sh/uv/pip/environments/),
[uv lock/sync](https://docs.astral.sh/uv/pip/compile/),
[uv Docker](https://docs.astral.sh/uv/guides/integration/docker/).
