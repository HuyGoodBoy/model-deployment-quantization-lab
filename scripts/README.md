# Lệnh chung

Chạy từ root repo:

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
.\.venv\Scripts\python.exe scripts/run_mobilenet.py --strict
```

- `setup.py`: tạo/đồng bộ hoặc kiểm môi trường bằng uv; [các profile](../environments/README.md).
- `run_mobilenet.py`: entry point MobileNetV2 ngày 07, giữ module `lab/` và đường dẫn output cũ.

Pipeline ResNet50/DistilBERT: `quantization/run_lab.py`, `quantization/run_gpu.py`.
Code hôm nay và runner riêng: [daily/2026-10-09](../daily/2026-10-09/README.md).
