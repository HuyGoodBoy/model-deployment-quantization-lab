# Quantization ResNet50 và DistilBERT trên CPU/GPU

**Ngày:** 08/10/2026. **Trạng thái:** đã có số đo; verifier GPU chưa chạy xác nhận.

Code, environment và evidence được giữ trong module [quantization](../../../../quantization/README.md).

| Thành phần | Đường dẫn từ thư mục gốc repo |
|---|---|
| Runner CPU/GPU | `quantization/run_lab.py`, `quantization/run_gpu.py` |
| Mã nguồn | `quantization/qlab/` |
| Dependency | `quantization/requirements*.txt` |
| Kết quả CPU/GPU | `quantization/results/`, `quantization/results/gpu/` |
| Nguồn model/dataset | `quantization/sources.lock.json` |
| Model và input cục bộ | `quantization/artifacts/`, `quantization/data/` |

Lệnh tái lập chạy từ thư mục `quantization/`, với môi trường tương ứng đã cài:

```powershell
..\.venv\Scripts\python.exe run_lab.py
.\.venv-gpu\Scripts\python.exe run_gpu.py
```

Môi trường CPU hiện tại nằm ở `.venv/` của repo gốc; bản clone mới có thể tạo venv theo README module. Pipeline tạo lại kết quả; lưu riêng phiên cũ trước khi khảo sát cấu hình khác.

- [Báo cáo ngày](../../REPORT.md)
- [Báo cáo đầy đủ](../../../../quantization/bao-cao-ngay-2026-10-08-cpu-gpu.md)
- [Hướng dẫn GPU](../../../../quantization/README-gpu.md)
