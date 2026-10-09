# MobileNetV2 FP32 trên PC

**Ngày thực nghiệm:** 07/10/2026. **Trạng thái:** hoàn thành trong phạm vi hai ảnh mẫu.

| Thành phần | Đường dẫn từ thư mục gốc repo |
|---|---|
| Runner | `run_lab.py` |
| Mã nguồn | `lab/` |
| Dependency | `requirements.txt`, `requirements-lock.txt` |
| Kết quả | `results/` |
| Model / input cục bộ | `artifacts/`, `data/` |

Chạy từ thư mục gốc repo, trong môi trường CPU đã cài dependency:

```powershell
.\.venv\Scripts\python.exe scripts/run_mobilenet.py --strict
```

Lệnh trên tạo lại kết quả ở `results/`. Khi khảo sát cấu hình khác, lưu riêng phiên kết quả trước khi chạy; không dùng số đo cũ với model/input đã thay đổi.

- [Báo cáo ngày](../../REPORT.md)
- [Code runner](../../../../scripts/run_mobilenet.py)
- [Báo cáo chi tiết](../../../../results/report.md)
