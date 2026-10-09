# Script hỗ trợ ngày 09/10/2026

Chạy tại root repo. Code model nằm trong `../experiments/`; dependency cài bằng
[uv](../../../environments/README.md).

| Script | Công việc |
|---|---|
| `run_matrix.py` | Điều phối conversion và benchmark Windows tuần tự |
| `post_checks.py` | Đối chứng thread, audit delegate và Java smoke test |
| `environment_probe.py` | Ghi package thực cài; không import model/runtime |
| `verify_results.py` | Kiểm hash, metric và timing offline; cần raw output/artifact cục bộ |
| `build_report.py` | Tạo REPORT từ JSON đã đo |
| `build_peer_report.py` | Tạo báo cáo reproduce Markdown từ JSON và báo cáo Đức ghim revision |

Tạo báo cáo từ số đã có:

```powershell
python daily/2026-10-09/tools/build_report.py
python daily/2026-10-09/tools/build_peer_report.py
```

Không chạy inference hoặc đóng ZIP trong các bước này. Generator peer cần
checkout Đức ghim revision theo [PEER-RUNNING.md](../experiments/01-reproduce/PEER-RUNNING.md).
Probe ghi file riêng trong `results/environment-probe/`, không ghi đè phiên cũ.
Metadata lịch sử giữ đường dẫn/hash lúc đo, kể cả khi script được chuyển thư mục.

[Cách chạy đầy đủ](../RUNNING.md) · [Mục lục code](../README.md)
