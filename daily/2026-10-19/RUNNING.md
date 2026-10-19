# Tái lập ngày 19/10

Chạy từ root repo, dùng `.venv-day09` và dependency đã khóa trong [runtime](../../environments/runtime/requirements-lock.txt). Không có training mới, download mới hay calibration mới.

Đọc [PROTOCOL.md](PROTOCOL.md) trước khi chạy. Cần checkpoint/artifact ngày15–16, dữ liệu local theo manifest và source các ngày trước. Các script từ chối ghi đè kết quả; tái lập trọn bộ trong bản sao dự án có thư mục kết quả/artifact ngày19 chưa tồn tại. Giữ nguyên bằng chứng gốc, không xóa để chọn lại kết quả.

```powershell
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/diagnose.py
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/export_selective.py
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/measure.py
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/regression.py
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/fusion_control.py
.venv-day09/Scripts/python.exe daily/2026-10-19/experiments/selective/src/verify.py
python daily/2026-10-19/experiments/selective/src/write_report.py
```

Chạy tuần tự; không chạy inference/training/profiler khác đồng thời benchmark. CPU 4 intra/1 inter threads, sequential, spinning tắt; graph optimizer ALL trừ đối chứng được ghi rõ. Mỗi model/vòng có session riêng, warm-up16 rồi timed16 ảnh development, 5 vòng xoay/đảo thứ tự, 80 raw timing/model. Timing inference-only, không gồm load, preprocess và decoder. Đối chứng tắt pass QDQ chạy riêng, không lấy số tuyệt đối của nó thay bảng benchmark chính.

`diagnosis.json` chứa warm profile và sai số từng lớp trên cùng input; `exports.json` chứa hai policy, qparams còn giữ và gate export. `selection.json` khóa lựa chọn bằng development/timing; `regression.json` ghi thời điểm bắt đầu inference sau khóa. Tập 54 ảnh đã dùng ngày16, nay chỉ là hồi quy. Bootstrap ảnh ghép cặp seed20261019, 1.000 lần. Không gọi tập này là holdout mới.

Verifier kiểm các hash, metric, raw timing, bootstrap, lựa chọn, qparams/bias/weight ONNX và kernel thực thi. Profiler có thể bỏ weight đã prepack khỏi danh sách input; weight dtype được xác minh theo đúng input slot của node trong graph đã tối ưu, activation dtype và kernel được đọc từ trace. Không suy vị trí weight theo danh sách input profiler bị rút gọn.

`source_archive.json` là bằng chứng source trước dọn lint/format cho lượt chạy gốc. Nếu tái lập từ source hiện tại trong bản sao trống, verifier chấp nhận script hash hiện tại, không cần file archive cũ. Không thay JSON hoặc model ngày15–16. Artifact lớn, raw tensor, trace và model được Git ignore; JSON/CSV nhỏ và code là bằng chứng review.
