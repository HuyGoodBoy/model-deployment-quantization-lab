# Thực nghiệm của ngày

Tạo một thư mục riêng cho mỗi task:

```text
01-ten-task/
  README.md
  src/
  results/<run-id>/
  artifacts/
```

README của task ghi mục tiêu, trạng thái, model/dữ liệu, vị trí code, lệnh chạy, bằng chứng, kết quả và giới hạn. `src/`, `results/`, `artifacts/` chỉ tạo khi cần dùng. Model/dataset/env/cache/build giữ cục bộ; JSON/CSV/Markdown kết quả nhỏ có thể lưu trong Git theo allowlist.
