# Báo cáo và thực nghiệm theo ngày

**Người thực hiện:** Huy.

| Ngày | Nội dung | Task | Báo cáo |
|---|---|---|---|
| [07/10/2026](2026-10-07/README.md) | Framework deploy, MobileNetV2 ONNX/TFLite FP32 trên PC | [Danh sách](2026-10-07/TASKS.md) | [Báo cáo](2026-10-07/REPORT.md) |
| [08/10/2026](2026-10-08/README.md) | Quantization ResNet50/DistilBERT, GPU và MobileNetV2 Android | [Danh sách](2026-10-08/TASKS.md) | [Báo cáo](2026-10-08/REPORT.md) |
| [09/10/2026](2026-10-09/README.md) | Runtime cũ/mới, DistilBERT INT8, M-LSD và prototype detect box | [Danh sách](2026-10-09/TASKS.md) | [Báo cáo](2026-10-09/REPORT.md) |

## Cấu trúc một ngày

```text
daily/YYYY-MM-DD/
  README.md                 Mục tiêu, trạng thái và liên kết bàn giao
  TASKS.md                  Task mentor giao, checklist, tiêu chí hoàn thành
  REPORT.md                 Phương pháp, kết quả, nhận xét, giới hạn, hướng tiếp theo
  ENVIRONMENT.md            Cấu hình và nguồn thông tin môi trường
  experiments/
    01-ten-task/
      README.md             Phạm vi, cách chạy, vị trí code và bằng chứng
      src/                  Code mới riêng của task, khi có
      results/              Kết quả mới riêng của task, khi có
      artifacts/            Model được tạo, không commit
  REPORT-CHAT.md            Bản chat cục bộ, không commit
```

`src/`, `results/`, `artifacts/` chỉ được tạo khi thực nghiệm cần dùng. Một task có một thư mục riêng; không đặt code hoặc kết quả mới lẫn với task khác. Phiên chạy bổ sung dùng thư mục `results/<run-id>/`, không ghi đè số đo của phiên trước.

## Quy tắc báo cáo

1. Tên ngày dùng `YYYY-MM-DD` theo múi giờ Asia/Saigon. Ngày trong file và trong thư mục phải trùng nhau.
2. `TASKS.md` ghi rõ phần chung, phần riêng của Huy và phần phối hợp với Đức.
3. `REPORT.md` chỉ ghi số đo đã chạy; trạng thái chưa chạy, thiếu dữ liệu hoặc thất bại được nêu riêng.
4. Mỗi bảng kết quả dẫn tới nguồn code, môi trường và dữ liệu kiểm chứng. Kết quả của người khác được ghi rõ nguồn, không gán thành số đo reproduce.
5. So sánh latency phải ghi model, input, runtime, thread, provider/delegate, batch, warm-up, số lượt và phạm vi timer. Độ chính xác luôn ghi tập đánh giá và nhãn tham chiếu.
6. Cuối ngày cập nhật bảng chỉ mục này và README gốc. Ngày cũ chỉ sửa lỗi diễn giải có ghi chú; thực nghiệm mới lưu trong ngày mới.

## Code và kết quả các ngày trước

Pipeline đã thực hiện được giữ ở vị trí hiện tại để bảo toàn lệnh chạy và các đường dẫn/hash trong evidence:

- `lab/`, `run_lab.py`, `results/`: MobileNetV2 PC.
- `android/`, `results/android/`: instrumentation Android MobileNetV2.
- `quantization/`: ResNet50, DistilBERT và ONNX CUDA.

Thư mục ngày dẫn tới các module này, không sao chép môi trường ảo, model hoặc dataset. Tiện ích mới chỉ chuyển thành module dùng chung khi có nhiều thực nghiệm thực sự dùng lại.

## Tạo ngày tiếp theo

Sao chép [_template](../_template/README.md) vào `daily/YYYY-MM-DD/` rồi thay các trường ngày, task, trạng thái và đường dẫn. Ví dụ, chạy từ thư mục gốc repo:

```powershell
$newReportDay = '2026-10-10'
if (Test-Path -LiteralPath "daily/$newReportDay") { throw 'Thu muc ngay da ton tai' }
Copy-Item -LiteralPath '_template' -Destination "daily/$newReportDay" -Recurse
```

Mẫu nằm ở thư mục gốc `_template/` để không bị nhầm với một ngày thực nghiệm.
