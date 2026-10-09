# Công việc ngày 09/10/2026

**Người thực hiện:** Huy. **Trạng thái:** đã khảo sát code và artifact; chưa chạy conversion hoặc benchmark mới.

Mục tiêu: đối chiếu môi trường với Đức, giải thích dynamic/static quantization, làm rõ DistilBERT calibrated mixed INT8, thử M-LSD và xây dựng hướng port post-processing detect box lên Android.

- [Task mentor giao và checklist](TASKS.md)
- [Báo cáo tiến độ](REPORT.md)
- [Môi trường và nguồn đối chiếu](ENVIRONMENT.md)

| Nhóm việc | Thư mục | Trạng thái |
|---|---|---|
| Reproduce và đối chiếu môi trường | [01-reproduce](experiments/01-reproduce/README.md) | Đã đọc repo Đức; chưa chạy |
| DistilBERT mixed/strict INT8 | [02-distilbert-int8](experiments/02-distilbert-int8/README.md) | Đã audit artifact cũ; chưa thử strict INT8 |
| Convert và quantize M-LSD | [03-mlsd](experiments/03-mlsd/README.md) | Đã khảo sát nguồn; chưa convert |
| Detect box và hướng port Android | [04-box-postprocess](experiments/04-box-postprocess/README.md) | Có đề xuất; chưa port/đo Android |

Số đo ngày 08/10 chỉ dùng làm baseline có ghi ngày, không gán thành kết quả ngày 09/10. Repo Đức là nguồn chỉ đọc; chưa sửa hoặc chạy code của repo đó.
