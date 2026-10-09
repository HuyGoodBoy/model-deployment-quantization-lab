# Công việc ngày 09/10/2026

**Người thực hiện:** Huy. **Trạng thái:** đã triển khai thực nghiệm CPU và prototype hình học; kết quả/giới hạn trong báo cáo.

Mục tiêu: đối chiếu môi trường với Đức, giải thích dynamic/static quantization, làm rõ DistilBERT calibrated mixed INT8, thử M-LSD và xây dựng hướng port post-processing detect box lên Android.

- [Task mentor giao và checklist](TASKS.md)
- [Báo cáo thực nghiệm](REPORT.md)
- [Môi trường và nguồn đối chiếu](ENVIRONMENT.md)
- [Hướng dẫn chạy lại và bàn giao](RUNNING.md)

| Nhóm việc | Thư mục | Trạng thái |
|---|---|---|
| Reproduce và đối chiếu môi trường | [01-reproduce](experiments/01-reproduce/README.md) | Đối chứng model Huy/runtime Windows; chưa reproduce đầy đủ ResNet18/Mac/converter Đức |
| DistilBERT mixed/strict INT8 | [02-distilbert-int8](experiments/02-distilbert-int8/README.md) | Strict convert được; thêm mask ablation, FP32 gate |
| Convert và quantize M-LSD | [03-mlsd](experiments/03-mlsd/README.md) | Cùng weights qua bridge Keras, FP32/FP16/dynamic/static/decoded |
| Detect box và hướng port Android | [04-box-postprocess](experiments/04-box-postprocess/README.md) | Java core qua 5 ca hình học; chưa đầy đủ Hough/scoring hoặc Android |

Số đo ngày 08/10 chỉ dùng làm baseline có ghi ngày, không gán thành kết quả ngày 09/10. Repo Đức là nguồn chỉ đọc; chưa sửa hoặc chạy code của repo đó.
