# Công việc ngày 09/10/2026

**Người thực hiện:** Huy. **Trạng thái:** đã triển khai thực nghiệm CPU và prototype hình học; kết quả/giới hạn trong báo cáo.

Mục tiêu: đối chiếu môi trường với Đức, giải thích dynamic/static quantization, làm rõ DistilBERT calibrated mixed INT8, thử M-LSD và xây dựng hướng port post-processing detect box lên Android.

- [Task mentor giao và checklist](TASKS.md)
- [Báo cáo thực nghiệm](REPORT.md)
- [Reproduce source Đức: bảng Mac/Linux và model Huy trên môi trường Linux](PEER-REPRODUCTION.md)
- [Môi trường và nguồn đối chiếu](ENVIRONMENT.md)
- [Hướng dẫn chạy lại và bàn giao](RUNNING.md)

| Nhóm việc | Thư mục | Trạng thái |
|---|---|---|
| Reproduce và đối chiếu môi trường | [01-reproduce](experiments/01-reproduce/README.md) | Đã chạy 6/6 ResNet18 từ source Đức trên Linux và 12 cấu hình model Huy trong môi trường đó; hardware/OS ghi riêng |
| DistilBERT mixed/strict INT8 | [02-distilbert-int8](experiments/02-distilbert-int8/README.md) | Strict convert được; thêm mask ablation, FP32 gate |
| Convert và quantize M-LSD | [03-mlsd](experiments/03-mlsd/README.md) | Cùng weights qua bridge Keras, FP32/FP16/dynamic/static/decoded |
| Detect box và hướng port Android | [04-box-postprocess](experiments/04-box-postprocess/README.md) | Java core qua 5 ca hình học; chưa đầy đủ Hough/scoring hoặc Android |

Số đo ngày 08/10 chỉ dùng làm baseline có ghi ngày, không gán thành kết quả ngày 09/10. Source Đức được ghim revision, mount read-only và giữ nguyên; code được thực thi với output riêng của Huy. Không yêu cầu cùng phần cứng Mac để reproduce.

## Cấu trúc code nộp

```text
2026-10-09/
  REPORT.md, PEER-REPRODUCTION.md   Kết quả và phân tích
  RUNNING.md, ENVIRONMENT.md        Cách chạy và dependency
  TASKS.md                         Checklist task mentor
  tools/                           Điều phối, kiểm chứng và tạo báo cáo
  experiments/
    01-reproduce/                  Reproduce Đức + so runtime model Huy
    02-distilbert-int8/             Strict INT8 và attention-mask ablation
    03-mlsd/                       Reference → Keras bridge → TFLite, benchmark
    04-box-postprocess/            Prototype hình học Java
```

Mỗi thí nghiệm giữ `src/` riêng. `results/<run-id>/` lưu số đo; `artifacts/` chứa
model cục bộ. [Mục lục script hỗ trợ](tools/README.md) giải thích từng lệnh.
Đường dẫn trong metadata cũ phản ánh lúc đo, không sửa lại theo cấu trúc mới.


## Đọc code theo thứ tự

| Task | File chính | Luồng xử lý |
|---|---|---|
| Reproduce Đức | `01-reproduce/src/reproduce_peer.py` | Kiểm source hash → prepare → convert → evaluate/benchmark |
| So môi trường Huy | `01-reproduce/src/runtime_compare.py` | Cùng model/input → runtime cũ/mới → metric và latency |
| Model Huy trên Linux | `01-reproduce/src/run_huy_peer_environment.py` | Chạy tuần tự 12 cấu hình, ghi trạng thái mỗi job |
| DistilBERT INT8 | `02-distilbert-int8/src/run.py` | Convert strict/mask ablation → audit → measure |
| M-LSD | `03-mlsd/src/prepare_torch.py` → `port_keras.py` → `measure.py` | Reference → bridge/quantization → so output và tốc độ |
| Box | `04-box-postprocess/src/BoxPostProcessor.java` | Line segment → giao điểm → tứ giác → score |

Đường dẫn trong bảng tính từ `experiments/`. Audit/verifier chạy riêng sau benchmark;
script tạo báo cáo nằm trong `tools/`. Dependency tập trung ở [environments](../../environments/README.md).

## Bàn giao Git

Đọc `REPORT.md` trước, sau đó xem code từng task trong `experiments/`. Dependency
và lock dùng [uv profiles](../../environments/README.md); [lệnh chạy](RUNNING.md)
tính từ root repo. JSON/CSV kết quả giữ trên Git; raw output/model/dataset/cache
và học liệu HTML giữ cục bộ. Metadata cũ giữ nguyên đường dẫn/hash tại thời điểm đo.
