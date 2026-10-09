# DistilBERT static mixed và chế độ ép INT8

**Ngày:** 09/10/2026. **Trạng thái:** đã đọc artifact ngày trước, chưa convert đối chứng.

Baseline ở `quantization/`: representative dataset 100 câu, IDs/mask INT32, output INT8, cho phép integer và float builtins. Audit có 372 INT8, 126 INT32, 1 FP32; tensor float là cast attention mask, không phải bằng chứng chủ động giữ attention/LayerNorm FP32.

## Thiết kế thử nghiệm

1. Giữ checkpoint, tokenizer, input và calibration/evaluation giống baseline.
2. Đối chứng cấu hình chỉ cho phép `TFLITE_BUILTINS_INT8`, giữ token IDs/mask INT32; không ép ID từ vựng sang INT8.
3. Nếu lỗi, lưu thông báo converter và operator liên quan; nếu thành công, audit graph và đo output/chất lượng/latency.
4. Khảo sát attention mask với một thay đổi có kiểm soát. Đánh giá sự tương đương FP32 trước khi quantize; không tune bằng evaluation.

Code mới ở `src/`, evidence mới ở `results/<run-id>/`, model ở `artifacts/` khi tạo thực nghiệm. Baseline cũ chỉ đọc và dẫn theo ngày, không ghi đè.

- [Converter baseline](../../../../quantization/qlab/convert.py)
- [Audit](../../../../quantization/results/text/conversion_tflite_static.json)
- [Diagnostic scales](../../../../quantization/results/text/diagnostic_tflite_static.json)
- [Task](../../TASKS.md), [báo cáo](../../REPORT.md)
