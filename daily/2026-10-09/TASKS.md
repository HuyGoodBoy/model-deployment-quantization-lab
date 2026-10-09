# Task ngày 09/10/2026

**Người thực hiện:** Huy. **Nguồn:** yêu cầu mentor giao ngày 09/10/2026.

## 1. Phần chung cho Huy và Đức

1. Reproduce code của người còn lại; chạy model của mình trên môi trường đó, so tốc độ/chất lượng với môi trường của mình và giải thích chênh lệch.
2. Giải thích dynamic/static quantization của ONNX Runtime và TFLite/LiteRT; giải thích vì sao cùng tên phương pháp vẫn cho kết quả khác nhau.
3. Tiếp tục thực nghiệm theo hướng tự đề xuất.
4. Convert model từ `lhwcv/mlsd_pytorch` sang TFLite/LiteRT, so tốc độ với TFLite chính thức trong `navervision/mlsd`, thử quantize và đề xuất port post-processing detect box lên Android.
5. Phần đọc/giải thích thuật toán hạn chế dùng trợ lý AI; người thực hiện tự đọc, viết lại theo ý hiểu và giải thích được từng bước.

## 2. Phần riêng của Huy

Giải thích DistilBERT TFLite calibrated mixed INT8 là gì, khác gì static INT8 và vì sao chọn mixed thay cho chế độ ép integer-only.

## 3. Phần riêng của Đức, phối hợp khi đối chiếu

Khảo sát vì sao TFLite calibration có range không bao phủ tốt dữ liệu evaluation trong kết quả đã báo cáo; thử nhiều subset calibration khác nhau. Chưa mặc định kết luận ONNX miễn nhiễm với vấn đề calibration.

## 4. Checklist và tiêu chí hoàn thành của Huy

### Reproduce

- [x] Đọc đủ 7 file trong repo Đức và cố định revision đối chiếu.
- [x] Ghi khác biệt model, preprocessing, runtime, quantizer và giao thức đo.
- [ ] Chạy reproduce trong bản làm việc/môi trường riêng sau khi có phạm vi thực thi được xác nhận.
- [ ] Chạy model Huy với cấu hình công cụ đối chiếu, lưu môi trường và evidence.
- [ ] Phân biệt tác động phần cứng với software/runtime/thread bằng các đối chứng phù hợp.

**Tiêu chí:** có output và latency thực đo, model/input hash, dependency thực cài; không coi đọc code là reproduce thành công.

### Dynamic/static và DistilBERT

- [x] Phân biệt thời điểm calibration với phạm vi dtype mixed/full integer.
- [x] Đọc cấu hình converter và audit graph DistilBERT đã tạo ngày 08/10.
- [x] Xác định tensor FP32 còn lại là cast attention mask `[1,64]`.
- [ ] Thử chế độ chỉ cho phép operator INT8 với token IDs/mask giữ INT32.
- [ ] Ghi lỗi converter nếu thất bại; nếu thành công, audit graph, đánh giá output/chất lượng và latency.
- [ ] Khảo sát vùng attention mask bằng thay đổi có kiểm soát; xác minh FP32 trước khi quy nguyên nhân cho quantization.

**Tiêu chí:** giải thích theo code và graph thực tế; không khẳng định strict INT8 không hỗ trợ khi chưa thử.

### M-LSD

- [x] Khảo sát repo PyTorch và repo TensorFlow/TFLite chính thức.
- [ ] Cố định checkpoint/revision, input, preprocessing và output contract.
- [ ] Chạy PyTorch reference, convert Tiny 512×512 sang TFLite FP32.
- [ ] Đánh giá sai khác so chính checkpoint PyTorch.
- [ ] Thử FP16 weights, dynamic range và static INT8 với calibration ảnh thật.
- [ ] So với TFLite chính thức, ghi khác biệt weights, RGBA và phần decoding nằm trong graph.

**Tiêu chí:** so cùng phạm vi inference/decoding, lưu số đo riêng cho graph và pipeline; không dùng output checkpoint khác làm chuẩn sai số conversion.

### Detect box và báo cáo

- [ ] Huy tự đọc, diễn giải các bước đoạn thẳng → gộp đường → giao điểm → góc → bốn cạnh → chấm điểm.
- [ ] Đề xuất contract và lựa chọn Kotlin hoặc C++/JNI cho Android.
- [ ] Đối chiếu thuật toán port với các ca hình học đã biết và input thực tế.
- [ ] Tổng hợp kết quả thực đo, giới hạn, code và bằng chứng vào `REPORT.md`.
- [ ] Tạo bản chat cục bộ và gói code/kết quả cần bàn giao khi thực nghiệm hoàn tất.
