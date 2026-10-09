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
- [ ] Reproduce đầy đủ ResNet18/pipeline Đức; còn thiếu converter tương thích Windows hoặc môi trường Linux/artifact từ Đức.
- [x] Chạy model Huy với runtime đối chiếu trên Windows, lưu môi trường và evidence.
- [x] Đối chứng cùng model/input/thread, ghi rõ chưa tái tạo Mac hoặc cô lập mọi dependency/kernel.

**Tiêu chí:** có output và latency thực đo, model/input hash, dependency thực cài; không coi đọc code là reproduce thành công.

### Dynamic/static và DistilBERT

- [x] Phân biệt thời điểm calibration với phạm vi dtype mixed/full integer.
- [x] Đọc cấu hình converter và audit graph DistilBERT đã tạo ngày 08/10.
- [x] Xác định tensor FP32 còn lại là cast attention mask `[1,64]`.
- [x] Thử chế độ chỉ cho phép operator INT8 với token IDs/mask giữ INT32; conversion thành công.
- [x] Audit graph strict, đánh giá output/chất lượng và latency; float còn ở CAST→QUANTIZE.
- [x] Thử mask sentinel 1e4/1e2, xác minh FP32 trên 100 mẫu rồi quantize; chưa khắc phục hết giảm chất lượng INT8.

**Tiêu chí:** giải thích theo code và graph thực tế; không khẳng định strict INT8 không hỗ trợ khi chưa thử.

### M-LSD

- [x] Khảo sát repo PyTorch và repo TensorFlow/TFLite chính thức.
- [x] Cố định checkpoint/revision/hash, input, preprocessing và output contract.
- [x] Chạy PyTorch reference, convert cùng weights qua bridge Keras sang TFLite FP32.
- [x] Đánh giá bridge FP32 so chính checkpoint; ngưỡng allclose chặt không đạt, lưu riêng gate theo sai số tọa độ.
- [x] Convert FP16 weights, dynamic range và static INT8 với 32 ảnh calibration thật, tách 10 evaluation.
- [x] So decoded_fp32 với TFLite chính thức trên hai runtime, ghi khác biệt weights, RGBA và decoding.

**Tiêu chí:** so cùng phạm vi inference/decoding, lưu số đo riêng cho graph và pipeline; không dùng output checkpoint khác làm chuẩn sai số conversion.

### Detect box và báo cáo

- [ ] Huy tự đọc, diễn giải các bước đoạn thẳng → gộp đường → giao điểm → góc → bốn cạnh → chấm điểm.
- [x] Đề xuất contract Android, Java/Kotlin trước, C++/JNI sau profiling.
- [x] Prototype qua 5 ca hình học và CSV demo (4 box ứng viên); chưa chứng minh tương đương NAVER.
- [x] Tổng hợp số đo, giới hạn, code và bằng chứng vào REPORT; verifier offline đã pass 40 job chính + 2 thread controls.
- [x] Tạo báo cáo chat cục bộ và script ZIP có allowlist/inventory SHA256; chat/model/cache/docs không commit.

## 5. Những phần chưa hoàn tất

- Reproduce đầy đủ ResNet18/pipeline converter Đức trên môi trường tương thích và đối chiếu Mac M1 Pro.
- Huy tự đọc/viết lại thuật toán theo yêu cầu mentor; tài liệu có hỗ trợ AI chưa thay phần này.
- Port Hough merge/scoring tương đương NAVER, kiểm bằng fixtures và benchmark M-LSD/box trên Android thật.
- Đánh giá line/box có nhãn; kiểm soát power/nhiệt độ và lặp nhiều session trước kết luận latency ổn định.
