# Báo cáo tiến độ ngày 09/10/2026

**Người thực hiện:** Huy. **Trạng thái:** khảo sát code và artifact; chưa có số đo conversion/inference mới.

## 1. Công việc đã thực hiện

- Đọc đủ 7 file của repo Đức tại revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`; không chỉnh sửa hoặc thực thi repo.
- Đối chiếu thiết kế ResNet18/PyTorch/LiteRT với bài ResNet50/DistilBERT của Huy.
- Đọc converter và FlatBuffer DistilBERT của ngày 08/10 để xác định ý nghĩa calibrated mixed INT8.
- Khảo sát repo M-LSD PyTorch/TensorFlow, input/output và hướng port post-processing.
- Tổ chức task, báo cáo và liên kết thực nghiệm theo ngày.

## 2. Kết quả đọc code Đức

| Nội dung | Đức | Baseline Huy ngày 08/10 |
|---|---|---|
| Model ảnh | ResNet18 torchvision | ResNet50 Keras |
| Output | Logits | Xác suất softmax |
| TFLite export | litert_torch | TensorFlow TFLiteConverter |
| Quantizer TFLite | AI Edge Quantizer recipes | tf.lite converter 2.15.1 |
| Calibration / evaluation | 100 / 300 ảnh | 100 / 100 ảnh |
| Chọn dữ liệu | Theo tên file đã sắp xếp | Theo seed |
| Máy / thread theo báo cáo | Mac M1 Pro / 4 | Windows i7-12700H / 1 |
| Warm-up / lượt đo | 10 / 100 | 30 / 200 |

Nguồn: [prepare](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q1_prepare.py), [quantize TFLite](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q3_quant_tflite.py), [evaluate/benchmark](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q4_evaluate.py).

Trong thư mục đã đọc chưa có model, raw output, latency samples hoặc code chẩn đoán range/tie. Nhận định output bị kẹp và 62/300 ảnh hòa điểm là thông tin trong [báo cáo Đức](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/BAO_CAO.md), chưa được reproduce độc lập.

## 3. Giải thích DistilBERT calibrated mixed INT8

**Calibrated mixed INT8 là một cấu hình static quantization.** Static/dynamic mô tả lúc tính scale và zero-point của activation; mixed/full integer mô tả phạm vi dtype/operator của graph. Hai khía cạnh độc lập.

Code Huy dùng representative dataset và cho phép cả `TFLITE_BUILTINS_INT8` lẫn `TFLITE_BUILTINS`, giữ token IDs/mask INT32, output INT8. Audit cũ có 372 tensor INT8, 126 INT32 và 1 FP32; đọc FlatBuffer xác định tensor float là `distilbert/Cast`, shape `[1,64]`.

INT32 token IDs, shape/index hoặc bias không tự chứng minh có float fallback. Không có bằng chứng graph này được thiết kế để giữ attention/LayerNorm FP32. Cấu hình cho phép mixed giúp converter có thể giữ float khi cần, nhưng chưa có đối chứng ép INT8 để kết luận mixed bắt buộc.

Nguồn: [converter](../../quantization/qlab/convert.py), [audit conversion](../../quantization/results/text/conversion_tflite_static.json), [diagnostic](../../quantization/results/text/diagnostic_tflite_static.json).

## 4. Dynamic/static giữa ONNX Runtime và TFLite/LiteRT

- Dynamic: weights được quantize trước; activation của operator hỗ trợ có thể được quantize động. Phạm vi operator, biểu diễn graph và kernel không giống nhau giữa các công cụ.
- Static: scale/zero-point activation lấy từ calibration và cố định khi inference; graph có thể còn float và các operator integer không phải INT8.
- Trong bài Huy, ONNX text chỉ chọn MatMul/Gemm; TFLite bao phủ graph rộng hơn. Đức dùng AI Edge Quantizer trên file TFLite FP32, khác pipeline TensorFlow converter của Huy.
- Muốn giải thích chênh lệch phải đối chiếu operator/dtype, calibration, granularity, fusion, delegate/provider, input và phần cứng; tên dynamic/static chưa đủ.

Metadata DistilBERT ghi scale khoảng `3,92e27` quanh attention mask và code TF có sentinel `1e30`. Đây là dấu hiệu cần kiểm chứng bằng đối chứng, chưa chứng minh TFLite calibration sai hoặc ONNX không gặp vấn đề tương tự.

Nguồn chính thức: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [LiteRT post-training quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization), [AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer).

## 5. Thiết kế M-LSD và post-processing

Chọn Tiny 512×512 làm phiên đầu. Sai số conversion sẽ so với chính checkpoint PyTorch. So tốc độ với TFLite chính thức phải ghi rõ khác biệt checkpoint, RGBA, normalization và phần decoding đã nằm trong graph. Đo riêng inference, decoding và detect box.

Hướng đọc thuật toán: đoạn thẳng → gộp đường → giao điểm → góc hợp lệ → chu trình bốn cạnh → chấm điểm. Cân nhắc Kotlin hoặc C++/JNI, buffer tái sử dụng và đối chiếu hình học với dữ liệu đã biết. Chưa có code port hoặc phép đo Android cho M-LSD.

Nguồn: [PyTorch M-LSD](https://github.com/lhwcv/mlsd_pytorch), [M-LSD chính thức](https://github.com/navervision/mlsd), [export](https://github.com/navervision/mlsd/blob/master/frozen_models.py), [post-processing](https://github.com/navervision/mlsd/blob/master/utils.py).

## 6. Kết quả thực nghiệm mới

Chưa chạy reproduce, convert M-LSD, quantize bổ sung hoặc benchmark ngày 09/10. Không có bảng số đo mới. Baseline ngày 08/10 được dẫn tại [báo cáo ngày trước](../2026-10-08/REPORT.md).

## 7. Việc tiếp theo và bàn giao

Các bước thực thi và tiêu chí hoàn thành nằm trong [TASKS.md](TASKS.md). Code mới/kết quả mới sẽ đặt theo từng task trong [experiments](README.md); môi trường và nguồn đối chiếu ghi tại [ENVIRONMENT.md](ENVIRONMENT.md). Báo cáo này được cập nhật sau khi có evidence thực đo.
