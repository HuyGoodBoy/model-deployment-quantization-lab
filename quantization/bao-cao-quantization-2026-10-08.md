# Báo cáo quantization: mô hình ảnh và text

**Ngày công việc: 08/10/2026.**

## 1. Mục tiêu công việc

Tìm hiểu và thực nghiệm post-training quantization bằng TFLite và ONNX Runtime. Dùng hai pretrained model khác bài MobileNetV2: ResNet50 cho CV và DistilBERT fine-tune SST-2 cho NLP. Đánh giá độ sai khác output, accuracy/F1, dung lượng và latency trên CPU máy tính; đóng ZIP kèm code và bằng chứng.

## 2. Kiến thức đã tìm hiểu

- Quantization biểu diễn giá trị thực bằng số nguyên: `q = clip(round(r/scale)+zero_point)`, `r_approx = scale*(q-zero_point)`. Phải clip trước cast để tránh wraparound INT8.
- Static PTQ lấy scale/zero-point từ calibration set; dynamic quantization tính tham số activation trong lúc inference cho operator được hỗ trợ. ONNX Runtime khuyến nghị static cho CNN và dynamic cho Transformer/RNN.
- TFLite dynamic range quantize weights; FP16 chủ yếu giảm storage, CPU có thể dequantize weights về FP32. Full INT8 cần representative dataset và kernels phù hợp.
- QDQ chèn QuantizeLinear/DequantizeLinear vào graph ONNX; việc fusion/chạy kernel INT8 phụ thuộc runtime và provider. Không mặc định mọi operator trong model đã chuyển INT8.
- Token ID/attention mask là chỉ số INT32, không phải activation FP32 để đổi trực tiếp thành INT8. Transformer có thể còn operator float; báo mixed INT8 nếu đúng graph thực tế.

Nguồn: [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [TFLite PTQ](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization).

## 3. Model, dữ liệu và điều kiện thử nghiệm

### ResNet50 — phân loại ảnh

- Pretrained model: Keras ResNet50; 25,636,712 parameters; không train lại.
- Dataset: Imagenette 160px; 100 calibration từ train, 100 evaluation từ val/validation, seed 20261008. Hai tập không giao nhau theo hash ảnh/nội dung câu.
- Preprocess: EXIF -> RGB -> bilinear resize 224x224 -> BGR -> subtract [103.939,116.779,123.68].
- Output để so số học: softmax probabilities. Dùng cùng canonical input giữa các runtime.
- CPU, batch=1, intra-op=1, inter-op=1 nếu API cho phép; 30 warm-up và 200 lượt đo. Mỗi biến thể ở process riêng, chạy nối tiếp.
- Timer: inference API including in-memory copy, Python overhead, input/output quantize/dequantize when needed; excludes load, image preprocessing/tokenization and disk I/O.
- Máy: Windows-10-10.0.26200-SP0; CPU: 12th Gen Intel(R) Core(TM) i7-12700H; Python 3.11.9.
- Phiên bản: TensorFlow 2.15.1, tf2onnx 1.16.1, ONNX Runtime 1.20.1; NLP thêm Transformers 4.38.2.

- ResNet50 dùng BGR trừ mean ImageNet; không dùng normalization [-1,1] của MobileNetV2. Accuracy dự đoán trên đủ 1.000 lớp, không mask về 10 lớp Imagenette.

### DistilBERT — sentiment tiếng Anh

- Pretrained model: DistilBERT SST-2; 66,955,010 parameters; không train lại.
- Dataset: SST-2; 100 calibration từ train, 100 evaluation từ val/validation, seed 20261008. Hai tập không giao nhau theo hash ảnh/nội dung câu.
- Preprocess: original uncased WordPiece tokenizer; right padding/truncation to 64 tokens.
- Output để so số học: logits. Dùng cùng canonical input giữa các runtime.
- CPU, batch=1, intra-op=1, inter-op=1 nếu API cho phép; 30 warm-up và 200 lượt đo. Mỗi biến thể ở process riêng, chạy nối tiếp.
- Timer: inference API including in-memory copy, Python overhead, input/output quantize/dequantize when needed; excludes load, image preprocessing/tokenization and disk I/O.
- Máy: Windows-10-10.0.26200-SP0; CPU: 12th Gen Intel(R) Core(TM) i7-12700H; Python 3.11.9.
- Phiên bản: TensorFlow 2.15.1, tf2onnx 1.16.1, ONNX Runtime 1.20.1; NLP thêm Transformers 4.38.2.

- Input token INT32 [1,64]; truncate 0/100 câu evaluation. Nhãn 0=negative, 1=positive; chỉ tiếng Anh.

## 4. Sai khác output và accuracy

So với TensorFlow FP32 của chính model đó. Accuracy dùng nhãn thật; agreement là tỷ lệ nhãn dự đoán trùng model gốc. SNR = `10*log10(sum(reference²)/sum(error²))`, cộng trên toàn bộ output của 100 mẫu.

### ResNet50 — phân loại ảnh

| Biến thể | MAE | RMSE | Max abs | SNR dB | Cosine | Accuracy | Top-5 accuracy | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 0 | 0 | 0 | inf | 1.000000000000 | 73.0% | 90.0% | 100.0% |
| ONNX FP32 | 2.57587e-09 | 5.9504e-08 | 5.126e-06 | 112.398 | 0.999999999997 | 73.0% | 90.0% | 100.0% |
| ONNX dynamic INT8 | 1.12665e-05 | 0.000238491 | 0.0192766 | 40.339 | 0.999953754432 | 73.0% | 90.0% | 100.0% |
| ONNX static INT8 QDQ | 0.000121562 | 0.00281662 | 0.321448 | 18.894 | 0.993549818483 | 70.0% | 93.0% | 92.0% |
| TFLite FP32 | 2.28759e-09 | 5.04809e-08 | 5.75185e-06 | 113.826 | 0.999999999998 | 73.0% | 90.0% | 100.0% |
| TFLite dynamic range | 4.86362e-05 | 0.00106669 | 0.111158 | 27.328 | 0.999079891739 | 73.0% | 90.0% | 97.0% |
| TFLite FP16 weights | 1.45475e-06 | 3.18385e-05 | 0.0039162 | 57.829 | 0.999999175832 | 73.0% | 90.0% | 100.0% |
| TFLite static INT8 | 0.000127302 | 0.00231846 | 0.332584 | 20.585 | 0.995684830382 | 74.0% | 90.0% | 91.0% |

Các bản FP32 phải đạt allclose trước khi diễn giải quantization. Không dùng tolerance FP32 làm tiêu chí thất bại cho INT8. Cosine làm tròn trong bảng; số đầy đủ nằm trong JSON/CSV.

### DistilBERT — sentiment tiếng Anh

| Biến thể | MAE | RMSE | Max abs | SNR dB | Cosine | Accuracy | F1 positive | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 0 | 0 | 0 | inf | 1.000000000000 | 91.0% | 91.1% | 100.0% |
| ONNX FP32 | 1.38223e-06 | 2.13386e-06 | 1.30236e-05 | 124.677 | 1.000000000000 | 91.0% | 91.1% | 100.0% |
| ONNX dynamic INT8 | 0.120432 | 0.22258 | 1.11057 | 24.310 | 0.998147311663 | 90.0% | 90.0% | 99.0% |
| ONNX static INT8 QDQ | 1.05498 | 1.134 | 2.8138 | 10.168 | 0.985065947302 | 92.0% | 92.0% | 97.0% |
| TFLite FP32 | 1.56581e-06 | 2.59228e-06 | 1.62125e-05 | 122.986 | 1.000000000000 | 91.0% | 91.1% | 100.0% |
| TFLite dynamic range | 0.187205 | 0.321403 | 1.59264 | 21.119 | 0.996341773647 | 90.0% | 90.2% | 97.0% |
| TFLite FP16 weights | 0.000510662 | 0.00123668 | 0.00893962 | 69.415 | 0.999999943692 | 91.0% | 91.1% | 100.0% |
| TFLite calibrated mixed INT8 | 3.46346 | 3.64078 | 4.8464 | 0.036 | 0.101077624920 | 52.0% | 66.7% | 55.0% |

Các bản FP32 phải đạt allclose trước khi diễn giải quantization. Không dùng tolerance FP32 làm tiêu chí thất bại cho INT8. Cosine làm tròn trong bảng; số đầy đủ nằm trong JSON/CSV.

## 5. Dung lượng model và tốc độ

Tỷ số dung lượng và speedup tính với FP32 **cùng runtime/cùng model**; speedup = median_FP32 / median_variant. Dung lượng file không phải RAM.

### ResNet50 — phân loại ảnh

| Biến thể | MiB | Giảm size × | Mean ms | Median ms | P95 ms | Speedup × |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 98.34 | 1.00 | 341.623 | 420.106 | 588.649 | 1.00 |
| ONNX FP32 | 97.42 | 1.00 | 104.065 | 80.908 | 311.292 | 1.00 |
| ONNX dynamic INT8 | 91.58 | 1.06 | 87.039 | 81.370 | 86.078 | 0.99 |
| ONNX static INT8 QDQ | 24.94 | 3.91 | 57.222 | 56.998 | 58.810 | 1.42 |
| TFLite FP32 | 97.42 | 1.00 | 378.409 | 408.996 | 573.731 | 1.00 |
| TFLite dynamic range | 24.76 | 3.93 | 1040.564 | 1006.991 | 1409.609 | 0.41 |
| TFLite FP16 weights | 48.74 | 2.00 | 403.964 | 424.462 | 528.457 | 0.96 |
| TFLite static INT8 | 25.07 | 3.89 | 333.706 | 326.472 | 396.072 | 1.25 |

### DistilBERT — sentiment tiếng Anh

| Biến thể | MiB | Giảm size × | Mean ms | Median ms | P95 ms | Speedup × |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 255.54 | 1.00 | 72.823 | 71.548 | 75.342 | 1.00 |
| ONNX FP32 | 254.27 | 1.00 | 68.897 | 68.294 | 72.665 | 1.00 |
| ONNX dynamic INT8 | 131.42 | 1.93 | 91.365 | 108.638 | 173.225 | 0.63 |
| ONNX static INT8 QDQ | 131.51 | 1.93 | 232.336 | 245.352 | 302.127 | 0.28 |
| TFLite FP32 | 254.19 | 1.00 | 384.040 | 409.824 | 490.950 | 1.00 |
| TFLite dynamic range | 63.93 | 3.98 | 513.549 | 429.866 | 932.933 | 0.95 |
| TFLite FP16 weights | 127.16 | 2.00 | 336.839 | 344.053 | 415.787 | 1.19 |
| TFLite calibrated mixed INT8 | 63.75 | 3.99 | 275.623 | 268.074 | 321.192 | 1.53 |

## 6. Kết luận và giới hạn

- ResNet50 — phân loại ảnh: bản quantized có median thấp nhất trong lượt đo là ONNX static INT8 QDQ, 56.998 ms; accuracy 70.0%, chênh -3.0 điểm phần trăm với TensorFlow gốc. Tốc độ này cần được cân nhắc cùng chất lượng.
- DistilBERT — sentiment tiếng Anh: bản quantized có median thấp nhất trong lượt đo là ONNX dynamic INT8, 108.638 ms; accuracy 90.0%, chênh -1.0 điểm phần trăm với TensorFlow gốc. Tốc độ này cần được cân nhắc cùng chất lượng.
- **Không chọn TFLite mixed INT8 cho text trong cấu hình này:** accuracy 52.0% so với 91.0% của model gốc (-39.0 điểm phần trăm), SNR 0.036 dB. Convert thành công và chạy nhanh hơn không có nghĩa model đủ chất lượng.
- Text TFLite FP16 weights là lựa chọn đáng thử tiếp nếu cần giảm storage: 127.16 MiB, accuracy 91.0%, SNR 69.415 dB, speedup 1.19× so với TFLite FP32. ONNX FP32 vẫn có latency thấp hơn các bản ONNX quantized ở lượt đo này.
- Kiểm tra FlatBuffer text static thấy 21 tensor có scale > 10⁶, lớn nhất 3.922e+27, xuất hiện quanh attention mask (`attention/mul_1`, `sub`, `add`, `MatMul`). Dải mask âm rất lớn có thể làm bước lượng tử quá thô và mất thông tin attention; đây là **suy luận chẩn đoán từ metadata**, chưa chứng minh nguyên nhân duy nhất bằng thí nghiệm loại trừ. Bằng chứng: `results/text/diagnostic_tflite_static.json`; chạy lại `python -m qlab.diagnose` sau khi tạo model.
- Audit dtype/operator được lưu trong conversion JSON. Với ONNX dynamic ResNet50 chỉ chọn MatMul/Gemm, phần Conv có thể vẫn FP32 nên không kỳ vọng giảm file 4×. NLP static TFLite được gọi mixed INT8, không full INT8.
- Giảm dung lượng không bảo đảm giảm latency: dynamic INT8 của ONNX text và dynamic range của TFLite CV chậm hơn FP32 cùng runtime ở lượt đo này. Nguyên nhân có thể liên quan kernel, overhead hoặc phần graph còn float; chưa có profiling để xác định tỷ lệ đóng góp.
- Benchmark chỉ một phiên trên máy dùng chung, không khóa power mode/nhiệt độ/tác vụ nền. Một số P95 cách xa median (đặc biệt ONNX FP32 CV); xem đủ 200 mẫu và độ lệch chuẩn. Speedup là mô tả phiên đo, chưa chứng minh mức cải thiện ổn định qua nhiều phiên.
- INT8 softmax có độ phân giải hữu hạn: không renormalize output để giấu sai khác. JSON ghi tổng xác suất và số tie top-1.
- Chỉ 100 ảnh Imagenette 160px resized và 100 câu SST-2 validation; kết quả không phải accuracy toàn bộ ImageNet/SST-2. Chênh +1 điểm phần trăm chỉ là thêm 1 mẫu đúng, chưa chứng minh quantization cải thiện chất lượng tổng quát. Một ảnh/một câu fixed length được lặp để benchmark, không đại diện mọi input.
- Chưa chạy các model quantized này trên Android, GPU, NNAPI; kết quả PC không đại diện điện thoại. Chưa đo RAM, năng lượng hoặc hiệu quả QAT.
- Calibration và lựa chọn op-types cố định trước evaluation. Không dùng evaluation để chọn scale hay tuning lại model.

## 7. Hướng thực hiện tiếp theo

Đo thêm nhiều mẫu/lượt, khảo sát độ dài text 32/128 token thành thử nghiệm riêng, kiểm tra calibration đa dạng hơn và triển khai các biến thể lên Android thật. Nếu accuracy giảm quá mức, cân nhắc calibration khác hoặc QAT; các bước này chưa thực hiện.

## 8. Code và bằng chứng bàn giao

- `README.md`: cách chạy và giải thích từng module; `hoc-quantization.html`: học liệu offline tương tác.
- `run_lab.py`, `qlab/`, `tests/`, `requirements.txt`, `requirements-lock.txt`, `sources.lock.json`.
- `results/summary.json`, `results/summary.csv`, `results/cv/outputs.npz`, `results/text/outputs.npz`: full raw outputs, nhãn và metric.
- `results/<task>/benchmark_*.json`: đủ mẫu latency; `conversion_*.json`: hash, strategy và audit; `data/<task>/manifest.json`: input/nhãn/source hashes.
- ZIP đóng theo allowlist, không chứa môi trường ảo, cache, credentials hay model/dataset lớn. Lệnh chạy lại tự tải từ revision/checksum đã pin.
- Unit tests và verifier kiểm tra phép quantization, label metrics, split leakage, hash output và thống kê latency. Verifier chạy được từ ZIP giải nén, không cần tải model.

Thời điểm tổng hợp (UTC+7): 2026-10-08T11:14:38.839716+07:00.