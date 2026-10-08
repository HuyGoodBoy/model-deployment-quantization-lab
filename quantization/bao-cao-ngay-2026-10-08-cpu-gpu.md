# Báo cáo thực nghiệm quantization ResNet50 và DistilBERT — 08/10/2026

## 1. Công việc và kết quả chính

- Thực nghiệm **ResNet50 pretrained ImageNet** và **DistilBERT fine-tune SST-2 tiếng Anh**, không train lại.
- Đã chạy 8 biến thể/model trên CPU với ONNX Runtime và TFLite; bổ sung ONNX FP32, dynamic INT8, static QDQ INT8 và FP16 trên CPU/CUDA cùng phiên bản runtime.
- Đánh giá 100 ảnh + 100 câu có nhãn; calibration riêng 100 mẫu/model. Đo cả sai khác output, accuracy/F1, dung lượng và latency.
- **Text TFLite mixed INT8 chưa đạt chất lượng:** accuracy 52% so với 91% của model gốc; lưu nguyên kết quả và bằng chứng chẩn đoán attention scale.
- GPU được xác minh bằng profiling provider/operator. Có CUDA trong danh sách provider chưa đủ để khẳng định graph chạy GPU.

## 2. Model, dữ liệu và cách đánh giá

| Nội dung | CV | Text/NLP |
|---|---|---|
| Model | Keras ResNet50, 1.000 lớp ImageNet | DistilBERT SST-2, 2 nhãn cảm xúc tiếng Anh |
| Calibration | 100 ảnh train Imagenette, 10/lớp | 100 câu SST-2 train, 50/nhãn |
| Evaluation | 100 ảnh val Imagenette, 10/lớp | 100 câu SST-2 validation, 50/nhãn |
| Input | FP32 NHWC [1,224,224,3] | INT32 input_ids + attention_mask [1,64] |
| Preprocess | RGB resize bilinear, BGR trừ mean ResNet | WordPiece uncased gốc, pad/truncate 64 |
| Output so sánh | 1.000 softmax probabilities | 2 logits |
| Chất lượng | Accuracy/top-5 trên đủ 1.000 lớp | Accuracy/F1 positive, nhãn 0 negative/1 positive |

- Seed 20261008; kiểm tra hai tập không giao theo hash ảnh/nội dung câu. Text có 0/100 câu evaluation bị truncate.
- Tất cả runtime dùng cùng canonical input đã kiểm checksum. Token ID/mask giữ INT32, không đổi thành INT8.
- Model gốc để so output: TensorFlow FP32 của chính model đó. Accuracy cần nhãn thật; agreement chỉ đo nhãn dự đoán trùng model gốc.
- MAE = mean(|error|), RMSE = sqrt(mean(error²)), max = max(|error|); SNR = 10 log₁₀(Σreference²/Σerror²). Cosine gần 1 đo hướng vector, không bảo đảm độ lớn giống nhau.
- SNR tính gộp toàn output của 100 mẫu; không so trực tiếp SNR CV probabilities với NLP logits. JSON/CSV giữ thêm relative L2 và số liệu đầy đủ.

## 3. Môi trường và giao thức đo

- CPU: **12th Gen Intel(R) Core(TM) i7-12700H**, RAM máy 16 GB; Windows x64, Python 3.11.9.
- GPU thực tế: **NVIDIA GeForce RTX 3050 Laptop GPU, 4.096 MiB VRAM, compute capability 8.6**, NVIDIA driver **572.61**.
- Phiên CPU ban đầu: TensorFlow 2.15.1, ONNX Runtime 1.20.1, TFLite Interpreter đi kèm TensorFlow 2.15.1; Transformers 4.38.2.
- Phiên CPU/GPU bổ sung: **onnxruntime-gpu 1.20.2**; cả hai provider dùng cùng wheel/runtime và cùng model hash. GPU venv riêng, không sửa môi trường CPU.
- CUDA runtime 12.6.77, cuBLAS 12.6.4.1, cuDNN 9.5.1.17; DLL từ các wheel NVIDIA cài trong venv.
- `nvidia-smi` báo CUDA 12.8 là khả năng của driver, không chứng minh đã cài CUDA Toolkit 12.8. Thư viện runtime thực tế được ghi riêng ở trên.
- Batch 1; intra-op CPU 1, inter-op 1; **30 warm-up + 200 lượt/biến thể**, process riêng chạy nối tiếp. Cùng ảnh/câu đầu tiên được lặp để đo.
- GPU dùng **session.run đồng bộ với NumPy input/output trên CPU**: gồm H2D/D2H, API và overhead Python. Loại model load, đọc file, preprocess/tokenization; không phải kernel-only.
- CUDA EP đặt `use_tf32=0`, `device_id=0`, copy trong default stream; các option thực tế được lưu trong JSON. Profiling bằng session riêng sau timing, không làm tăng latency benchmark.
- Speedup GPU = median CPU phiên bổ sung / median CUDA của **cùng variant**. Không lấy CPU 1.20.1 phiên trước làm mẫu số cho GPU 1.20.2.
- TensorFlow cài sẵn báo `is_built_with_cuda=False`, GPU physical list rỗng. TFLite Interpreter 2.15 trong bài chưa cấu hình GPU delegate; các hàng TFLite dưới đây đều CPU. Không kết luận mọi LiteRT/Windows đều không hỗ trợ GPU.

## 4. Các kiểu quantization đã thử

- ONNX dynamic: QInt8 weights per-channel, chọn MatMul/Gemm. ResNet Conv và embedding có thể còn FP32, không gọi toàn bộ model là INT8.
- ONNX static: QDQ S8S8, MinMax calibration 100 mẫu; CV chọn Conv/MatMul/Gemm, NLP chọn MatMul/Gemm. Runtime có thể fusion hoặc chia graph theo provider.
- TFLite dynamic range: Optimize.DEFAULT không representative dataset. FP16 weights giảm storage, không bảo đảm CPU compute FP16.
- TFLite static CV: builtins INT8, input/output INT8, audit không còn tensor FP32. Text cho phép mixed INT8/float, giữ token IDs/mask INT32; không bật SELECT_TF_OPS.
- ONNX FP16 bổ sung: chuyển từ ONNX FP32, giữ kiểu IO, dùng default blocked operators; không phải integer PTQ. Converter clip constants về finite bounds 1e-7…1e4 nên sentinel attention mask có thể đổi; kiểm chất lượng bằng dữ liệu evaluation, không tune bằng evaluation.

## 5. Kết quả CPU ban đầu — ONNX Runtime và TFLite

Các số liệu dưới đây giữ nguyên phiên CPU đã thực hiện. Median/P95 tính từ đủ 200 mẫu; dung lượng file là MiB, không phải RAM.

### ResNet50 — phân loại ảnh

| Biến thể | MiB | Median ms | P95 ms | Accuracy | Top-5 | Max error | SNR dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 98.34 | 420.106 | 588.649 | 73.0% | 90.0% | 0 | ∞ |
| ONNX FP32 | 97.42 | 80.908 | 311.292 | 73.0% | 90.0% | 5.126e-06 | 112.398 |
| ONNX dynamic INT8 | 91.58 | 81.370 | 86.078 | 73.0% | 90.0% | 0.019277 | 40.339 |
| ONNX static INT8 QDQ | 24.94 | 56.998 | 58.810 | 70.0% | 93.0% | 0.32145 | 18.894 |
| TFLite FP32 | 97.42 | 408.996 | 573.731 | 73.0% | 90.0% | 5.7518e-06 | 113.826 |
| TFLite dynamic range | 24.76 | 1006.991 | 1409.609 | 73.0% | 90.0% | 0.11116 | 27.328 |
| TFLite FP16 weights | 48.74 | 424.462 | 528.457 | 73.0% | 90.0% | 0.0039162 | 57.829 |
| TFLite static INT8 | 25.07 | 326.472 | 396.072 | 74.0% | 90.0% | 0.33258 | 20.585 |

Bảng MAE/RMSE/cosine/agreement đầy đủ: [báo cáo CPU](bao-cao-quantization-2026-10-08.md), [JSON](results/summary.json).

### DistilBERT — sentiment tiếng Anh

| Biến thể | MiB | Median ms | P95 ms | Accuracy | F1 positive | Max error | SNR dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorFlow FP32 | 255.54 | 71.548 | 75.342 | 91.0% | 91.1% | 0 | ∞ |
| ONNX FP32 | 254.27 | 68.294 | 72.665 | 91.0% | 91.1% | 1.3024e-05 | 124.677 |
| ONNX dynamic INT8 | 131.42 | 108.638 | 173.225 | 90.0% | 90.0% | 1.1106 | 24.310 |
| ONNX static INT8 QDQ | 131.51 | 245.352 | 302.127 | 92.0% | 92.0% | 2.8138 | 10.168 |
| TFLite FP32 | 254.19 | 409.824 | 490.950 | 91.0% | 91.1% | 1.6212e-05 | 122.986 |
| TFLite dynamic range | 63.93 | 429.866 | 932.933 | 90.0% | 90.2% | 1.5926 | 21.119 |
| TFLite FP16 weights | 127.16 | 344.053 | 415.787 | 91.0% | 91.1% | 0.0089396 | 69.415 |
| TFLite calibrated mixed INT8 | 63.75 | 268.074 | 321.192 | 52.0% | 66.7% | 4.8464 | 0.036 |

Bảng MAE/RMSE/cosine/agreement đầy đủ: [báo cáo CPU](bao-cao-quantization-2026-10-08.md), [JSON](results/summary.json).

## 6. Kết quả chạy GPU và CPU đối chứng mới

Mỗi cặp dùng cùng graph/hash, ONNX Runtime 1.20.2 và cùng input. “CUDA + CPU” nghĩa profiling thấy graph có cả hai provider, kể cả khi chỉ một số op hỗ trợ/shape ở CPU. Không gọi INT8 CUDA là full INT8 GPU chỉ theo tên file.

### Kiểm tra dao động ResNet FP32: giữ cả hai lượt

CPU FP32 lượt đầu lệch lớn với phiên CPU cũ và bản dynamic, nên đo thêm đúng một cặp CPU/CUDA sau các phép đo khác. Bảng chính dùng cặp đo cuối, không chọn minimum của hai lượt. Không có bằng chứng để quy chênh lệch riêng cho phiên bản runtime hoặc GPU.

| Lượt | CPU median / P95 ms | CUDA median / P95 ms | Speedup cùng lượt |
|---|---:|---:|---:|
| Đầu tiên | 320.746 / 405.686 | 6.494 / 12.302 | 49.39× |
| Đối chiếu cuối | 83.232 / 85.689 | 5.903 / 6.009 | 14.10× |

Raw data lượt đầu giữ ở `results/gpu/cv/first_pass_fp32/`, cùng raw profile gốc. Hai phiên CPU cũ/mới còn khác môi trường/thời điểm; chưa có nhiều lượt kiểm soát để kết luận tốc độ ổn định.

### ResNet50 — phân loại ảnh — dung lượng và latency

| Biến thể | Thiết bị thực thi | MiB | Mean ms | Median ms | P95 ms | GPU/CPU speedup |
|---|---|---:|---:|---:|---:|---:|
| ONNX FP32 | CPU | 97.42 | 83.367 | 83.232 | 85.689 | — |
| ONNX FP32 | CUDA | 97.42 | 5.917 | 5.903 | 6.009 | 14.10× |
| ONNX dynamic INT8 | CPU | 91.58 | 130.684 | 84.489 | 326.859 | — |
| ONNX dynamic INT8 | CUDA + CPU | 91.58 | 6.114 | 6.103 | 6.231 | 13.84× |
| ONNX static INT8 QDQ | CPU | 24.94 | 59.341 | 59.124 | 61.966 | — |
| ONNX static INT8 QDQ | CUDA + CPU | 24.94 | 9.262 | 9.198 | 9.643 | 6.43× |
| ONNX FP16 (IO giữ nguyên) | CPU | 48.74 | 149.551 | 114.555 | 503.395 | — |
| ONNX FP16 (IO giữ nguyên) | CUDA | 48.74 | 3.775 | 3.665 | 4.376 | 31.26× |

### ResNet50 — phân loại ảnh — output CUDA so TensorFlow gốc

| Biến thể | MAE | RMSE | Max error | SNR dB | Cosine | Accuracy | Top-5 | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP32 | 2.7829e-09 | 6.28704e-08 | 5.45382e-06 | 111.920 | 1.0000000000 | 73.0% | 90.0% | 100.0% |
| ONNX dynamic INT8 | 1.12637e-05 | 0.000238479 | 0.019277 | 40.340 | 0.9999537590 | 73.0% | 90.0% | 100.0% |
| ONNX static INT8 QDQ | 0.000120249 | 0.00281865 | 0.408937 | 18.888 | 0.9935493697 | 73.0% | 91.0% | 93.0% |
| ONNX FP16 (IO giữ nguyên) | 2.40561e-06 | 5.29736e-05 | 0.0051913 | 53.407 | 0.9999977184 | 73.0% | 90.0% | 100.0% |

### ResNet50 — phân loại ảnh — kiểm chứng provider và sai khác cùng graph

| Biến thể CUDA | Node CUDA / CPU | Op tính toán trên CUDA | Op trên CPU | Max error so cùng variant CPU |
|---|---|---|---|---:|
| ONNX FP32 | 124 / 0 | Add, Conv, Gemm, GlobalAveragePool, MaxPool, Relu, Softmax, Squeeze, Transpose | — | 2.44379e-06 |
| ONNX dynamic INT8 | 129 / 2 | Add, BiasSoftmax, Cast, Conv, GlobalAveragePool, MaxPool, Mul, Relu, Squeeze, Transpose | DynamicQuantizeLinear×1, MatMulInteger×1 | 5.97462e-05 |
| ONNX static INT8 QDQ | 462 / 54 | Add, Conv, DequantizeLinear, Gemm, GlobalAveragePool, MaxPool, QuantizeLinear, Relu, Softmax, Squeeze, Transpose | DequantizeLinear×54 | 0.161769 |
| ONNX FP16 (IO giữ nguyên) | 126 / 0 | Add, Cast, Conv, Gemm, GlobalAveragePool, MaxPool, Relu, Softmax, Squeeze, Transpose | — | 0.00263092 |

Node count là số node duy nhất sau optimization, không nhân với số lượt profiling; không phải phần trăm FLOPs. Raw profile lưu type/shape của input kernel, các phép memcpy và CPU fallback. TensorRT/INT8 tensor core chưa được thử.

- Dtype input của op CUDA `onnx_static` đọc từ profile: `{'Conv': ['float'], 'GlobalAveragePool': ['float'], 'Gemm': ['float']}`. INT8 ở storage/QDQ không đồng nghĩa Conv/MatMul tính INT8; FP16 input cũng không tự chứng minh kiểu accumulation hoặc tensor core đã được sử dụng.
- Dtype input của op CUDA `onnx_fp16` đọc từ profile: `{'Conv': ['float16'], 'GlobalAveragePool': ['float16'], 'Gemm': ['float16']}`. INT8 ở storage/QDQ không đồng nghĩa Conv/MatMul tính INT8; FP16 input cũng không tự chứng minh kiểu accumulation hoặc tensor core đã được sử dụng.

### DistilBERT — sentiment tiếng Anh — dung lượng và latency

| Biến thể | Thiết bị thực thi | MiB | Mean ms | Median ms | P95 ms | GPU/CPU speedup |
|---|---|---:|---:|---:|---:|---:|
| ONNX FP32 | CPU | 254.27 | 70.378 | 70.281 | 72.606 | — |
| ONNX FP32 | CUDA | 254.27 | 4.880 | 4.833 | 5.247 | 14.54× |
| ONNX dynamic INT8 | CPU | 131.42 | 25.579 | 25.325 | 26.939 | — |
| ONNX dynamic INT8 | CUDA + CPU | 131.42 | 27.060 | 26.914 | 28.386 | 0.94× |
| ONNX static INT8 QDQ | CPU | 131.51 | 54.223 | 54.038 | 56.189 | — |
| ONNX static INT8 QDQ | CUDA + CPU | 131.51 | 9.238 | 8.950 | 10.564 | 6.04× |
| ONNX FP16 (IO giữ nguyên) | CPU | 127.27 | 97.974 | 97.176 | 102.229 | — |
| ONNX FP16 (IO giữ nguyên) | CUDA | 127.27 | 13.412 | 11.168 | 27.149 | 8.70× |

### DistilBERT — sentiment tiếng Anh — output CUDA so TensorFlow gốc

| Biến thể | MAE | RMSE | Max error | SNR dB | Cosine | Accuracy | F1 positive | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP32 | 1.53877e-06 | 2.37352e-06 | 1.61678e-05 | 123.752 | 1.0000000000 | 91.0% | 91.1% | 100.0% |
| ONNX dynamic INT8 | 0.10837 | 0.185936 | 0.907237 | 25.873 | 0.9987065364 | 90.0% | 90.0% | 99.0% |
| ONNX static INT8 QDQ | 1.0551 | 1.12859 | 2.84959 | 10.209 | 0.9860319852 | 91.0% | 90.9% | 96.0% |
| ONNX FP16 (IO giữ nguyên) | 0.216828 | 0.350858 | 1.64926 | 20.357 | 0.9953879558 | 89.0% | 89.1% | 98.0% |

### DistilBERT — sentiment tiếng Anh — kiểm chứng provider và sai khác cùng graph

| Biến thể CUDA | Node CUDA / CPU | Op tính toán trên CUDA | Op trên CPU | Max error so cùng variant CPU |
|---|---|---|---|---:|
| ONNX FP32 | 408 / 0 | Add, BiasSoftmax, Cast, Erf, Gather, Gemm, GlobalAveragePool, MatMul, Mul, Reciprocal, Relu, Reshape, Slice, Sqrt, Squeeze, Sub, Transpose | — | 8.52346e-06 |
| ONNX dynamic INT8 | 576 / 64 | Add, BiasSoftmax, Cast, Erf, Gather, GlobalAveragePool, MatMul, Mul, Reciprocal, Relu, Reshape, Slice, Sqrt, Squeeze, Sub, Transpose | DynamicQuantizeLinear×26, MatMulInteger×38 | 0.818146 |
| ONNX static INT8 QDQ | 820 / 2 | Add, BiasSoftmax, Cast, DequantizeLinear, Erf, Gather, Gemm, GlobalAveragePool, MatMul, Mul, QuantizeLinear, Reciprocal, Relu, Reshape, Slice, Sqrt, Squeeze, Sub, Transpose | DequantizeLinear×2 | 1.35979 |
| ONNX FP16 (IO giữ nguyên) | 409 / 0 | Add, BiasSoftmax, Cast, Erf, Gather, Gemm, GlobalAveragePool, MatMul, Mul, Reciprocal, Relu, Reshape, Slice, Sqrt, Squeeze, Sub, Transpose | — | 1.65366 |

Node count là số node duy nhất sau optimization, không nhân với số lượt profiling; không phải phần trăm FLOPs. Raw profile lưu type/shape của input kernel, các phép memcpy và CPU fallback. TensorRT/INT8 tensor core chưa được thử.

- Dtype input của op CUDA `onnx_static` đọc từ profile: `{'GlobalAveragePool': ['float'], 'MatMul': ['float'], 'Gemm': ['float']}`. INT8 ở storage/QDQ không đồng nghĩa Conv/MatMul tính INT8; FP16 input cũng không tự chứng minh kiểu accumulation hoặc tensor core đã được sử dụng.
- Dtype input của op CUDA `onnx_fp16` đọc từ profile: `{'GlobalAveragePool': ['float16'], 'MatMul': ['float16'], 'Gemm': ['float16']}`. INT8 ở storage/QDQ không đồng nghĩa Conv/MatMul tính INT8; FP16 input cũng không tự chứng minh kiểu accumulation hoặc tensor core đã được sử dụng.


## 7. Nhận xét và lựa chọn tiếp theo

- **ResNet50 — phân loại ảnh:** ONNX FP32 CUDA median 5.903 ms, speedup 14.10× so với CPU cùng graph; accuracy 73.0%. FP32 đạt allclose so TensorFlow gốc.
- ONNX FP16 CUDA cv: median 3.665 ms, 48.74 MiB, accuracy 73.0%, SNR 53.407 dB. Precision và chất lượng phải đọc cùng tốc độ.
- cv/ONNX dynamic INT8: profiling thấy các op tính toán {'MatMulInteger': 1} ở CPU. Đây là graph chạy kết hợp CPU/CUDA, không chứng minh kernel INT8 chính chạy trên GPU; transfer/fallback có thể ảnh hưởng latency.
- **DistilBERT — sentiment tiếng Anh:** ONNX FP32 CUDA median 4.833 ms, speedup 14.54× so với CPU cùng graph; accuracy 91.0%. FP32 đạt allclose so TensorFlow gốc.
- ONNX FP16 CUDA text: median 11.168 ms, 127.27 MiB, accuracy 89.0%, SNR 20.357 dB. Precision và chất lượng phải đọc cùng tốc độ.
- text/ONNX dynamic INT8: profiling thấy các op tính toán {'MatMulInteger': 38} ở CPU. Đây là graph chạy kết hợp CPU/CUDA, không chứng minh kernel INT8 chính chạy trên GPU; transfer/fallback có thể ảnh hưởng latency.
- **TFLite text mixed INT8 không chọn triển khai ở cấu hình này:** giảm 39 điểm phần trăm accuracy. FP16 weights giữ 91%, giảm file khoảng 2×; dynamic range giữ 90% nhưng không chắc nhanh hơn FP32.
- ONNX FP16 CUDA của text đạt 89% (so với 91% FP32), chậm hơn FP32 CUDA ở lượt này. CPU cùng file FP16 vẫn đạt 91%; TFLite FP16 CPU cũng đạt 91%. Storage FP16, CUDA compute với FP16 và CPU có cast sang FP32 là các điều kiện khác nhau, phải đọc dtype/profile và metric thực tế; chưa xác định nguyên nhân số học duy nhất bằng thí nghiệm loại trừ.
- QDQ static cùng file có accuracy khác giữa CPU/CUDA (CV 70%/73%, NLP 92%/91% ở lượt này). Profile CUDA thấy Conv/MatMul/Gemm nhận float; không gọi đây là full INT8 GPU hoặc khẳng định GPU cải thiện accuracy tổng quát.
- ONNX CPU text dynamic/static phiên mới nhanh hơn FP32 CPU, trong khi phiên CPU ban đầu chậm hơn. Hai phiên khác thời điểm và wheel/runtime; chưa cô lập nguyên nhân. Giữ cả hai bảng thay vì suy rộng một phiên ra mọi cấu hình.
- Kiểm tra TFLite text static thấy 21 tensor scale > 10⁶, lớn nhất 3.922e+27 quanh attention mask. Dải mask rất lớn có thể gây bước quantization quá thô; đây là suy luận từ metadata, chưa chứng minh nguyên nhân duy nhất bằng thí nghiệm loại trừ.
- ONNX static CV CPU ban đầu giảm file khoảng 3,91× và median khoảng 1,42×, nhưng accuracy 70% so với 73% gốc: cần cân đối mức mất chất lượng. TFLite static CV đạt 74% trên subset, không suy ra cải thiện toàn dataset.
- Giảm file không bảo đảm nhanh hơn. FP16 floating-point và INT8 integer có kernel/phạm vi hỗ trợ khác nhau; chọn theo graph, hardware và số đo thực tế.

## 8. Giới hạn và việc chưa làm

- Chỉ 100 mẫu/model; chênh 1 điểm phần trăm = 1 mẫu đúng. Không phải accuracy toàn ImageNet/SST-2, không đánh giá text tiếng Việt.
- Một phiên/laptop Windows WDDM dùng chung; chưa khóa nhiệt độ/power mode/tác vụ nền hoặc chạy nhiều phiên để có confidence interval. CPU 1 thread không đại diện CPU tối ưu toàn bộ core.
- Một input batch 1 lặp để timing, text cố định 64 token. Không đại diện throughput nhiều batch, mọi độ dài câu hay startup latency.
- Benchmark H2D/D2H, chưa tối ưu I/O Binding/CUDA Graphs, chưa TensorRT, QAT hoặc tuning calibration. Không chỉnh quantization dựa trên evaluation.
- Model quantized của bài này chưa chạy Android/GPU điện thoại; RTX 3050 không đại diện điện thoại. Chưa đo peak RAM/VRAM hoặc năng lượng; nvidia-smi chỉ kiểm cấu hình, không phải phép đo bộ nhớ đỉnh.

## 9. Code, bằng chứng và cách chạy lại

- [README GPU](README-gpu.md): môi trường riêng và các lệnh chạy.
- `run_gpu.py`, `qlab/gpu_runtime.py`, `gpu_experiment.py`, `gpu_verify.py`, `daily_report.py`: runner, kiểm CUDA/provider, phép đo, verifier và báo cáo.
- `requirements-gpu.txt` và `requirements-gpu-lock.txt`: package GPU, CUDA/cuDNN thực tế.
- `results/gpu/<task>/output_*.npy`: đủ 100 output/variant/provider; `cpu_*.json`, `cuda_*.json`: đủ latency samples, metric, hash, provider options.
- `results/gpu/<task>/profile_*.json`: raw profiling; `results/gpu/summary.json`, `summary.csv`: bảng tổng hợp.
- `results/<task>/outputs.npz`, manifest và conversion audit CPU được giữ nguyên làm chuẩn.
- Repo: [GitHub](https://github.com/HuyGoodBoy/model-deployment-quantization-lab). Giữ code, báo cáo và raw evidence cần thiết; bỏ venv/cache/credentials/models lớn, log và file build/ZIP.
- Trạng thái kiểm chứng: kết quả CPU đã được kiểm chứng offline và 10 unit test đã pass trong quá trình thực nghiệm. Verifier GPU đã có mã nguồn nhưng chưa được chạy xác nhận trong lần bàn giao này. Các lệnh dưới đây phục vụ tái lập.

```powershell
.\.venv-gpu\Scripts\python.exe run_gpu.py
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_verify
.\.venv-gpu\Scripts\python.exe -m qlab.daily_report
```

Tài liệu chính thức: [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [ORT FP16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html), [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [TensorFlow Windows](https://www.tensorflow.org/install/pip#windows-native), [LiteRT GPU](https://ai.google.dev/edge/litert/next/gpu).

Tạo báo cáo lúc (UTC+7): 2026-10-08T17:46:48.735483+07:00.
