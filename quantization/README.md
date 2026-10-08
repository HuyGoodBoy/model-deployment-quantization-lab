# Thực nghiệm quantization ResNet50 và DistilBERT

Module đánh giá post-training quantization bằng ONNX Runtime và TensorFlow Lite trên hai mô hình pretrained. Trọng số gốc được giữ nguyên, không huấn luyện lại.

- [Báo cáo đầy đủ CPU/GPU ngày 08/10/2026](bao-cao-ngay-2026-10-08-cpu-gpu.md)
- [Báo cáo chi tiết CPU](bao-cao-quantization-2026-10-08.md)
- [Cấu hình và tái lập GPU](README-gpu.md)

## 1. Mô hình và dữ liệu

| Nội dung | ResNet50 | DistilBERT SST-2 |
|---|---|---|
| Bài toán | Phân loại ảnh ImageNet 1.000 lớp | Phân loại cảm xúc tiếng Anh, 2 nhãn |
| Dataset | Imagenette | SST-2 |
| Calibration / evaluation | 100 ảnh train / 100 ảnh validation | 100 câu train / 100 câu validation |
| Input | FP32 NHWC `[1,224,224,3]` | INT32 IDs/mask `[1,64]` |
| Tiền xử lý | RGB resize bilinear → BGR → trừ mean ResNet | WordPiece uncased, padding/truncation 64 |
| Output | 1.000 xác suất softmax | 2 logits |
| Metric chất lượng | Accuracy, top-5 accuracy | Accuracy, F1 positive |

Seed `20261008`; hai tập không giao theo hash ảnh/nội dung câu. Imagenette có 10 lớp nhưng accuracy được tính trên đủ 1.000 lớp đầu ra của ResNet50. Mỗi tập văn bản cân bằng 50 mẫu/nhãn; kết quả không đại diện văn bản tiếng Việt.

## 2. Các biến thể

| Runtime | Biến thể | Cấu hình |
|---|---|---|
| TensorFlow | FP32 | Reference so sánh output |
| ONNX Runtime | FP32 | Export từ model gốc |
| ONNX Runtime | Dynamic INT8 | QInt8 weights per-channel, chọn MatMul/Gemm |
| ONNX Runtime | Static INT8 QDQ | S8S8, MinMax; CV chọn Conv/MatMul/Gemm, text chọn MatMul/Gemm |
| TFLite | FP32 | Không quantize |
| TFLite | Dynamic range | Optimize.DEFAULT, không representative dataset |
| TFLite | FP16 weights | Giảm precision lưu trữ trọng số |
| TFLite | Static INT8 | CV full INT8 với I/O INT8; text mixed INT8/float, I/O token INT32 |

TFLite không bật SELECT_TF_OPS. Wrapper dùng scale/zero-point thực tế từ Interpreter để xử lý I/O. FP16 weights không bảo đảm CPU tính toán FP16; dynamic quantization có thể giữ nhiều operator FP32.

## 3. Môi trường và cài đặt

Môi trường CPU đã đo: Windows x64, Python 3.11.9, TensorFlow 2.15.1/Keras 2.15, ONNX Runtime 1.20.1, Transformers 4.38.2. Lock file ghi dependency thực tế trên Windows.

Các lệnh chạy trong thư mục `quantization/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run_lab.py
```

Lần đầu cần Internet. `sources.lock.json` cố định revision/checksum; downloader kiểm checksum trước khi dùng cache. Trọng số, dataset đầy đủ và canonical input lớn không được lưu trong Git. Linux/macOS dùng `.venv/bin/python` và `requirements.txt`; môi trường đó chưa được đo trong báo cáo này.

## 4. Pipeline thực nghiệm

```powershell
python run_lab.py --task all --stages download
python run_lab.py --task all --stages prepare
python run_lab.py --task all --stages convert
python run_lab.py --task all --stages evaluate
python run_lab.py --task all --stages benchmark --threads 1 --warmup 30 --runs 200
python run_lab.py --stages report
```

Thay `python` bằng interpreter đã cài dependency. Các stage phụ thuộc theo thứ tự prepare → convert → evaluate/benchmark → report. Khi đổi model, input hoặc runtime, cần tạo lại kết quả tương ứng.

Metric output gồm MAE, RMSE, max error, relative L2, cosine và SNR gộp theo năng lượng toàn output. Accuracy dùng nhãn thật; agreement so với TensorFlow gốc. Latency gồm inference API, tensor copy, overhead Python và q/deq; loại model load, đọc file và tiền xử lý. Benchmark lặp input đầu tiên, batch 1, CPU 1 thread, process riêng cho từng biến thể.

## 5. Mã nguồn

| File | Chức năng |
|---|---|
| `run_lab.py` | Điều phối stage/subprocess, ghi trạng thái lỗi |
| `qlab/common.py` | Paths, hash, cấu hình CPU và metric |
| `qlab/download.py` | Tải theo revision, kiểm checksum, resume |
| `qlab/prepare.py` | Chọn mẫu, tách hai tập, tiền xử lý |
| `qlab/runtime.py` | Wrapper TensorFlow/ONNX/TFLite và xử lý I/O |
| `qlab/convert.py` | Export, quantization, audit tensor/operator |
| `qlab/evaluate.py` | Lưu output, tính metric và accuracy/F1 |
| `qlab/benchmark.py` | Warm-up, đo và lưu đủ latency samples |
| `qlab/report.py` | Tổng hợp JSON, CSV, Markdown và HTML |
| `qlab/verify.py` | Tính lại metric/latency và kiểm checksum offline |
| `qlab/diagnose.py` | Phân tích scale trong TFLite FlatBuffer |
| `qlab/package.py` | Đóng gói ZIP CPU với inventory SHA256 |

## 6. Kết quả và kiểm chứng

```text
results/summary.json, summary.csv
results/cv/outputs.npz, evaluation_summary.json, benchmark_*.json, conversion_*.json
results/text/outputs.npz, evaluation_summary.json, benchmark_*.json, conversion_*.json
data/cv/manifest.json, data/text/manifest.json
sources.lock.json
```

NPZ có nhãn, reference `tensorflow_fp32` và output các biến thể; shape `[100,1000]` cho ảnh, `[100,2]` cho text. JSON benchmark chứa đủ 200 mẫu latency. Kết quả CPU đã được kiểm chứng offline; 10 unit test của module đã pass trong quá trình thực nghiệm. Verifier GPU chưa được chạy xác nhận trong lần bàn giao này.

Lệnh kiểm chứng từ dữ liệu CPU đã lưu:

```powershell
python -m qlab.verify
python -m unittest discover -s tests -v
```

Tạo lại báo cáo và ZIP CPU:

```powershell
python run_lab.py --stages report verify package
```

ZIP CPU nằm trong `dist/`, có `FILE_MANIFEST.json`; không lưu build/ZIP vào Git. Bộ ZIP CPU không bao gồm thực nghiệm GPU bổ sung.

## 7. Nhận xét

ResNet50 ONNX static INT8 giảm file khoảng 3,91× nhưng accuracy giảm 73% → 70%. DistilBERT TFLite FP16 weights giữ 91%, giảm file khoảng 2×. TFLite text mixed INT8 chỉ đạt 52%, chưa phù hợp triển khai. Scale lớn quanh attention mask là dấu hiệu cần phân tích thêm, chưa chứng minh nguyên nhân duy nhất.

Evaluation chỉ gồm 100 mẫu/model; thay đổi 1 điểm phần trăm tương ứng một mẫu đúng. Dung lượng file không phải RAM; giảm dung lượng không bảo đảm tăng tốc. Kết quả CPU không suy ra hiệu năng Android hoặc GPU.

## 8. Nguồn tham khảo

- [Keras ResNet50](https://keras.io/api/applications/resnet/resnet_models/)
- [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [Imagenette](https://github.com/fastai/imagenette)
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [TFLite quantization](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization)

Revision/checksum nằm trong `sources.lock.json` và `results/source_checksums.json`. Điều kiện sử dụng model/dataset theo từng nguồn.