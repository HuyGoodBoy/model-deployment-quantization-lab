# Thực nghiệm triển khai và lượng tử hóa mô hình

**Người thực hiện:** Huy. **Phạm vi:** ONNX Runtime, TensorFlow Lite, CPU, GPU NVIDIA và Android.

## Báo cáo theo ngày

| Ngày | Task | Trạng thái | Báo cáo |
|---|---|---|---|
| [07/10/2026](daily/2026-10-07/README.md) | MobileNetV2 ONNX/TFLite FP32 trên PC | Có kết quả PC | [Báo cáo](daily/2026-10-07/REPORT.md) |
| [08/10/2026](daily/2026-10-08/README.md) | Quantization ảnh/text, GPU và Android FP32 | Có số đo; verifier GPU chưa xác nhận | [Báo cáo](daily/2026-10-08/REPORT.md) |
| [09/10/2026](daily/2026-10-09/README.md) | Runtime cũ/mới, DistilBERT INT8, M-LSD, detect box | Có thực nghiệm CPU/prototype; chưa reproduce đầy đủ Đức/Android | [Báo cáo](daily/2026-10-09/REPORT.md) |

Quy tắc tổ chức: [daily/README.md](daily/README.md). Mẫu ngày mới: [_template](_template/README.md).

## Tổng quan kết quả đến ngày 08/10/2026

Dự án đánh giá ảnh hưởng của chuyển đổi định dạng và post-training quantization đến sai khác output, độ chính xác, dung lượng và thời gian suy luận. Hai mô hình chính là **ResNet50 pretrained ImageNet** và **DistilBERT fine-tune SST-2 tiếng Anh**. Trọng số pretrained được giữ nguyên, không huấn luyện lại.

## 1. Tài liệu bàn giao

| Tài liệu | Nội dung |
|---|---|
| [Báo cáo ngày 09/10](daily/2026-10-09/REPORT.md) | Đối chứng runtime, strict/mixed INT8, mask ablation, M-LSD và Java geometry |
| [Tái lập ngày 09/10](daily/2026-10-09/RUNNING.md) | Hai venv riêng, pipeline tuần tự, kiểm chứng và ZIP |
| [Báo cáo quantization CPU/GPU](quantization/bao-cao-ngay-2026-10-08-cpu-gpu.md) | Phương pháp, kết quả, phân tích provider và giới hạn |
| [Báo cáo chi tiết CPU](quantization/bao-cao-quantization-2026-10-08.md) | Metric đầy đủ của ONNX Runtime và TFLite |
| [Hướng dẫn tái lập CPU](quantization/README.md) | Môi trường, pipeline và mã nguồn |
| [Hướng dẫn tái lập GPU](quantization/README-gpu.md) | ONNX Runtime CUDA và đối chứng CPU |
| [Báo cáo MobileNetV2 FP32 trên PC](results/report.md) | Thực nghiệm chuyển đổi định dạng ngày 07/10/2026 |
| [Báo cáo MobileNetV2 trên Android](results/android/bao-cao-2026-10-08.md) | Kết quả thiết bị thật trên Firebase Test Lab |

## 2. Thiết kế thực nghiệm

| Hạng mục | Phân loại ảnh | Phân loại văn bản |
|---|---|---|
| Mô hình | Keras ResNet50, ImageNet 1.000 lớp | DistilBERT SST-2, 2 nhãn |
| Dữ liệu | Imagenette | SST-2 tiếng Anh |
| Calibration | 100 ảnh train | 100 câu train |
| Evaluation | 100 ảnh validation | 100 câu validation |
| Input | FP32 NHWC `[1,224,224,3]` | INT32 IDs/mask `[1,64]` |
| Output | 1.000 xác suất softmax | 2 logits |
| Metric chất lượng | Accuracy, top-5 accuracy | Accuracy, F1 positive |

Calibration và evaluation tách riêng, seed `20261008`. Mọi runtime dùng cùng input sau tiền xử lý; TensorFlow FP32 là chuẩn so sánh output. Metric gồm MAE, RMSE, max absolute error, relative L2, cosine, SNR và top-1 agreement. Accuracy được tính với nhãn thật, độc lập với agreement.

CPU thử TensorFlow FP32; ONNX FP32, dynamic INT8, static INT8 QDQ; TFLite FP32, dynamic range, FP16 weights và static INT8. TFLite static ResNet50 dùng full INT8; DistilBERT dùng mixed INT8/float. GPU bổ sung ONNX FP16 và đối chứng CPU/CUDA cùng phiên bản runtime.

## 3. Môi trường và giao thức đo

- Windows x64, Python 3.11.9; Intel Core i7-12700H, RAM 16 GB.
- NVIDIA RTX 3050 Laptop GPU, VRAM 4 GB, driver 572.61.
- CPU ban đầu: TensorFlow 2.15.1, ONNX Runtime 1.20.1, Transformers 4.38.2.
- Đối chứng CPU/GPU: ONNX Runtime GPU 1.20.2, CUDA runtime 12.6.77, cuDNN 9.5.1.17.
- Batch 1, CPU 1 thread intra-op/inter-op, 30 warm-up và 200 lượt đo; process riêng chạy tuần tự.
- Latency gồm inference API, tensor copy và overhead Python; loại tải model, đọc file và tiền xử lý. GPU gồm truyền input/output CPU–GPU.

Provider thực thi được kiểm tra bằng profiling ngoài vùng đo. Graph có CPU fallback được ghi rõ; tên biến thể INT8 không đồng nghĩa mọi phép tính chạy INT8 trên GPU.

## 4. Kết quả chính

| Thực nghiệm | Median latency | Chất lượng | Nhận xét |
|---|---|---|---|
| ResNet50 ONNX FP32, CPU → CUDA | 83,232 → 5,903 ms | Accuracy 73%; SNR CUDA 111,920 dB | Nhanh hơn 14,10× trong cặp đối chứng |
| ResNet50 ONNX FP16 CUDA | 3,665 ms | Accuracy 73%; SNR 53,407 dB | File 48,74 MiB; nhanh hơn FP32 CUDA trong lượt đo |
| DistilBERT ONNX FP32, CPU → CUDA | 70,281 → 4,833 ms | Accuracy 91%; SNR CUDA 123,752 dB | Nhanh hơn 14,54× trong cặp đối chứng |
| DistilBERT ONNX dynamic INT8, CPU → CUDA/CPU | 25,325 → 26,914 ms | Accuracy 90% | 38 MatMulInteger chạy CPU; không tăng tốc trong lượt đo |
| DistilBERT ONNX FP16 CUDA | 11,168 ms | Accuracy 89% | Chậm hơn và giảm 2 điểm phần trăm so với FP32 CUDA |
| DistilBERT TFLite FP16 weights CPU | 344,053 ms | Accuracy 91% | File giảm khoảng 2× so với TFLite FP32 |
| DistilBERT TFLite mixed INT8 CPU | 268,074 ms | Accuracy 52%; FP32 91% | Chưa đạt chất lượng để triển khai |

ResNet50 ONNX static INT8 trên phiên CPU ban đầu giảm file 97,42 → 24,94 MiB và median 80,908 → 56,998 ms, nhưng accuracy giảm 73% → 70%. Hai phiên CPU khác runtime/thời điểm được trình bày riêng; không dùng chéo baseline để tính speedup GPU.

Thực nghiệm Android riêng dùng **MobileNetV2 FP32** trên **Pixel 8a thật, Android 14/API 34**: ONNX Runtime median 35,284 ms, TFLite XNNPACK 22,092 ms; SNR lần lượt 119,179 và 124,650 dB. Cả hai đạt top-1 agreement 2/2 ảnh. Đây là kiểm tra pipeline FP32; chưa đánh giá mô hình quantized trên Android.

## 5. Cấu trúc mã nguồn

```text
quantization/
  run_lab.py, run_gpu.py       Điều phối thực nghiệm CPU/GPU
  qlab/                       Download, prepare, convert, evaluate, benchmark, report
  requirements*.txt           Dependency và lock file CPU/GPU
  sources.lock.json           Revision và checksum nguồn model/dataset
  results/                    Output, metric, latency samples và profiling
  data/*/manifest.json        Danh sách mẫu calibration/evaluation
lab/                          Pipeline MobileNetV2 FP32 trên PC
android/                      Ứng dụng và instrumentation test Android
results/                      Kết quả MobileNetV2 trên PC/Android
```

Repository chứa mã nguồn, báo cáo và dữ liệu kiểm chứng. Môi trường ảo, cache, trọng số, dataset lớn, credentials, log và file build được loại khỏi Git. Dữ liệu lớn được tải lại theo revision/checksum đã ghi.

## 6. Tái lập thực nghiệm

Từ thư mục gốc repository, cài môi trường CPU:

```powershell
cd quantization
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run_lab.py
```

Sau khi có model, input và reference từ pipeline CPU, tạo môi trường GPU riêng:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-lock.txt
.\.venv-gpu\Scripts\python.exe run_gpu.py
.\.venv-gpu\Scripts\python.exe -m qlab.daily_report
```

Các lệnh tái lập sẽ tạo lại kết quả. Hướng dẫn từng stage và kiểm chứng offline nằm trong README của module. MobileNetV2 chạy bằng `run_lab.py` tại thư mục gốc; build và chạy Android theo [hướng dẫn Android](android/README.md).

## 7. Kết luận và giới hạn

Quantization giảm dung lượng nhưng mức tăng tốc phụ thuộc operator, kernel và thiết bị. ONNX FP32 CUDA cho kết quả tốt trên cả hai mô hình trong cấu hình đã đo. ResNet50 FP16 CUDA giữ accuracy; DistilBERT FP16 CUDA và TFLite mixed INT8 cần phân tích thêm về chất lượng.

Evaluation chỉ gồm 100 mẫu/model; benchmark lặp một input batch 1, chưa khóa nhiệt độ/power mode hoặc đo nhiều phiên để tính khoảng tin cậy. Kết quả không đại diện toàn ImageNet/SST-2, mọi thiết bị hay mọi độ dài văn bản. TFLite trong bài quantization chạy CPU; mô hình quantized chưa được đo trên Android. Verifier GPU đã có mã nguồn nhưng chưa được chạy xác nhận trong lần bàn giao này.
