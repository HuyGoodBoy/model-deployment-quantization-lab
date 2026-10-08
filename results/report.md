# Báo cáo công việc: triển khai và đánh giá model pretrained

**Ngày báo cáo: 07/10/2026.**

## 1. Mục tiêu công việc

Tìm hiểu framework/runtime triển khai model trên các nền tảng; chọn một model pretrained, convert sang ONNX và TFLite ở FP32, chạy thử và đánh giá sai khác output cùng tốc độ inference.

**Phạm vi đã hoàn thành hôm nay:** tìm hiểu công cụ, convert hai định dạng và đo trên CPU máy tính. Chưa quantize, chưa chạy Android.

## 2. Nội dung đã tìm hiểu

| Công cụ | Nền tảng thường dùng | Vai trò |
|---|---|---|
| [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/) | PC/server/mobile, CPU và accelerator tùy provider | Chạy model ONNX |
| [LiteRT / TensorFlow Lite](https://developers.google.com/edge/litert) | Mobile và edge | Chạy model .tflite |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | GPU NVIDIA, Jetson | Tối ưu và chạy inference trên GPU NVIDIA |
| [OpenVINO](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) | CPU/GPU và thiết bị Intel được hỗ trợ | Tối ưu và chạy inference |
| [Core ML](https://developer.apple.com/documentation/coreml) | iOS/macOS | Chạy trên thiết bị Apple |

ONNX là định dạng biểu diễn model; ONNX Runtime là engine chạy model. Việc hỗ trợ operator và accelerator tùy phiên bản/thiết bị.

ONNX Runtime và LiteRT/TFLite được dùng trong thử nghiệm hôm nay. TensorRT, OpenVINO và Core ML mới được tìm hiểu về vai trò và nền tảng, chưa có benchmark thực nghiệm.

## 3. Cách thực hiện và điều kiện thử nghiệm

Quy trình: tải MobileNetV2 pretrained → chuẩn bị input dùng chung → export độc lập sang ONNX/TFLite → chạy từng runtime → lưu output → tính metric → đo latency → tổng hợp báo cáo.

- Model: Keras MobileNetV2 alpha=1, pretrained ImageNet; không train lại.
- Input: RGB NHWC [1,224,224,3], float32, resize bilinear, chuẩn hóa [-1,1].
- Output: xác suất softmax [1,1000]; so sánh toàn bộ vector.
- Export độc lập từ cùng model gốc sang ONNX và TFLite; FP32, chưa quantize.
- Phiên bản: TensorFlow 2.15.1, tf2onnx 1.16.1, ONNX 1.16.2, opset 13.
- ONNX Runtime 1.20.1: CPUExecutionProvider, ORT_ENABLE_ALL.
- TensorFlow: tf.function trên CPU. TFLite: Interpreter với delegate mặc định; xem log runtime.
- Máy: Windows-10-10.0.26200-SP0; CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; Python 3.11.9.
- Đánh giá 2 ảnh; dataset: `two_demo_images_not_representative`.
- Benchmark: CPU, batch=1, 1 thread intra-op; 30 warm-up, 200 lượt đo cho mỗi runtime trong process riêng.
- Benchmark dùng lặp lại ảnh `grace_hopper.jpg`; không suy rộng sang tất cả ảnh/tất cả thiết bị.
- Đo inference API gồm truyền input/output trong RAM và overhead Python; loại thời gian tải, load model, preprocess.

## 4. Kết quả đánh giá sai khác output

| Runtime | MAE | RMSE | Max abs | Relative L2 | Cosine | SNR dB | Top-1 agreement | Top-5 overlap | Allclose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| onnx | 1.18377e-09 | 1.69094e-08 | 6.55651e-07 | 6.17317e-07 | 0.99999999999983957 | 124.19 | 100.0% | 100.0% | True |
| tflite | 1.25701e-09 | 2.27284e-08 | 9.53674e-07 | 8.29755e-07 | 0.99999999999974787 | 121.621 | 100.0% | 100.0% | True |

So sánh 2 ảnh × 1.000 lớp trên cùng input FP32; TensorFlow gốc là tham chiếu. Các metric số học ở trên tính trên toàn bộ phần tử, không chỉ top-1.
MAE, RMSE, max abs và relative L2 càng nhỏ càng gần output gốc; cosine càng gần 1 và SNR càng cao càng tốt.
Cosine được giữ đủ chữ số để tránh làm tròn thành 1 rồi hiểu nhầm output giống hệt nhau.

### Sai khác output theo từng ảnh

| Ảnh | Runtime | MAE | RMSE | Max abs | Relative L2 | SNR dB | Allclose |
|---|---|---:|---:|---:|---:|---:|---|
| grace_hopper.jpg | onnx | 1.18606e-09 | 1.15586e-08 | 2.98023e-07 | 4.54145e-07 | 126.856 | True |
| sunflower.jpg | onnx | 1.18148e-09 | 2.09345e-08 | 6.55651e-07 | 7.1685e-07 | 122.891 | True |
| grace_hopper.jpg | tflite | 2.16542e-09 | 3.18853e-08 | 9.53674e-07 | 1.2528e-06 | 118.042 | True |
| sunflower.jpg | tflite | 3.48606e-10 | 4.0602e-09 | 1.19209e-07 | 1.39032e-07 | 137.138 | True |

### Dự đoán từng ảnh

| Ảnh | Runtime | Top-1 gốc | Top-1 sau convert |
|---|---|---|---|
| grace_hopper.jpg | onnx | military_uniform | military_uniform |
| sunflower.jpg | onnx | daisy | daisy |
| grace_hopper.jpg | tflite | military_uniform | military_uniform |
| sunflower.jpg | tflite | daisy | daisy |

Allclose dùng |candidate-reference| <= 1e-05 + 0.0001 * |reference| cho từng phần tử.
SNR = 10 log10(sum(reference²) / sum((candidate-reference)²)); tổng trên toàn bộ ảnh và lớp.
Top-1 agreement đo tỷ lệ nhãn dự đoán giống model gốc. Top-5 overlap đo tỷ lệ lớp chung trong hai tập top-5.

**Nhận xét:** trên hai ảnh mẫu, cả ONNX và TFLite có sai số tuyệt đối lớn nhất dưới 1e-6; cosine gần 1, SNR trên 120 dB ở mức tổng hợp, top-1 agreement và top-5 overlap đều đạt 100%. Cả hai đạt tolerance đã đặt. Output sau convert rất gần model gốc trong thử nghiệm này, nhưng không giống hệt từng phần tử.

## 5. Kết quả đánh giá tốc độ trên máy tính

| Runtime | File MiB | Mean ms | Median ms | P95 ms | Images/s | Speedup theo mean |
|---|---:|---:|---:|---:|---:|---:|
| tensorflow | 13.99 | 30.165 | 24.831 | 72.665 | 33.15 | 1.00x |
| onnx | 13.34 | 9.453 | 9.112 | 11.690 | 105.79 | 3.19x |
| tflite | 13.34 | 50.705 | 53.418 | 62.264 | 19.72 | 0.59x |

Images/s = 1000 / mean_ms với batch=1, chạy tuần tự. File size là dung lượng lưu trữ, không phải RAM sử dụng.

**Nhận xét:** ONNX Runtime nhanh hơn TensorFlow khoảng 3.19 lần theo mean. TFLite chậm hơn TensorFlow trên cấu hình CPU đã đo. Thứ hạng tốc độ này chỉ áp dụng cho lần thử nghiệm hiện tại; không suy ra tốc độ trên Android. P95 của TensorFlow cao hơn rõ rệt median, cho thấy thời gian các lượt chạy có biến động.

## 6. Kết luận và giới hạn

Đã hoàn thành pipeline với MobileNetV2 pretrained: export ONNX/TFLite FP32, kiểm tra sai khác trên toàn bộ vector output và đo tốc độ trên CPU máy tính. Hai model sau convert đạt tolerance trên tập mẫu đã dùng. ONNX Runtime có mean latency thấp nhất trong ba runtime ở lần đo này.

Chưa có ground-truth labels nên không báo cáo accuracy. Agreement cao chỉ cho biết model sau convert gần model gốc.

- Hai ảnh mẫu chỉ xác minh pipeline. Muốn kết luận về độ chính xác cần tập dữ liệu có nhãn lớn hơn, phù hợp bài toán.
- Chưa đo Android. Kết quả CPU máy tính không đại diện cho điện thoại; tốc độ giả lập cũng không đại diện máy thật.
- Cùng số thread không bảo đảm các runtime dùng cùng kernel. Các tối ưu graph/delegate được ghi nhận ở trên.
- Sai số FP32 có thể xuất hiện do thứ tự cộng/nhân, fuse layer và kernel khác nhau dù không quantize.
- Kết quả thời gian chịu ảnh hưởng tải máy, power mode và nhiệt độ; khi báo cáo nên giữ điều kiện ổn định.

## 7. Kế hoạch ngày mai — 08/10/2026

**Chạy thử và đo trên Android Emulator; phần này chưa thực hiện và chưa có số liệu.**

1. Chuẩn bị Android Studio, SDK và AVD; dùng system image x86_64 nếu máy tính có CPU Intel/AMD.
2. Chuẩn bị ứng dụng benchmark Android chạy cả ONNX Runtime và LiteRT/TFLite trên CPU, giữ model FP32.
3. Đưa cùng tensor input đã preprocess từ PC sang Android để tránh sai khác do resize/normalization.
4. Lưu đủ 1.000 xác suất mỗi ảnh, đưa về PC và so với output TensorFlow gốc bằng các metric như hôm nay.
5. Đo latency với batch=1, 1 thread, warm-up 30 lần và 200 lượt đo; ghi mean, median, P95 và phạm vi timer.
6. Bổ sung bảng kết quả riêng cho Android, ghi Android API, ABI, runtime, delegate và cấu hình emulator.

Kết quả giả lập sẽ được ghi là **Android Emulator**, không coi là hiệu năng của điện thoại thật. Nếu có thiết bị Android thật, có thể đo thêm và ghi thành môi trường riêng.

Tài liệu chuẩn bị: [tạo AVD](https://developer.android.com/studio/run/managing-avds), [ONNX Runtime Mobile](https://onnxruntime.ai/docs/get-started/with-mobile.html), [benchmark LiteRT](https://developers.google.com/edge/litert/models/measurement).

## 8. File bàn giao và dữ liệu kiểm chứng

- Model: `artifacts/mobilenetv2.keras`, `artifacts/mobilenetv2.onnx`, `artifacts/mobilenetv2.tflite`.
- Output gốc và sau convert: `results/outputs.npz`.
- Metric tổng hợp/từng ảnh: `results/evaluation.json`, `results/summary.csv`, `results/per_image.csv`.
- Dữ liệu latency: `results/benchmark_tensorflow.json`, `results/benchmark_onnx.json`, `results/benchmark_tflite.json`.
- Hướng dẫn học và thực hành: `hoc-deploy-model.html`, `README.md`.

Thời điểm tổng hợp báo cáo: 2026-10-07T12:52:14+07:00 (UTC+7). Ngày báo cáo ở đầu tài liệu xác định ngày công việc.
