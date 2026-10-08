# Chạy MobileNetV2 trên Firebase Test Lab

Project thực nghiệm: **gggg-a7df3**. Cấu hình đã sử dụng: **Pixel 8a
(model ID `akita`), Android 14/API 34, ARM64, physical device**. Đây là cấu hình
đọc được từ danh sách Test Lab ngày 08/10/2026; script sẽ kiểm tra lại trước khi chạy.

## 1. Phạm vi và phương pháp đo

Hai model giữ FP32, không quantize. ONNX dùng CPUExecutionProvider;
TFLite bật XNNPACK, tắt NNAPI và không thêm GPU delegate. Android dùng ONNX
Runtime 1.20.0 (bản 1.20.1 của Python không được phát hành cho Android trên
Maven Central) và TFLite 2.15.0. TensorFlow gốc
vẫn chạy trên PC; không cần cài TensorFlow gốc lên điện thoại.

App chứa model và input; APK instrumentation chứa test tự động. Vì vậy cần
upload **hai APK** và chọn **Instrumentation**, không chọn Robo test.
Không cần thêm Firebase SDK, `google-services.json` hoặc mật khẩu vào app.

Test dùng đúng tensor RGB NHWC `[1,224,224,3]` đã preprocess trên PC, kiểm tra
SHA-256 của input/model và lưu 1.000 xác suất FP32 của mỗi ảnh. Benchmark
chạy ảnh đầu tiên với batch=1, 1 thread intra-op, warm-up 30 lần và 200 lượt
đo; ONNX chạy trước TFLite. File kết quả có toàn bộ mẫu latency, phiên bản
runtime, Android/ABI/build fingerprint và nhiệt độ/thermal status.

Timer bao gồm gọi API Java, copy input/output và tạo tensor khi API yêu cầu;
loại tải/load model, đọc file, preprocess, ghi kết quả và tính metric. Hai
runtime chạy trong cùng một test process; giới hạn này khác benchmark PC
đã chạy từng process riêng. Nếu đo thêm, nên đảo thứ tự để kiểm tra ảnh hưởng.

## 2. Build cục bộ

Môi trường build dùng bộ công cụ portable trong `.cache/android-tools`: JDK 21,
Gradle 8.9, Android SDK/API 35 và build tools 35.0.0. Không sửa Java/SDK
của hệ thống. Chỉ lần đầu cần tải SDK và chấp nhận license của gói đã chọn.

Từ thư mục dự án:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\android\build-apks.ps1 -InstallSdk
```

Các lần sau bỏ `-InstallSdk`. Script tự đóng gói assets từ kết quả PC hiện có,
không chạy lại inference hay ghi đè số đo PC.

File tạo ra:

- `artifacts/android/deploy-lab.apk`: ứng dụng release, không debuggable,
  ký bằng khóa phát triển cục bộ; không dùng để publish Play Store.
- `artifacts/android/deploy-lab-test.apk`: instrumentation test tương ứng.

## 3. Xem cấu hình trước khi gửi test

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\android\run-testlab.ps1
```

Lệnh mặc định chỉ xem cấu hình, **chưa gửi cloud test**. Nó dùng project đã
chỉ định trong tham số, không đổi project mặc định của gcloud. Tài khoản
đang đăng nhập phải có quyền trên `gggg-a7df3`.

## 4. Chạy trên thiết bị thật

Kiểm tra quota trong Firebase trước khi chạy. Theo tài liệu được kiểm tra
ngày 08/10/2026, Spark có tối đa 5 lượt physical device/ngày; Blaze có 30 phút
physical device miễn phí/ngày, phần vượt tính phí theo thời gian. Không coi
đây là bảo đảm còn quota cho project: usage thực tế cần kiểm tra trong console.

Lệnh gửi instrumentation test với cấu hình và quota đã kiểm tra:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\android\run-testlab.ps1 -Submit
```

Chỉ một model/API, không tự retry nếu test lỗi, timeout 5 phút. Script tắt
video và performance metrics hệ thống để giảm phần chạy phụ. Kết quả inference
là metric do test tính, không phải biểu đồ startup/UI có sẵn của Firebase.

Nếu thao tác qua web: mở Test Lab trong project `gggg-a7df3`, tạo test
Instrumentation, upload hai APK, chọn **Pixel 8a/API 34/physical device**.
Trong tùy chọn nâng cao, đặt thư mục thu kết quả:
`/sdcard/Android/data/com.vin.deploylab/files/benchmark`.
CLI ở trên đã cấu hình việc thu thư mục này để tránh bỏ sót output.

Nếu upload trực tiếp bị ngắt mạng, có thể tải hai APK vào bucket mặc định
Test Lab trước, kiểm tra checksum, rồi truyền thư mục Cloud Storage vào script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\android\run-testlab.ps1 -ApkStoragePrefix 'gs://BUCKET/THU_MUC_APK' -Submit
```

Thư mục phải chứa đúng `deploy-lab.apk` và `deploy-lab-test.apk`. Không cần
tạo bucket mới hay bật billing để xử lý lỗi mạng; việc tải APK chưa phải
là tạo lượt test. Tham số này chỉ thay nguồn APK, giữ nguyên cấu hình benchmark.

## 5. Lấy và phân tích kết quả

Console Test Lab/đầu ra CLI có link matrix và đường dẫn Cloud Storage của
kết quả. Trong thư mục artifacts đã thu, tìm `android_run.json` và bốn file
`onnx_0.f32`, `onnx_1.f32`, `tflite_0.f32`, `tflite_1.f32`; giữ cùng thư mục.
Thư mục con thực tế có thể khác theo device/matrix; không đoán đường dẫn bucket.

Có thể tải từ giao diện Cloud Storage hoặc dùng đường dẫn thực tế CLI trả:

```powershell
gcloud.cmd storage cp --recursive 'gs://BUCKET/RESULTS_DIR' '.\results\android\raw'
```

Thay `BUCKET/RESULTS_DIR` bằng giá trị thực tế, không dùng nguyên placeholder.
Sau khi xác định thư mục chứa các file:

```powershell
.\.venv\Scripts\python.exe -m lab.android_results '.\results\android\raw\THU_MUC_CHUA_JSON'
```

Python kiểm tra checksum, model/input bundle, toàn bộ output hữu hạn và
softmax, số mẫu latency và thống kê; sau đó tính MAE, RMSE, max abs,
relative L2, cosine, SNR, top-1 agreement, top-5 overlap và allclose so với
TensorFlow gốc. Báo cáo lưu ở `results/android/<run hash>/report.md`, cùng
`summary.json`, giữ báo cáo PC riêng.

## 6. Cách đọc và xử lý lỗi

- Build APK thành công chưa có nghĩa model đã chạy thành công trên Android.
  Chỉ báo kết quả Android sau khi test và phân tích output thực tế hoàn tất.
- Không có ground truth; agreement 100% không phải accuracy 100%.
- Không so trực tiếp latency Java và Python như thể cùng phạm vi đo/phần cứng.
- Nhiệt độ, thứ tự runtime, phiên bản TFLite Android 2.15.0/PC 2.15.1 và tải
  nền của thiết bị đều cần ghi nhận. Hai ảnh chỉ xác minh pipeline.
- Test thất bại: xem log `DeployLab` và stack trace, không tự chạy lại nhiều
  lần để tránh tiêu quota. Nếu thiếu file, kiểm tra directories-to-pull và
  các hạn chế scoped storage của thiết bị trước khi chạy lại.

## 7. Kết quả đã chạy ngày 08/10/2026

Một lượt instrumentation trên **Pixel 8a thật, Android 14/API 34, ARM64** đã
Passed. Đã thu đủ output của hai ảnh và 200 mẫu latency mỗi runtime, kiểm
tra checksum và đối chiếu với TensorFlow gốc trên PC.

| Runtime | Median ms | P95 ms | Max abs error | SNR dB |
|---|---:|---:|---:|---:|
| ONNX Runtime | 35.284 | 40.406 | 1.25170e-6 | 119.179 |
| TFLite + XNNPACK | 22.092 | 24.640 | 6.55651e-7 | 124.650 |

Cả hai đạt allclose; top-1 agreement 2/2 ảnh. Không suy ra accuracy từ
agreement hoặc suy rộng thứ hạng tốc độ ra mọi điện thoại.

- [Báo cáo ngày 08/10 theo cấu trúc](../results/android/bao-cao-2026-10-08.md).
- [Báo cáo metric tự sinh](../results/android/ac619fe4f102/report.md).
- [Lượt chạy trên Firebase](https://console.firebase.google.com/project/gggg-a7df3/testlab/histories/bh.78eb0635a2d11da0/matrices/5956474525617947943).

Lệnh phân tích lại dữ liệu đã tải, không dùng thêm quota:

```powershell
.\.venv\Scripts\python.exe -m lab.android_results .\results\android\raw\matrix-5mhbmvu8ifjga\benchmark
```

## Tài liệu chính thức

- [Instrumentation tests](https://firebase.google.com/docs/test-lab/android/instrumentation-test)
- [Test Lab CLI](https://firebase.google.com/docs/test-lab/android/command-line)
- [Tham số thu kết quả](https://docs.cloud.google.com/sdk/gcloud/reference/firebase/test/android/run)
- [Quota và giá](https://firebase.google.com/docs/test-lab/usage-quotas-pricing)
- [Benchmark trên thiết bị thật](https://developer.android.com/topic/performance/benchmarking/benchmarking-in-ci)
