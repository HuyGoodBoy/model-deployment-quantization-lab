# Thực nghiệm ONNX Runtime CUDA trên RTX 3050

Phần mở rộng bài quantization ResNet50 + DistilBERT. Giữ nguyên dữ liệu,
preprocessing, reference TensorFlow FP32 và kết quả CPU ban đầu. GPU dùng
ONNX Runtime CUDA, có thêm ONNX FP16. Các hàng TFLite/TensorFlow cũ vẫn CPU.

## Tài liệu kết quả

- `bao-cao-ngay-2026-10-08-cpu-gpu.md`: đầy đủ model/dataset, output,
  accuracy/F1, size, CPU/GPU latency, provider audit, kết luận và giới hạn.
- `results/gpu/summary.json`, `summary.csv`: số liệu tổng hợp.

## Môi trường GPU riêng

Windows x64, Python 3.11, NVIDIA driver tương thích CUDA 12.6. Phần GPU đã
thử trên RTX 3050 Laptop 4 GB. Không cài `onnxruntime` CPU và
`onnxruntime-gpu` vào cùng venv vì chúng cùng cung cấp module `onnxruntime`.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-lock.txt
.\.venv-gpu\Scripts\python.exe -m pip check
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --action inventory
```

`requirements-gpu.txt` là các dependency chính; lock file ghi toàn bộ
phiên bản đã cài. CUDA runtime/cuBLAS/cuDNN/cuFFT là wheel NVIDIA cài trong
venv, không phải đổi driver hoặc cài global CUDA Toolkit. ORT 1.20.2 chưa
có `preload_dlls()`, `gpu_runtime.py` thêm DLL directories và preload các
thư viện cần thiết trong process hiện tại.

`nvidia-smi` báo CUDA 12.8 là khả năng driver, không phải phiên bản Toolkit
đã cài. Metadata phiên bản các wheel được lưu ở `results/gpu/environment.json`.

## Chạy lại

Trên workspace đã có model/canonical inputs/reference của bài CPU:

```powershell
.\.venv-gpu\Scripts\python.exe run_gpu.py
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_verify
.\.venv-gpu\Scripts\python.exe -m qlab.daily_report
```

Từ ZIP mới giải nén, model/dataset lớn không nằm trong ZIP. Muốn đo lại,
chạy toàn bộ bài CPU theo `README.md` bằng **venv CPU riêng**, để tải
weights/dataset và tạo lại input/reference/kết quả có hash khớp nhau, rồi
chạy GPU như trên. Không giữ số đo cũ khi thay model/input/version.

Lệnh từng phần (cần dữ liệu và model của bài CPU):

```powershell
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --action convert --task cv
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --action convert --task text
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --task cv --variant onnx_fp32 --device cpu
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --task cv --variant onnx_fp32 --device cuda
.\.venv-gpu\Scripts\python.exe -m qlab.gpu_experiment --task text --variant onnx_fp16 --device cuda
```

`run_gpu.py` chạy CPU rồi CUDA cho mỗi variant, process riêng, nối tiếp;
batch 1, CPU intra/inter-op 1, warm-up 30, runs 200. Không chạy workload
khác trong lúc benchmark nếu muốn giảm biến động.

## Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `run_gpu.py` | Điều phối các process, giữ log và trạng thái lỗi |
| `qlab/gpu_runtime.py` | Load DLL, cấu hình CPU/CUDA EP, kiểm provider, đọc profiling |
| `qlab/gpu_experiment.py` | Tạo FP16, warm-up/đo, evaluate 100 mẫu, profile ngoài timer |
| `qlab/gpu_verify.py` | Tính lại metric, timing, checksum và provider từ raw evidence offline |
| `qlab/daily_report.py` | Tạo hai bản báo cáo và JSON/CSV từ kết quả đã đo |
| `qlab/daily_package.py` | ZIP allowlist và kiểm tra một bản giải nén mới |

Timer `session.run` trả NumPy output ở CPU, nên bao gồm H2D/D2H và đã đồng
bộ khi trả kết quả. Không đặt `disable_synchronize_execution_providers`,
không dùng I/O Binding để loại transfer khỏi số đo. Profiling bật trong
session riêng **sau** timing, không dùng latency có profiling làm benchmark.

`session.disable_fallback()` ngăn Python tự retry bằng provider khác khi
run lỗi. Nó **không** cấm graph partition CPU: CPU nodes được phép và ghi
trong profile. Nếu session thiếu CUDA EP hoặc profile chỉ thấy memcpy,
không chấp nhận đó là GPU compute. Với mixed graph, phải đọc operator và
input dtype của từng node, không suy ra mọi phép tính đều INT8 từ tên file.

FP16 converter giữ IO, dùng default blocked ops và finite clipping
1e-7…1e4. Attention mask sentinel có thể được clip. Đây là chuyển precision
floating-point, không phải INT8 quantization; cần kiểm output/accuracy.

## Kiểm chứng kết quả offline, không cần GPU

Chỉ cần Python 3.11, NumPy 1.26.4 và Pillow 10.4.0:

```powershell
python -m qlab.verify
python -m qlab.gpu_verify
python -m unittest discover -s tests -v
python -m qlab.daily_report
```

Verifier GPU không import ONNX Runtime hoặc TensorFlow lúc kiểm chứng,
không tải weights. NPY chứa `[100,1000]` hoặc `[100,2]`; JSON giữ đủ 200
latency samples, model/input/output/profile hashes, provider options và
metric. Original reference/labels trong `results/<task>/outputs.npz`.

Các unit test gồm 8 bài quantization/metric/nhãn và 2 bài phân tích provider;
10 test đã pass trong quá trình thực nghiệm. Verifier GPU đã có mã nguồn
nhưng chưa được chạy xác nhận trong lần bàn giao này. Các lệnh trên là
hướng dẫn tái lập, không phải bản ghi kết quả verifier.

## Nguồn chính thức

- [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime FP16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [TensorFlow trên Windows native](https://www.tensorflow.org/install/pip#windows-native)
- [LiteRT GPU theo platform](https://ai.google.dev/edge/litert/next/gpu)
