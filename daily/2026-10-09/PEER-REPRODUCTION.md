# Reproduce code Đức và đối chiếu phần cứng — 09/10/2026

Đã chạy source ResNet18 của Đức trong Linux; 6/6 biến thể đo được.

**Không cần cùng phần cứng để reproduce.** Cần giữ pipeline, model/weights, preprocessing, selection dữ liệu, dependency chính và giao thức đo; ghi riêng cấu hình phần cứng/OS. So latency qua hai máy là quan sát toàn hệ thống, không tự quy cho riêng framework hoặc CPU.

## 1. Code và môi trường thực chạy

- Source Đức ghim revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`; SHA256 bảy file trong `peer-sources.lock.json`. Source giữ nguyên và được mount read-only.
- q1/q2/q3 chạy nguyên script trong process riêng. q4 dùng nguyên make_runner/evaluate/benchmark, tách từng variant để lỗi một model không chặn model khác; thu raw `times` sau timer qua `np.median`. Adaptation nằm trong JSON.
- ResNet18 pretrained ImageNet; 100 calibration train / 300 evaluation val, đúng sorted first 10/30 ảnh mỗi class theo code Đức. Dùng archive Imagenette đã có; cache weights chính thức.
- PyTorch reference accuracy: **64.6667%**, đối chiếu Đức báo **64.7%**. Không có artifact/hash raw của Đức để khẳng định tensor bit-identical giữa hai máy.
- Đức: Mac M1 Pro theo báo cáo. Huy: i7-12700H, Windows host, Linux x86_64 trong Docker/WSL2. Virtualization, OS, ISA, native kernel, backend build và dependencies gián tiếp cũng có thể ảnh hưởng tốc độ.
- Protocol của Đức: CPU 4 thread, batch 1, warm-up 10, đo 100 lượt median; API đồng bộ gồm copy/q-deq, không gồm load/preprocess.
- Main versions thực cài: `{"torch": "2.13.0+cpu", "torchvision": "0.28.0+cpu", "numpy": "2.4.6", "pillow": "12.3.0", "onnx": "1.23.2", "onnxscript": "0.7.2", "onnxruntime": "1.30.0", "litert-torch": "0.9.4", "ai-edge-litert": "2.2.0", "ai-edge-quantizer": "0.9.0"}`. Torch/torchvision là CPU build; chỉ ghim release versions khai báo, không có full lockfile môi trường Mac của Đức.
- Máy Linux nhìn thấy: `{"architecture": "x86_64", "visible_logical_cpus": 20, "cpu_model": "12th Gen Intel(R) Core(TM) i7-12700H", "cpu.max": "max 100000", "memory.max": "max"}`.
- Source unchanged: `True`. Evidence: [peer_reproduction.json](experiments/01-reproduce/results/run-04-peer-linux/peer_reproduction.json). Kết quả Mac là **số Đức báo cáo**, không phải Huy đo trên Mac.
- Collector đã sửa tên file raw FP32 trùng nhau. Output ONNX FP32 được phục hồi riêng và kiểm lại toàn bộ metric; các số quality/latency chính không đổi. [Kiểm chứng export output](experiments/01-reproduce/results/run-04-peer-linux/output_export_verification.json). [Verifier offline 18 cấu hình / 3.000 samples](experiments/01-reproduce/results/run-04-peer-linux/verification.json).

## 2. Cùng ResNet18: Đức báo cáo và Huy reproduce

| Model | Đức báo cáo Mac ms | Huy đo Linux ms | Accuracy Đức → Huy % | Agreement Đức → Huy % | SNR Đức → Huy dB |
| --- | --- | --- | --- | --- | --- |
| ONNX FP32 | 14.90 | 10.177 | 64.7 → 64.67 | 100.0 → 100.00 | 110.3 → 120.95 |
| ONNX dynamic | 9.50 | 40.724 | 65.7 → 65.67 | 93.3 → 93.33 | 20.5 → 20.51 |
| ONNX static | 3.70 | 10.999 | 68.0 → 67.33 | 96.3 → 96.33 | 24.6 → 24.59 |
| TFLite FP32 | 15.30 | 11.920 | 64.7 → 64.67 | 100.0 → 100.00 | 109.9 → 117.27 |
| TFLite dynamic | 4.70 | 5.122 | 67.3 → 67.33 | 95.3 → 95.33 | 27.7 → 27.72 |
| TFLite static | 4.30 | 4.940 | 61.0 → 61.00 | 85.0 → 85.00 | 20.9 → 20.87 |

SNR dùng công thức float32 gốc trong q4 để đối chiếu số Đức; JSON có thêm SNR float64, MAE/max error, hash và 100 raw latency samples. Không gọi tỉ số giữa hai máy là speedup của riêng framework. Chưa kiểm soát nhiệt/power và lặp nhiều session.

### Kiểm chứng clipping và profile riêng

- TFLite static output INT8 có scale 0.083820656, zero-point -47; max biểu diễn **14.584794**, trong khi reference max **32.523399**. Có 821 giá trị reference ngoài range output; không đồng nhất con số này với số lỗi phân loại.
- Có **62/300** mẫu hòa điểm top-1, trong đó argmax vẫn đúng **32** mẫu và sai **30** mẫu. Không gọi cả 62 mẫu hòa là 62 lỗi; ONNX static cũng có 10 mẫu hòa, nên không nói ONNX miễn nhiễm.
- Profile riêng 5 lượt sau 3 warm-up: ONNX dynamic, `ConvInteger` chiếm **93.3%** tổng thời lượng node-kernel events. `DynamicQuantizeLinear` chỉ 1.3%; ở phép thử này, chi phí kernel convolution integer là phần chính, không chỉ overhead tính range.
- ONNX static profile còn 11 `Conv` và 9 `QLinearConv` mỗi lượt; Quantize/Dequantize chiếm **20.1%** node-event time. Cùng tên static không bảo đảm cùng mức fusion/kernel coverage giữa backend.
- Profile là phép đo riêng có overhead, không thay số median trong bảng. Không có profile máy Đức nên chưa xác định nguyên nhân riêng của toàn bộ chênh lệch Mac/Linux. [Audit output và profile](experiments/01-reproduce/results/run-04-peer-linux/output_and_profile_audit.json).


## 3. Variant lỗi hoặc chưa hoàn tất

- Không có variant lỗi.

Trạng thái pipeline: `full_pipeline_measured`. Chỉ báo đủ sáu biến thể khi cả sáu có output/latency hợp lệ. Kết quả lỗi cũng là bằng chứng về tính portable của pipeline; không điền số latency giả.

## 4. Model Huy trên bộ môi trường đã reproduce

Đối chứng giữ nguyên artifact/input Huy, cùng protocol 4 thread / 30 warm-up / 200 lượt; so với Windows run-01. Không trộn protocol này với bảng ResNet18 10/100.

| Model Huy | Windows cũ ms | Windows mới ms | Linux peer env ms | Accuracy Linux | SNR Linux dB |
| --- | --- | --- | --- | --- | --- |
| cv/onnx_dynamic | 30.608 | 25.465 | 22.822 | 73% | 40.339 |
| cv/onnx_fp32 | 32.768 | 27.234 | 22.602 | 73% | 112.398 |
| cv/onnx_static | 29.831 | 32.544 | 23.621 | 70% | 18.894 |
| cv/tflite_dynamic | 163.850 | 15.669 | 11.845 | 73% | 26.576 |
| cv/tflite_fp32 | 57.345 | 93.937 | 25.332 | 73% | 114.099 |
| cv/tflite_static | 24.907 | 14.906 | 10.169 | 74% | 20.585 |
| text/onnx_dynamic | 11.658 | 11.753 | 9.324 | 91% | 25.011 |
| text/onnx_fp32 | 24.567 | 21.528 | 21.421 | 91% | 128.327 |
| text/onnx_static | 95.143 | 94.119 | 31.624 | 92% | 10.170 |
| text/tflite_dynamic | 1054.815 | 39.582 | 7.550 | 90% | 21.270 |
| text/tflite_fp32 | 163.676 | 36.111 | 20.993 | 91% | 124.530 |
| text/tflite_static | 114.788 | 86.863 | 24.954 | 48% | 0.030 |

Windows mới và Linux dùng cùng release ORT/LiteRT; Windows cũ còn khác runtime version. Hash model/input đã đối chiếu giữa hai môi trường. Những chênh lệch này không chứng minh tác động riêng của hardware vì môi trường native/OS/container cũng thay đổi.

## 5. Cách chạy

Xem [PEER-RUNNING.md](experiments/01-reproduce/PEER-RUNNING.md). Không sửa hoặc push repo Đức. q1 Windows ONNX-only trước đó nằm riêng ở `run-03-peer-windows`, có bỏ converter và không dùng thay cho kết quả Linux full-mode.
