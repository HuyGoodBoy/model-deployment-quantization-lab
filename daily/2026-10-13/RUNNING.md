# Chạy lại task 13/10

Thí nghiệm này được chạy bổ sung ngày 14/10; timestamp thực nằm trong kết quả. Cần artifact M-LSD ngày 09 còn trên máy, gồm torch_state.npz và fp32/fp16/static.tflite. Clone mới chuẩn bị chúng theo [hướng dẫn ngày 09](../2026-10-09/RUNNING.md).

Chạy từ root repo, dùng các môi trường đã khóa bằng uv:

```powershell
python scripts/setup.py --profile baseline
python scripts/setup.py --profile runtime
.venv/Scripts/python.exe daily/2026-10-13/experiments/mlsd-ptq/src/data.py
.venv-day09/Scripts/python.exe daily/2026-10-13/experiments/mlsd-ptq/src/reference.py
.venv/Scripts/python.exe daily/2026-10-13/experiments/mlsd-ptq/src/ptq.py --runs 40
.venv/Scripts/python.exe daily/2026-10-13/experiments/mlsd-ptq/src/verify.py
```

Data: 32 ảnh Wireframe train để calibration, 16 ảnh YorkUrban test chỉ để đánh giá. Seed 20261013; revision, URL và hash ảnh/nhãn được khóa trong [data_manifest.json](experiments/mlsd-ptq/results/data_manifest.json). Dữ liệu tải từ mirror [Wireframe](https://huggingface.co/datasets/lh9171338/Wireframe) và [YorkUrban](https://huggingface.co/datasets/lh9171338/YorkUrban). M-LSD gốc cũng dùng các bộ này; nguồn checkpoint được giữ nguyên ở manifest ngày 09.

Reference là PyTorch với cùng checkpoint, chỉ lấy 5 kênh dùng để detect: center và 4 displacement. Mỗi cách đổi head phải qua gate FP32 trên đủ 16 ảnh trước khi quantize. Không train hoặc fit weight trên tập đánh giá trong PTQ.

sAP10 ở đây dùng tọa độ chuẩn hóa 128×128, khoảng cách bình phương tổng hai endpoint <10, không phân biệt thứ tự hai đầu, ghép một-một rồi tính AP. Decoder: NMS 3×3, top 200, score ≥0,01 và chiều dài trên map >20. Đây là protocol cho subset nhỏ, khác protocol báo cáo paper. Không lấy số này so trực tiếp với sAP paper.

Đo TFLite 2.15.1 CPU, 4 thread, warmup 10 và đo 40 lần. Bao gồm input quantize, inference, copy/dequantize output; không gồm resize ảnh và decode line. Không so latency khác phiên với số ngày 09.

FP16 selective chỉ đổi cách lưu một số weight/bias sang FP32; CPU vẫn tính FP32. INT8 split đã kiểm tra không còn tensor FLOAT32. Các bản mixed có float fallback và được ghi riêng. Quantize chọn lọc dùng API MLIR nội bộ của TF 2.15.1, nên phải giữ đúng phiên bản.

Model, tensor và checkpoint nằm trong artifacts/data/cache, được ignore. Git chỉ giữ source, manifest và kết quả JSON. verify.py tính lại metric từ output thô và kiểm tra loss/weight/checkpoint QAT nếu đã chạy task 14.
