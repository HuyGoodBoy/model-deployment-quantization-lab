# Chạy pilot QAT 14/10

Đây là mốc đầu của task tuần này: chạy được training với fake quant, chưa phải bản deploy INT8 hoàn chỉnh.

Cần dữ liệu/reference của task 13 đã chuẩn bị, checkpoint TF DistilBERT và dữ liệu SST-2 ngày 08 còn trên máy. Dùng baseline và runtime đã khóa; setup dùng uv qua scripts/setup.py.

```powershell
.venv/Scripts/python.exe daily/2026-10-14/experiments/qat/src/prepare.py
.venv-day09/Scripts/python.exe daily/2026-10-14/experiments/qat/src/train.py --task mlsd --steps 8
.venv-day09/Scripts/python.exe daily/2026-10-14/experiments/qat/src/train.py --task text --steps 8
.venv/Scripts/python.exe daily/2026-10-13/experiments/mlsd-ptq/src/verify.py
```

Training dùng PyTorch 2.13 CPU, 4 thread; AdamW lr=1e-5, weight decay=0,01, clip gradient=1. Weight fake quant INT8 đối xứng per-channel; activation INT8 affine per-tensor. Observer MinMax được warm-up trên dữ liệu train/calibration rồi freeze; fake quant vẫn bật trong train và eval.

- M-LSD: 32 ảnh Wireframe train, batch 1. BN được fold trước QAT và kiểm tra lại output float. Loss gồm BCE mềm cho center và Smooth L1 cho displacement tại vị trí teacher có confidence >0,1. Teacher FP32 tạo pseudo-label; chưa dùng loss supervised từ nhãn line. Center/displacement có fake quant đầu ra riêng. Chạy 8 bước nghĩa là dùng 8 ảnh cho gradient, không phải đã train hết bộ dữ liệu.
- DistilBERT: chuẩn bị 256 câu SST-2 train cân bằng hai nhãn, không trùng 100 câu observer hoặc 100 câu validation cũ. Batch 2; loss 0,5 CE + 0,5 distillation KL, temperature 2. Freeze embedding; 38 Linear được fake quant. Attention MatMul/Softmax, LayerNorm và mask còn float. Chạy 8 bước dùng 16 câu cho gradient.

Model text được dựng lại bằng PyTorch từ đúng 104 tensor của checkpoint TF, có gate output trên 100 câu trước khi QAT. Pilot giữ mask −1e4 trong float; gate kiểm tra tương đương với reference cũ, không mặc định đúng trên mọi input.

Checkpoint gồm model, optimizer, step và RNG; có thể chạy thêm lượt ngắn bằng --resume --steps 8. Không dùng --resume để ghi đè một thí nghiệm khác. Lần chạy không có --resume sẽ khởi động lại từ checkpoint FP32.

Progress JSON ghi loss từng bước, số parameter, weight hash trước/sau, checkpoint hash và metric fake-quant trước/sau. Không benchmark tốc độ fake quant rồi gọi đó là tốc độ INT8. Phần còn lại của tuần: train lâu hơn với dữ liệu/validation rộng hơn, xuất INT8 và audit graph, sau đó mới đo chất lượng/tốc độ bản deploy.

Nguồn phương pháp: [PyTorch FakeQuantize](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html), [QAT prepare và convert](https://docs.pytorch.org/blog/quantization-aware-training/). Fake quant mô phỏng lượng tử trong float; bước convert/export và kiểm chứng runtime vẫn cần làm riêng.
