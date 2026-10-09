# Môi trường và nguồn đối chiếu ngày 09/10/2026

**Trạng thái:** thông tin khảo sát, chưa phải inventory của một phiên reproduce mới.

## 1. Baseline Huy ngày 08/10

- Windows x64, Python 3.11.9, Intel i7-12700H, RAM 16 GB.
- CPU: TensorFlow 2.15.1, ONNX Runtime 1.20.1, Transformers 4.38.2.
- GPU đối chứng: ONNX Runtime GPU 1.20.2, RTX 3050 Laptop 4 GB.
- Benchmark: 1 thread, batch 1, 30 warm-up, 200 lượt.

Nguồn: [môi trường ngày 08/10](../2026-10-08/ENVIRONMENT.md), [inventory GPU](../../quantization/results/gpu/environment.json).

## 2. Môi trường Đức theo repo

- Repo: [TruongDuke/Quantization, 4-quantization](https://github.com/TruongDuke/Quantization/tree/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization).
- Revision đã đọc: `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`.
- Báo cáo ghi Mac M1 Pro, CPU 4 thread; README hướng dẫn Python 3.11.
- Dependency khai báo: torch 2.13.0, torchvision 0.28.0, NumPy 2.4.6, Pillow 12.3.0, ONNX 1.23.2, onnxscript 0.7.2, ONNX Runtime 1.30.0, litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0.
- Benchmark trong code: batch 1, 10 warm-up, 100 lượt, median; CPUExecutionProvider và LiteRT Interpreter 4 thread.

Đây là phiên bản khai báo trong [requirements.txt](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/requirements.txt), chưa được cài hoặc kiểm khả dụng trên máy Huy. Delegate thực tế và inter-op thread chưa được ghi rõ trong code Đức.

## 3. Nội dung cần lưu khi chạy phiên mới

Mỗi phiên thực nghiệm phải lưu OS/CPU/GPU, Python, dependency thực cài, converter, checkpoint/revision/hash, input/preprocess, runtime/provider/delegate, thread, batch, warm-up/runs, phạm vi timer và timestamp theo Asia/Saigon. Cài cùng dependency trên Windows không biến máy Windows thành Mac M1 Pro; phải phân biệt đối chiếu software với đối chiếu phần cứng.
