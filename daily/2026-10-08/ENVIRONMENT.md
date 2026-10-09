# Môi trường thực nghiệm ngày 08/10/2026

| Phiên | Cấu hình và dependency |
|---|---|
| CPU quantization | Windows x64, Python 3.11.9; i7-12700H, RAM 16 GB; TensorFlow 2.15.1, ONNX Runtime 1.20.1, Transformers 4.38.2 |
| CPU/CUDA đối chứng | ONNX Runtime GPU 1.20.2; RTX 3050 Laptop 4 GB, driver 572.61; CUDA runtime 12.6.77, cuDNN 9.5.1.17 |
| Android FP32 | Pixel 8a thật, ARM64, Android 14/API 34; ONNX Runtime 1.20.0, TFLite 2.15.0, XNNPACK |

- [Lock CPU](../../environments/baseline/requirements-lock.txt)
- [Lock GPU](../../environments/gpu/requirements-lock.txt)
- [Inventory GPU thực tế](../../quantization/results/gpu/environment.json)
- [Cấu hình Android](../../android/README.md)

CPU/CUDA: batch 1, intra/inter-op 1, warm-up 30, runs 200, process riêng. Android: 1 thread, warm-up 30, runs 200; ONNX rồi TFLite trong cùng test process. Không dùng chéo baseline giữa phiên CPU cũ, phiên CPU/CUDA và Android để tính speedup.
