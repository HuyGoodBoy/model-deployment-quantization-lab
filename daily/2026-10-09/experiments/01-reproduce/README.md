# Reproduce và đối chiếu môi trường

**Ngày:** 09/10/2026. **Trạng thái:** đã tạo môi trường runtime đối chiếu Windows; chưa reproduce đầy đủ ResNet18/converter/Mac của Đức.

Nguồn chỉ đọc: [4-quantization của Đức](https://github.com/TruongDuke/Quantization/tree/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization), revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`. Không sửa/push repo nguồn. Yêu cầu hiện tại chỉ cho phép đọc; chưa thực thi code nguồn.

## Thiết kế đối chứng

1. Thử cài dependency Đức; thiếu Windows distribution litert-converter 0.4.*, nên phần converter/ResNet18 đầy đủ còn thiếu.
2. Chạy model Huy bằng các runtime/công cụ đối chiếu tương ứng; ghi rõ phần chuyển đổi nào cần thay vì giả định graph/model có thể dùng trực tiếp.
3. Tách tác động model/preprocessing/thread/software khỏi phần cứng. Ghi riêng mỗi cấu hình, dùng cùng input và phạm vi timer trong từng cặp.

## Code và evidence

- `src/`: wrapper/adaptation của Huy, không chỉnh repo Đức.
- `results/<run-id>/`: môi trường, output, metric, latency samples, hash và log trạng thái.
- `artifacts/`: model tạo ra, giữ cục bộ.

[`src/runtime_compare.py`](src/runtime_compare.py) dùng cùng artifact/input Huy, ORT 1.20.1→1.30.0 và TF Lite 2.15.1→LiteRT 2.2.0; 4 thread, inter-op 1, warm-up 30/runs 200. [`results/run-01`](results/run-01/) có JSON latency/output metrics, environment, execution và verification. Raw NPY ở máy/ZIP, không commit. [Cách chạy](../../RUNNING.md).

Không gán số đo trong báo cáo Đức thành kết quả reproduce. Cài cùng dependency trên Windows chỉ đối chiếu software, không tái tạo Mac M1 Pro.

- [Môi trường và dependency khai báo](../../ENVIRONMENT.md)
- [Task/tiêu chí](../../TASKS.md)
- [Báo cáo tiến độ](../../REPORT.md)
