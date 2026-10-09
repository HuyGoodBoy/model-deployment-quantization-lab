# Reproduce và đối chiếu môi trường

**Ngày:** 09/10/2026. **Trạng thái:** đã chạy 6/6 biến thể ResNet18 từ source Đức trong Linux Docker/WSL2 và thêm 12 cấu hình model Huy trên môi trường đó.

Nguồn chỉ đọc: [4-quantization của Đức](https://github.com/TruongDuke/Quantization/tree/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization), revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`. Không sửa/push repo nguồn. Chạy bản sao ghim revision, ghi artifacts/results riêng; SHA256 nguồn trước/sau không đổi.

## Thiết kế đối chứng

1. Windows thiếu litert-converter; bổ sung Linux với đủ mười release dependencies Đức khai báo và pip check thành công. Chạy nguyên q1/q2/q3, giữ nguyên các hàm q4; wrapper tách variant và thu output/raw timing ngoài timer.
2. Reproduce ResNet18 dùng đúng 100/300 ảnh, preprocess/weights, 4 thread/10 warm-up/100 lượt. Đối chiếu riêng số Mac Đức báo cáo, không yêu cầu cùng phần cứng.
3. Model Huy giữ nguyên artifact/input, đo trên Linux với 4/30/200 để so matrix Windows cùng protocol; kiểm model/input hash. Khác OS/native dependencies/container vẫn là giới hạn quy nguyên nhân.

## Code và evidence

- `src/`: wrapper/adaptation của Huy, không chỉnh repo Đức.
- `results/<run-id>/`: môi trường, output, metric, latency samples, hash và log trạng thái.
- `artifacts/`: model tạo ra, giữ cục bộ.

[`src/runtime_compare.py`](src/runtime_compare.py) dùng cùng artifact/input Huy, ORT 1.20.1→1.30.0 và TF Lite 2.15.1→LiteRT 2.2.0; 4 thread, inter-op 1, warm-up 30/runs 200. [`results/run-01`](results/run-01/) có JSON latency/output metrics, environment, execution và verification. Raw NPY giữ cục bộ và tạo lại khi cần, không commit. [Cách chạy](../../RUNNING.md).

Không gán số Mac Đức báo cáo thành số Huy thực đo. Reproduce code không yêu cầu tái tạo phần cứng Mac; cần ghi rõ cấu hình khi so latency. [Báo cáo bổ sung](../../PEER-REPRODUCTION.md), [cách chạy Linux](PEER-RUNNING.md), [wrapper](src/reproduce_peer.py), [lock nguồn](peer-sources.lock.json).

- [Môi trường và dependency khai báo](../../ENVIRONMENT.md)
- [Task/tiêu chí](../../TASKS.md)
- [Báo cáo tiến độ](../../REPORT.md)
