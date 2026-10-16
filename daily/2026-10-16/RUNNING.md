# Tái lập kết quả cuối tuần 16/10

Các lệnh chạy từ root repo. Dùng `.venv` chuẩn bị text; `.venv-day09` chạy PyTorch/ONNX Runtime; Python có Matplotlib để vẽ hình. Dependency của hai môi trường nằm trong [environments](../../environments/). Không có training mới trong lượt này.

## Dữ liệu và artifact cần có

- Checkpoint và kết quả ngày [14/10](../2026-10-14/RUNNING.md), [15/10](../2026-10-15/RUNNING.md), model PTQ tách head ngày 13 và cache dataset ở revision đã khóa.
- Các model, checkpoint, input/output NumPy và profiler trace ở `experiments/finalize/artifacts/` được Git ignore. JSON nhỏ, mã tái lập và hình trong báo cáo là bằng chứng đi kèm.
- `prepare_final.py` chọn toàn bộ 54 ID YorkUrban ngoài 16 ảnh development và 32 ảnh holdout trước, kiểm thêm hash ảnh với train/calibration. Download dùng đúng revision trong metadata, không lấy revision mới.
- Text lấy 772 câu validation còn lại, loại câu trùng nội dung chuẩn hóa với 100 câu development, 100 câu calibration và 256 câu train. Tokenizer local, right padding/truncation ở 64 token. Manifest được ghi trước inference; không thay mẫu hoặc chọn model sau khi xem kết quả mới.

## Thứ tự chạy

Chạy tuần tự, đặc biệt không chạy inference/training khác trong lúc benchmark. Các bước selection/evaluation/benchmark từ chối ghi đè kết quả đã có. Tái lập toàn bộ trong một bản sao riêng có thư mục kết quả ngày 16 trống; không xóa hoặc thay JSON đã công bố để chọn lại cấu hình.

```powershell
.venv/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/prepare_final.py --task text
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/prepare_final.py --task mlsd
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/finalize_mlsd.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/export_text.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/select_text.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/evaluate_final.py --task mlsd
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/evaluate_final.py --task text
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/benchmark_final.py --task mlsd
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/benchmark_final.py --task text
python daily/2026-10-16/experiments/finalize/src/plot_final.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/verify_final.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/write_reports.py
```

## Protocol

M-LSD giữ checkpoint QAT bước 32 đã chọn ngày 15 và calibration foreground/channel. Một nhánh giữ 41 ConvInteger, một nhánh biểu diễn cùng range/weight bằng QDQ U8U8. Optimizer ORT thực tế chạy 22 QLinearConv và 19 Conv float ở nhánh QDQ. Không gọi hai nhánh là cùng precision. Sai lệch tích lũy được so ở 41 đầu vào convolution trên ảnh development đầu tiên; instrumentation phải giữ nguyên output cuối so với bản không instrumentation.

DistilBERT dùng checkpoint QAT bước 128. Có đối chứng teacher FP32, weight QAT tắt fake quant và fake quant. Export Linear bằng MatMulInteger UINT8×UINT8→INT32, weight scale theo output channel và static activation scale QAT, không thêm calibration. Profiler xác nhận 38 Linear integer; attention MatMul, embedding, Softmax, normalization và boundary còn float. Thử thêm cách làm tròn explicit trên development nhưng không cải thiện gate; bản static gốc được khóa trước đánh giá mới.

Gate M-LSD giữ relative L2 ≤2% và |chênh sAP10| ≤0,5 điểm. Text khai báo trước phép đo: relative L2 logits ≤2%, label agreement ≥99% và |chênh accuracy| ≤1 điểm trên 100 câu development. Việc chọn ứng viên tốt hơn khi không bản nào qua gate chỉ là chọn bản thực nghiệm; không biến trạng thái chưa đạt thành đạt.

Evaluator M-LSD được kiểm tra bằng 9 ca tính tay: đảo đầu mút, duplicate, false positive đứng trước true positive, ảnh không vuông, khoảng cách đúng 10/nhỏ hơn 10, tập rỗng và giải mã center/displacement. Vẫn dùng protocol subset ngày 13, chưa khẳng định tương đương evaluator paper. CI AP theo bootstrap ảnh 1.000 lần; CI chênh AP dùng paired bootstrap seed 20261015 kế thừa utility, mô tả sample count đã sửa thành 54. CI accuracy dùng Wilson; chênh accuracy dùng paired bootstrap 1.000 lần, seed 20261016.

Benchmark dùng ORT CPU, 4 intra/1 inter threads, sequential, ALL optimizations, tắt spinning. Chỉ một session/thread pool tồn tại mỗi lượt. Năm vòng xoay/đảo thứ tự model; mỗi model/vòng warm-up 16 rồi đo 16 input development, tổng 80 mẫu/model. Batch 1; M-LSD `1×4×512×512`, text 64 token. Không gồm load, preprocessing/postprocessing, decoder hoặc Android. Không so trực tiếp thời gian tuyệt đối với các lượt khác ngày/protocol.

Hình line dùng bốn ảnh ở chỉ số 0, 17, 35, 53 theo thứ tự ID, không chọn theo điểm đẹp/xấu. Overlay prediction hiển thị score ≥0,1; AP vẫn dùng decoder score gate 0,01. Mọi CI và kết luận chỉ áp dụng trên các tập đã đo, không chứng minh dữ liệu chưa từng xuất hiện trong pretrain.

`verify_final.py` tính lại metric, bootstrap, hash, kiểm split/selection và thống kê timing. Sai khác phép giảm tổng ở mức làm tròn float64 sau khi lưu tensor transpose được kiểm với rtol/atol 1e-10; ngưỡng chất lượng model không đổi. `status=passed` của verifier chỉ xác nhận bằng chứng nhất quán; trạng thái gate model được lưu riêng.

Nguồn operator/runtime: [ONNX MatMulInteger](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html), [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Số liệu, lựa chọn và giới hạn của repo là kết quả thực nghiệm riêng.

## Đối chiếu GPU máy hiện tại

Dùng `quantization/.venv-gpu` với ONNX Runtime GPU 1.20.2 và DLL CUDA 12/cuDNN 9 từ wheel NVIDIA trong venv. Không cần PyTorch GPU. Cấu hình phần cứng, DLL đã nạp và dependency được lưu trong `results/gpu/environment.json` và `results/gpu/verification.json`. CPU và CUDA dùng cùng wheel; không trộn timing với lượt CPU ORT 1.30 phía trên.

```powershell
$gpuScript = 'daily/2026-10-16/experiments/finalize/src/gpu_final.py'
quantization/.venv-gpu/Scripts/python.exe $gpuScript --prepare
foreach ($gpuTask in @('mlsd', 'text')) {
  $gpuVariants = if ($gpuTask -eq 'mlsd') { @('fp32', 'fp16', 'qat_41_integer', 'qat_qdq') } else { @('fp32', 'fp16', 'qat_static') }
  foreach ($gpuVariant in $gpuVariants) {
    foreach ($gpuDevice in @('cpu', 'cuda')) {
      quantization/.venv-gpu/Scripts/python.exe $gpuScript --task $gpuTask --variant $gpuVariant --device $gpuDevice
    }
  }
}
quantization/.venv-gpu/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/verify_gpu.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/verify_final.py
.venv-day09/Scripts/python.exe daily/2026-10-16/experiments/finalize/src/write_reports.py
```

Chạy trong bản sao có kết quả GPU trống. FP16 chuyển từ teacher FP32 bằng converter ORT, giữ IO float và default blocked ops, clip hằng số vào [1e-7; 1e4], không train lại. Các QAT model giữ nguyên hash đã chọn. Tập 54 ảnh/772 câu được dùng lại để đối chiếu runtime, không xem đây là holdout mới. Lượt CUDA FP32 M-LSD được probe trước khi chạy vòng CPU/CUDA còn lại; không đảo thứ tự model giữa các vòng.

Mỗi variant/device dùng process riêng, session 4 intra/1 inter threads, sequential ALL, spinning tắt. Warm-up 16 rồi đo 5×16 input ở chỉ số cách đều trên tập cuối, tổng 80 mẫu; profile trong session riêng sau timing. CUDA device 0, TF32 tắt, arena giới hạn 2 GiB, cuDNN heuristic và không bật max workspace. Timing gồm NumPy IO đồng bộ và truyền H2D/D2H, không gồm load, preprocess hoặc decode. Kết quả chỉ áp dụng cấu hình này, chưa phải latency đã tối ưu cho GPU.

`disable_fallback()` ngăn Python retry khi runtime lỗi; CPU partition vẫn được phép và phải báo cáo. Verifier đọc lại trace để đếm node và kiểu input theo provider, tính lại metric/parity/timing và kiểm hash. Đặc biệt ConvInteger/MatMulInteger ở CPU và Conv float ở CUDA không được tính là GPU INT8. Các gate QAT ban đầu giữ nguyên trạng thái chưa đạt.

Tài liệu: [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [chuyển FP16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).
