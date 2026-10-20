# Chạy lại thực nghiệm ngày20/10

Chạy từ `D:\vin\710`, PowerShell. Model, audio, tensor và profiler trace bị ignore; clone repo chưa có đủ artifact để chạy verifier. M-LSD cần checkpoint/model/dataset của các ngày13–19, đúng hash trong manifest; không thay bằng model mới rồi dùng lại số cũ. Các script giữ kết quả đã ghi, không ghi đè JSON. Muốn chạy nghiên cứu mới, dùng thư mục ngày/experiment mới và registry riêng.

## Môi trường và nguồn

```powershell
docker build -t huy-day20-zipformer:onnx2tf269 daily/2026-10-20/experiments/zipformer
.venv-day09/Scripts/python.exe daily/2026-10-20/experiments/zipformer/src/prepare.py
.venv/Scripts/python.exe daily/2026-10-20/experiments/zipformer/src/prepare_eval.py
```

`prepare_eval.py` cần pyarrow, soundfile, NumPy và kết nối HF. Chuẩn bị/khóa source và73 audio trước tối ưu. Nhánh M-LSD dùng image CPU đã có `huy-peer-day09:cpu`; [package lock thực tế](environments/mlsd-requirements-lock.txt), [image ID và CPU](environments/runtime.json). Không dùng image Zipformer để chạy AEQ vì không cài torch/AEQ; không tự đổi phiên bản rồi gộp timing.

Hàm chạy Linux, không cần network sau chuẩn bị:

```powershell
function Run-Zip([string]$Script, [string[]]$ExtraArgs = @()) {
    docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-day20-zipformer:onnx2tf269 python "daily/2026-10-20/experiments/zipformer/src/$Script" @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Script" }
}
function Run-Mlsd([string]$Script, [string[]]$ExtraArgs = @()) {
    docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-peer-day09:cpu python "daily/2026-10-20/experiments/mlsd/src/$Script" @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Script" }
}
```

Nếu workspace ở chỗ khác, sửa source mount cho đúng. Không convert trực tiếp lên model nguồn: onnx2tf có thể simplify ghi đè input. Script conversion hôm nay tạo bản sao trước; không chạy CLI trực tiếp lên file ngày cũ.

## Zipformer

```powershell
Run-Zip baseline.py
Run-Zip convert_models.py @('--variant', 'raw')
Run-Zip convert_models.py @('--variant', 'static')
Run-Zip validate_artifacts.py
Run-Zip repair_decoder.py
Run-Zip evaluate_asr.py @('--variant', 'onnx')
Run-Zip evaluate_asr.py @('--variant', 'fixed')
Run-Zip optimize_decoder.py
Run-Zip evaluate_asr.py @('--variant', 'optimized')
Run-Zip benchmark.py
Run-Zip check_native_context.py
```

`benchmark.py` hiện sinh `matched_benchmark_v2.json`, prime metadata ngoài timing và kiểm counter nguyên exact. Không chạy nhánh `convert_models.py --variant optimized`: đó là đường generic simplification dự phòng chưa dùng; tối ưu thực đã chạy là `optimize_decoder.py`, có registry và parity riêng. `.tflite` xuất hiện sau raw conversion là kết quả âm có chủ đích; phải allocate/invoke mới biết có custom op chưa hỗ trợ.

Thời gian RTF chính dùng3 audio smoke,3 vòng; WER dùng73 câu một speaker. Giữ `[0,0]` context cho cảONNX/fixed/lookup. `check_native_context.py` chỉ là diagnostic giải thích vì sao native Sherpa dùng `[-1,0]` cho transcript khác, không thay pipeline đã khóa.

## M-LSD

```powershell
.venv-day09/Scripts/python.exe daily/2026-10-20/experiments/mlsd/src/profile_onnx.py
docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-day20-zipformer:onnx2tf269 python daily/2026-10-20/experiments/mlsd/src/convert_float.py
Run-Mlsd quantize_litert.py
Run-Mlsd evaluate_litert.py
Run-Mlsd diagnose_output_quantization.py
Run-Mlsd explore_float_output.py
Run-Mlsd thread_tuning.py
```

`convert_float.py` xác nhận nguồn SHA256 không đổi. Khi bản `float_conversion` đã có, kiểm byte identity với model đã đo. Khi chưa có, copy model mới sang vị trí float chuẩn rồi chạy parity/AEQ; trường `reference_existed_before_conversion=false` ghi rõ đây là bootstrap, chưa có đối chứng byte độc lập. Để tái lập trong checkout riêng, lưu trữ các JSON kết quả cũ trước rồi chạy pipeline tạo kết quả mới; không xóa/ghi đè bằng chứng ở workspace hiện tại. Checkpoint/dataset ngày cũ vẫn là điều kiện tiên quyết.

`evaluate_litert.py` chạy dev16, benchmark4 variants, khóa policy rồi mới chạy54 ảnh hồi quy. Tốn thời gian ghi tensor qua Docker Windows bind mount; không tính phần ghi file vào inference timing. `explore_float_output.py` chạy **development only**, không chạy lại54 ảnh hoặc chọn lại candidate. Hàm `--benchmark` cho nhánh thăm dò có sẵn nhưng hôm nay không dùng vì sAP chưa đạt gate. Thread tuning là matched benchmark khác, so1/2/4/8 luồng trên cùng FP32. Chạy các benchmark tuần tự khi không còn container thí nghiệm khác.

## Claude, kiểm chứng và commit

Người dùng cần tự đăng nhập Claude Code bằng `claude auth login` trong terminal tương tác. Không đọc/copy credentials. Lượt ngày20 đã thử thật nhưng không login, bằng chứng `claude_execution.json`; không ghi Claude đã sử dụng Read/Grep/Glob. Sau đăng nhập, tạo lượt execution mới để giữ bằng chứng lần lỗi, dùng [prompt](tools/claude_prompt.txt), lưu actual tools/model/response và patch được kiểm riêng. `run_claude_review.py` giữ evidence cũ nên không tự ghi đè.

```powershell
ruff check --no-cache daily/2026-10-20
ruff format --no-cache --check daily/2026-10-20
.venv-day09/Scripts/python.exe daily/2026-10-20/tools/verify.py
.venv-day09/Scripts/python.exe daily/2026-10-20/tools/test_scheduled_commit.py
Get-ScheduledTask -TaskName Codex-Commit-2026-10-20-1700
Get-ScheduledTaskInfo -TaskName Codex-Commit-2026-10-20-1700
```

Verifier là bằng chứng nhất quán, không phải gate triển khai cho mọi model. Chỉ chạy lần đầu trên output chưa có `VERIFICATION_final.json`; giữ log/JSON cũ nếu kiểm lại bằng lượt khác. `VERIFICATION.json` là lượt kiểm đầu, bản final bổ sung audit graph/trace,38 Conv INT8 và tensor8luồng exact theo review. Ruff/compile check không cần chạy lại inference khi chỉ sửa định dạng/prose. Test commit dùng repo tạm riêng trong `.cache`, không commit repo thật.

Task17:00 chỉ gọi `scheduled_commit.py`: kiểm marker review, HEAD/branch/index tree và không có unstaged tracked change, rồi `git commit -m 20/10`; không stage mới, không push. Kết quả thực nằm ở `.cache/scheduled-commit-2026-10-20/result.json`; chưa đến giờ không gọi là đã commit. Nếu index/working tree thay đổi sau review, task skip để tránh commit nhầm. Cần máy bật và user session sẵn; xem [schedule.json](tools/schedule.json).
