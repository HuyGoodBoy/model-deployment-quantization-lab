# Chạy lại thực nghiệm ngày 21/10

Chạy từ root `D:\vin\710`, PowerShell. Cần artifact/checkpoint/dataset các ngày13–20 đúng manifest và hai Docker image đã pin. Git không chứa model, audio, tensor hoặc raw profile lớn; clone trống chưa chạy được toàn bộ verifier.

Các script không ghi đè JSON/artefact đã có. Để tái lập, dùng workspace riêng với đúng nguồn ngày cũ và output ngày21 trống, giữ kết quả tham chiếu bên ngoài workspace chạy lại. Không xóa hoặc thay thế bằng chứng ở workspace hiện tại; không sửa file ngày20 để chạy thử ngày21.

## M-LSD

- Evidence: [Protocol](PROTOCOL.md) · [Runtime/packages](environments/) · [Kết quả](experiments/mlsd/results/).

Môi trường Windows `.venv-day09` đã có NumPy/ONNX/ORT/LiteRT/PyTorch; image `huy-peer-day09:cpu` có LiteRT2.2.0 và AEQ0.9.0. Image `huy-day20-zipformer:onnx2tf269` chỉ dùng để convert ONNX bằng native backend, không dùng để chạy AEQ.

```powershell
function Run-Mlsd([string]$Script, [string[]]$ExtraArgs = @()) {
    docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-peer-day09:cpu python "daily/2026-10-21/experiments/mlsd/src/$Script" @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Script" }
}
Run-Mlsd diagnose_activations.py
docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-day20-zipformer:onnx2tf269 python daily/2026-10-21/experiments/mlsd/src/split_heads.py
Run-Mlsd export_candidates.py
Run-Mlsd measure21.py @('development')
Run-Mlsd measure21.py @('benchmark')
Run-Mlsd benchmark8.py
Run-Mlsd measure21.py @('regression')
.venv-day09/Scripts/python.exe daily/2026-10-21/experiments/mlsd/src/analyze_timing.py
```

`split_heads.py` chỉ đọc ONNX FP32 ngày19, ghi graph mới rồi convert trên bản sao ngày21; kiểm SHA nguồn không đổi. `export_candidates.py` kiểm splitFP32 parity trước PTQ, xác nhận30INT8/11float cho A và42INT8/2output cho B. Dữ liệu calibration/development không đổi; adapter split chuyển NHWC sangNCHW và ghép sau khi dequantize.

`measure21.py benchmark` khóa chọn4threads trước regression. `benchmark8.py` chỉ so các biến thể đã đủ chất lượng development, không chọn theo54ảnh. Các output dùng `.npz` nén để giảm I/O Docker Windows bind mount; việc lưu tensor/tính metric ngoài vùng timing. Diagnostic preserve-all-tensors khác chế độ XNNPACK benchmark, không dùng timing diagnostic để kết luận hiệu năng.

## ZipFormer

- Evidence: [Source/patch](experiments/zipformer/src/) · [Kết quả](experiments/zipformer/results/) · [Nguồn LibriSpeech](https://www.openslr.org/12).

Dùng baseline/model/audio/manifest ngày20 đã kiểm hash; không convert lại hoặc sửa chúng. Image ASR có LiteRT2.1.2. Node profile dùng Windows ORT1.30.0, tách khỏi benchmark Linux; operator inventory và parity kiểm trên LiteRT thực.

```powershell
function Run-Asr([string]$Script, [string[]]$ExtraArgs = @()) {
    docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-day20-zipformer:onnx2tf269 python "daily/2026-10-21/experiments/zipformer/src/$Script" @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $Script" }
}
.venv-day09/Scripts/python.exe daily/2026-10-21/experiments/zipformer/src/profile_encoder.py
Run-Asr rewrite_fc.py
Run-Asr evaluate21.py @('baseline')
Run-Asr evaluate21.py @('fc')
Run-Asr evaluate21.py @('continuous')
Run-Asr benchmark21.py
.venv-day09/Scripts/python.exe daily/2026-10-21/experiments/zipformer/src/prepare_multispeaker.py
Run-Asr evaluate_multispeaker.py
```

`rewrite_fc.py` chỉ chấp nhận RHS FLOAT32 rank2 có buffer constant, không adjoint, weight không chia sẻ với consumer khác. Dùng FCv5 keepNumDims, transposeweight bit-exact, bias optional và giữ nguyên ADD. Nếu graph khác112candidate, dừng để kiểm thay vì áp dụng mù. Compact buffer giữ metadata. Parity15fixture phải qua trước đánh giá.

`prepare_multispeaker.py` là bước duy nhất cần network mới: tải khoảng346MB từ OpenSLR, resume HTTP Range nếu có partial, kiểm MD5 công bố rồi chọn seed cố định. Chỉ đọc member tar cần thiết bằng extractfile và ghi vào đường dẫn đã kiểm trong artifact, không extractall. Manifest100câu/20speaker khóa sau patch và trước inference; không chỉnh patch theo kết quả đó.

ASR RTF đo trên3smoke audio/3vòng, encoder timing160mẫu/5vòng; WER100câu là phép đánh giá chất lượng riêng. Session79,5s là lặp12lần audio, không phải tập WER mới. Không trộn với context[-1,0] của native Sherpa; toàn bộ pipeline ngày21 giữ[0,0] như control.

## Kiểm chứng và báo cáo

```powershell
ruff check --no-cache daily/2026-10-21
ruff format --no-cache --check daily/2026-10-21
.venv-day09/Scripts/python.exe daily/2026-10-21/tools/verify.py
```

Verifier yêu cầu đủ regression, matched benchmark, manifest và hai tập WER; chỉ tạo `VERIFICATION.json` khi tất cả pass. Tái tính AP/WER/raw timing/paired round bootstrap, kiểm native dtype, ma trận FC, hash và nguồn ngày20. Không cần rerun inference nếu chỉ sửa Markdown; đối chiếu bảng/link và hash dữ liệu cho lượt biên tập báo cáo.

REPORT.md là bản chi tiết; báo cáo chat ngắn và chia ZipFormer/M-LSD. Mỗi nhóm đặt Evidence trước nội dung. Công cụ thay Claude bằng Codex đã được người dùng yêu cầu ngày21; không ghi có Claude inference/tool execution. Không tự chạy lại lịch commit17:00 của ngày20 hoặc viết lại commit cũ.

## Bổ sung sau phản biện, không ghi đè lượt đầu

- Evidence: [Protocol bổ sung](PROTOCOL_REFINEMENT.md) · [Báo cáo cập nhật](REPORT.md) · [Verification bổ sung](REFINEMENT_VERIFICATION.json).

Các lệnh sau cũng chỉ chạy trong workspace tái lập với output mới còn trống. Giữ JSON/tensor lượt đầu để làm control. Run-Mlsd/Run-Asr là helper ở trên; dùng tham số -ExtraArgs để PowerShell truyền đủ danh sách đối số.

~~~powershell
docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-day20-zipformer:onnx2tf269 python daily/2026-10-21/experiments/mlsd/src/prune_head.py
Run-Mlsd refinement_mlsd.py -ExtraArgs @('control')
Run-Mlsd refinement_mlsd.py -ExtraArgs @('C1')
Run-Mlsd refinement_mlsd.py -ExtraArgs @('C2')
# Chỉ chạy C3 nếu C1 qua và C2 trượt như protocol; script cũng kiểm điều kiện này.
Run-Mlsd refinement_mlsd.py -ExtraArgs @('C3')
Run-Mlsd refinement_mlsd.py -ExtraArgs @('freeze')

foreach ($refinementThreads in @(4, 8)) {
    foreach ($refinementSession in @(1, 2)) {
        Run-Mlsd refinement_measure.py -ExtraArgs @('benchmark', '--threads', "$refinementThreads", '--session', "$refinementSession")
    }
}
Run-Mlsd refinement_measure.py -ExtraArgs @('lock')
Run-Mlsd refinement_measure.py -ExtraArgs @('regression')
Run-Mlsd refinement_io_profile.py
docker run --rm --network none --mount type=bind,source=D:\vin\710,target=/workspace huy-peer-day09:cpu python daily/2026-10-21/tools/probe_runtime.py

Run-Asr fuse_bias.py
Run-Asr refinement_asr.py -ExtraArgs @('quality')
Run-Asr refinement_asr.py -ExtraArgs @('benchmark')
Run-Asr coldstart_zipformer.py
.venv-day09/Scripts/python.exe daily/2026-10-21/tools/analyze_refinement.py
.venv-day09/Scripts/python.exe daily/2026-10-21/tools/verify_refinement.py
~~~

Không chạy hai benchmark cùng lúc; hoàn tất các container inference/calibration khác trước timing. C2/C3 trượt quality gate được ghi skip, không đưa vào benchmark/hồi quy đầy đủ. Selection được khóa trước hồi quy.

Benchmark cuối dùng C-contiguous NCHW output cho mọi model, gồm cả chi phí adapter/copy. Hai JSON exploratory refinement_benchmark_t*_s1.json thuộc contract cũ có C1 strided output, chỉ lưu provenance, không dùng lựa chọn cuối. Không cần tạo các exploratory đó trong lượt tái lập mới.

refinement_io_profile.py là diagnostic direct set/invoke/get; giải phóng SignatureRunner trước direct invoke để giữ safety guard, kiểm output exact trước timing. static_original_op_counts và delegate boundary không phải active execution plan hoặc số kernel Q/DQ CPU. Muốn native per-op attribution phải bổ sung profiler build tương thích; không coi flag profiling trên constructor là API đọc kết quả profiler.

coldstart_zipformer.py chạy 15 child process tách model, đọc /proc/self/status. RSS tính toàn process; không flush OS file cache hoặc đo Android. Các median init/first/peak tính từ sample độc lập, không cộng median các stage để thay median tổng.

Verifier bổ sung kiểm lại lượt đầu mà không ghi đè VERIFICATION.json; kết quả mới vào REFINEMENT_VERIFICATION.json. SNAPSHOT.initial.json giữ manifest lượt đầu; SNAPSHOT.json là bản bàn giao sau cập nhật báo cáo. Công cụ freeze_refinement.py kiểm bảng/link/hash trước khi cập nhật snapshot cuối.

~~~powershell
ruff check --no-cache daily/2026-10-21
ruff format --no-cache --check daily/2026-10-21
.venv-day09/Scripts/python.exe -m compileall -q daily/2026-10-21
.venv-day09/Scripts/python.exe daily/2026-10-21/tools/freeze_refinement.py
git diff --check
~~~

Theo yêu cầu hiện tại, chỉ lưu local, không commit/push và không tự đặt lịch hoặc sửa history ngày cũ.
