# Tái lập thực nghiệm ngày 22/10/2026

## Môi trường và dữ liệu

- Chạy từ root repository, PowerShell với Docker Linux. Hai image giữ nguyên so ngày 21; package lock: [M-LSD](environments/mlsd-requirements-lock.txt), [ZipFormer](environments/zipformer-requirements-lock.txt).
- M-LSD: `huy-peer-day09:cpu`, image ID `sha256:0da2a299ffa5f1a4f469c3af94c18e90b68d619dbabd7e8e0fbd340380f0ec60`, LiteRT 2.2.0.
- ZipFormer: `huy-day20-zipformer:onnx2tf269`, image ID `sha256:bdda599868579fdee3b68c769b046fb7da2a57bdf5ab8c71d86aeb979ee83616`, LiteRT 2.1.2.
- Khôi phục artifact theo [ngày 20](../2026-10-20/RUNNING.md) và [ngày 21](../2026-10-21/RUNNING.md): C1/FP32 M-LSD, encoder FC ZipFormer, decoder/joiner, input/state fixture, 32 ảnh Wireframe calibration, 16 ảnh YorkUrban development, manifest 54 ảnh và 173 câu speech. Script kiểm hash trước khi đọc artifact được tham chiếu.
- Model, audio và tensor lớn không nằm trong Git. JSON chứa đường dẫn, hash và raw timing nhỏ; cần tạo lại artifact để chạy kiểm tensor/AP.

Các lệnh dưới đây dùng cho một checkout tái lập có đủ artifact ngày trước và chưa có kết quả ngày 22. Script từ chối ghi đè kết quả. Để giữ bản đo gốc, dùng một bản sao workspace riêng. Trong **bản sao**, các output ngày 22 sau cần chưa tồn tại trước khi chạy:

- `experiments/mlsd/results/`, `experiments/mlsd/artifacts/`.
- `experiments/zipformer/results/`, `experiments/zipformer/artifacts/`.
- `environments/mlsd.json`, `environments/zipformer.json` và hai file `*-requirements-lock.txt` của ngày 22.
- `VERIFICATION.json`.

Giữ nguyên source và toàn bộ artifact/kết quả ngày trước. Các output này được tạo lại bằng lệnh phía dưới. `SNAPSHOT.json`, artifact review độc lập và bảng báo cáo mô tả lần đo đã công bố; không dùng chúng để xác nhận timing của lần chạy mới.

```powershell
$mount22 = "type=bind,source=$((Get-Location).Path),target=/workspace"
$mlsdImage22 = 'huy-peer-day09:cpu'
$zipImage22 = 'huy-day20-zipformer:onnx2tf269'
$mlsdSrc22 = 'daily/2026-10-22/experiments/mlsd/src'
$zipSrc22 = 'daily/2026-10-22/experiments/zipformer/src'

function Run22 {
    param(
        [string]$image22,
        [string]$script22,
        [Parameter(ValueFromRemainingArguments=$true)][string[]]$arguments22
    )
    docker run --rm --network none --mount $mount22 $image22 python $script22 @arguments22
    if ($LASTEXITCODE -ne 0) { throw "Failed: $script22 $arguments22" }
}
```

## M-LSD

```powershell
Run22 $mlsdImage22 "$mlsdSrc22/io_models.py" @('C1_roundtrip')
Run22 $mlsdImage22 "$mlsdSrc22/quality22.py" @('control')
Run22 $mlsdImage22 "$mlsdSrc22/io_models.py" @('M1')
Run22 $mlsdImage22 "$mlsdSrc22/quant_probe.py" @()
Run22 $mlsdImage22 "$mlsdSrc22/quality22.py" @('M1')
# Chỉ tiếp tục M2 nếu quality_M1.json có quality_passed=true.
Run22 $mlsdImage22 "$mlsdSrc22/io_models.py" @('M2')
Run22 $mlsdImage22 "$mlsdSrc22/quality22.py" @('M2')
Run22 $mlsdImage22 "$mlsdSrc22/quality22.py" @('freeze')

foreach ($session22 in @(1,2)) {
    foreach ($threads22 in @(4,8)) {
        Run22 $mlsdImage22 "$mlsdSrc22/measure22.py" @('benchmark','--threads',"$threads22",'--session',"$session22")
    }
}
Run22 $mlsdImage22 "$mlsdSrc22/quality22.py" @('configuration')
Run22 $mlsdImage22 "$mlsdSrc22/measure22.py" @('lock')
Run22 $mlsdImage22 "$mlsdSrc22/measure22.py" @('regression')
Run22 $mlsdImage22 "$mlsdSrc22/io_profile22.py" @()
```

Đường dựng M1 cuối dùng C1 đóng gói lại đã kiểm output. Prototype được đo ban đầu dựng từ C1 gốc và materialize trong cùng bước; kiểm cấu trúc độc lập xác nhận base chuẩn hóa khớp hoàn toàn đối chứng đóng gói lại. Bản thiếu external buffer ban đầu chỉ là lỗi serialization, chưa chạy inference và không thuộc tập benchmark.

## ZipFormer

```powershell
Run22 $zipImage22 "$zipSrc22/evaluate22.py" @('fixtures')
Run22 $zipImage22 "$zipSrc22/evaluate22.py" @('regression')
Run22 $zipImage22 "$zipSrc22/evaluate22.py" @('continuous')
foreach ($session22 in @(1,2)) {
    Run22 $zipImage22 "$zipSrc22/benchmark22.py" @('benchmark','--session',"$session22")
}
Run22 $zipImage22 "$zipSrc22/benchmark22.py" @('lock')
```

Stream 600 giây lặp từ `0.wav`, có 1.875 chunk, mỗi wrapper khởi tạo riêng rồi giữ state xuyên waveform. Đối chiếu encoder output, token và final counter; không đối chiếu mọi cached-state tensor ở từng chunk. Fixture encoder đối chiếu đủ 36 output.

## Môi trường và kiểm chứng

```powershell
Run22 $mlsdImage22 'daily/2026-10-22/tools/environment22.py' @('mlsd')
Run22 $zipImage22 'daily/2026-10-22/tools/environment22.py' @('zipformer')
.venv-day09/Scripts/python.exe daily/2026-10-22/tools/verify22.py
ruff check daily/2026-10-22
ruff format --check daily/2026-10-22
```

Không chạy experiment khác trong lúc benchmark. Phân rã I/O thực hiện riêng sau timing; không cộng median từng bước hoặc suy thời gian node LiteRT từ profile ONNX Runtime. Quy tắc chất lượng và lựa chọn được ghi tại [protocol](PROTOCOL.md); kết luận ở [báo cáo](REPORT.md).
