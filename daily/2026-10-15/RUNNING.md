# Tiếp tục QAT 15/10

Phần đầu mô tả lượt continuation/export ban đầu. Kết quả research bổ sung nằm ở cuối trang và [RESEARCH.md](RESEARCH.md); [REPORT.md](REPORT.md) là báo cáo cập nhật.

Dùng `.venv-day09` của profile `runtime` đã khóa trong [requirements-lock.txt](../../environments/runtime/requirements-lock.txt). Nếu cần setup: `python scripts/setup.py --profile runtime`. Không cần thêm dependency hay GPU.

Cần giữ các artifact ngày 13–14 theo [RUNNING 14/10](../2026-10-14/RUNNING.md): dữ liệu Wireframe/YorkUrban và reference, checkpoint M-LSD gốc, weight DistilBERT chuyển từ TF, dữ liệu SST-2 và hai checkpoint QAT bước 8. Model/dataset/checkpoint/tensor thô không đưa vào Git.

Từ root repo, trên máy chưa có kết quả ngày 15:

```powershell
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/train_continue.py --task mlsd --targets 32 64 128
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/train_continue.py --task text --targets 128
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/export_integer.py --checkpoint daily/2026-10-15/experiments/qat/artifacts/mlsd_step0032.pt
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/verify.py
```

Chạy tuần tự, tránh chạy training và benchmark cùng lúc. Các mốc là **tổng số bước**, gồm 8 bước ngày 14; hôm nay M-LSD và DistilBERT đều thêm 120 bước. Runner từ chối ghi đè mốc training đã tồn tại. Muốn tiếp tục về sau, dùng `--resume-from <checkpoint>` cùng các target lớn hơn; không cần tạo lại checkpoint ngày 14. File ngày 14 không bị sửa.

Training dùng đúng mô hình/loss/dataset loader ngày 14. Khôi phục cả optimizer và Torch RNG, giữ permutation seed 20261014 và lặp lại sau khi hết train; không reshuffle giữa các epoch. M-LSD batch 1, BCE mềm center + Smooth L1 displacement, pseudo-label teacher. DistilBERT batch 2, 0,5 CE + 0,5 KL distillation T=2, embedding bị freeze. AdamW lr=1e-5, weight decay=0,01, gradient clip=1. Observer vẫn freeze, fake quant vẫn bật.

Mỗi mốc lưu checkpoint model/optimizer/RNG, raw output trước/sau, loss/gradient từng bước, số mẫu, hash và metric. `verify.py` kiểm tra chuỗi resume, hash/weight, observer, chỉ số train, dữ liệu text không trùng calibration/evaluation và tính lại metric từ output đã lưu. Không có dữ liệu evaluation dùng cho gradient; subset được theo dõi nhiều lần nên chưa phải tập test cuối độc lập.

Export M-LSD dùng ONNX opset 17 và TorchScript symbolic để biểu diễn rõ `ConvInteger`. Weight giữ scale per-channel và clamp −127…127; activation giữ scale/zero-point đã train. Cộng offset 128 cho mã signed và zero-point để chạy kernel UINT8 của ORT CPU, tích lũy INT32; nhân lại scale và cộng bias ở float. Đây là mixed integer/float, không phải full INT8 và không thêm một lượt PTQ calibration. Output center/displacement giữ QDQ riêng.

Float reference của phép chuyển mã khớp output fake quant trên 16 ảnh. Khi chạy integer thực tế, relative L2 khoảng 8,40%, vượt gate 2%; sAP10 giảm khoảng 0,28 điểm phần trăm, nằm trong gate 0,5 điểm. Kết quả ghi rõ `exported_parity_gate_failed`. Một phép kiểm tra activation trên ảnh đầu cho thấy QDQ của ORT có một số giá trị lệch một nấc so với PyTorch ở cả signed và unsigned; chưa phân tách hết tác động của làm tròn/accumulation qua mạng. Không quy toàn bộ độ lệch cho cách lưu UINT8.

Audit kiểm graph trước/sau tối ưu và profiler: 41 ConvInteger được chạy trên CPU, không có Conv float. Residual, ReLU/Clip, Resize, scale, bias và I/O còn float. Benchmark hai ONNX model trên cùng 16 ảnh NCHW `1×4×512×512`, batch 1, 4 intra/1 inter threads, sequential, ORT ALL optimizations. Mỗi model warm-up 16 lượt rồi đo 5 vòng × 16 ảnh, đảo thứ tự model giữa các ảnh, không bật profiler. JSON giữ 80 thời gian thô/model, median và p95; không gồm load, preprocess, decode line hay Android. FP32 đối chứng là checkpoint pretrained gốc, không phải weight QAT tắt fake quant. Đây là phép đo cách lowering hiện tại, chưa phải một runtime integer đã tối ưu.

`verification.json: status=passed` nghĩa là bằng chứng nhất quán; trường `export_parity_gate_passed=false` vẫn ghi rõ chất lượng chuyển đổi chưa đạt gate. Hash source ở các run được giữ theo lúc ghi kết quả; trainer ngày 15 có được Ruff format sau lượt M-LSD, nên verifier ghi riêng khác biệt này. Source lịch sử ngày 14 giữ nguyên.

Nguồn định nghĩa operator: [ONNX ConvInteger](https://onnx.ai/onnx/operators/onnx__ConvInteger.html). Tham khảo cách biểu diễn quantized graph và kiểm tra độ lệch: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Research bổ sung: range và loss

Giữ toàn bộ kết quả ban đầu. Research ghi vào `experiments/qat/results/research/` và artifact vào `experiments/qat/artifacts/research/`. Dùng checkpoint bước 32, không thay dữ liệu/decoder của lượt đầu. Đã chuẩn bị thêm 32 ảnh YorkUrban ở revision cố định, loại 16 ảnh development theo ID và hash.

Các lệnh đã chạy theo thứ tự bên dưới. Chạy lại nghiên cứu đầy đủ trên một bản sao riêng có thư mục research trống; giữ JSON đã công bố để đối chiếu. Các bước `select` và `evaluate` từ chối ghi đè quyết định hoặc kết quả holdout đã có. Điều này giúp tránh chọn lại model sau khi xem điểm holdout. Download cần kết nối Internet; các bước khác dùng artifact local.

```powershell
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/research_mlsd.py --stage scan
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/prepare_holdout.py
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/research_mlsd.py --stage center-scan
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/research_mlsd.py --stage train --variant range_only
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/research_mlsd.py --stage train --variant loss_only
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/research_mlsd.py --stage train --variant range_and_loss
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/evaluate_research.py --stage select
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/export_integer.py --checkpoint daily/2026-10-15/experiments/qat/artifacts/research/selected_checkpoint.pt --artifact-dir daily/2026-10-15/experiments/qat/artifacts/research/deploy --result-file daily/2026-10-15/experiments/qat/results/research/deployment.json --disable-spinning
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/evaluate_research.py --stage evaluate
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/benchmark_research.py
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/verify_research.py
```

Research gồm 8 ablation ban đầu, 3 policy center/range, 3 nhánh train × 96 bước với mốc 64/128, rồi chọn trên development. Control range cũ/loss cũ ở bước 64/128 lấy từ lượt continuation ban đầu, cùng checkpoint bước 32 và sampling. Không dùng 32 ảnh mới để chọn nhánh hay mốc. Checkpoint tốt nhất sau lựa chọn chỉ đổi range, có cùng hash parameter với QAT bước 32; optimizer/RNG vẫn được giữ trong checkpoint.

`verify_research.py` tính lại metric của 11 ablation, 6 mốc train và 7 model trên 32 ảnh mới; kiểm hash, sample order, optimizer step, observer/fake-quant flags, model selection, graph/profiler và tính lại bootstrap. `status=passed` nói về tính nhất quán của bằng chứng; dense-output parity gate vẫn thất bại và precision vẫn mixed.

Phép đo tốc độ dùng trong báo cáo là `matched_benchmark.json`: chỉ một session/thread pool tồn tại mỗi lượt, 5 vòng xoay/đảo thứ tự ba model, mỗi model mỗi vòng khởi tạo session rồi warm-up 16 ảnh và đo 16 ảnh. Tổng 80 mẫu/model; không có training/holdout/bootstrap chạy cùng benchmark. Hai lượt thử nhiều pool cùng tồn tại hoặc tắt spinning được giữ ở `matched_benchmark_shared_pools.json` và `deployment.json`; không trộn số ms giữa các protocol hay với lượt đo đầu ngày.

Biểu đồ là artifact SVG độc lập, tạo bằng Matplotlib từ JSON. Có thể tạo lại tùy chọn bằng Python có Matplotlib:

```powershell
python daily/2026-10-15/experiments/qat/src/plot_research.py
```

Matplotlib chỉ dùng vẽ hình, không tham gia training, inference, AP hoặc bootstrap. Source phương pháp: [paper M-LSD](https://arxiv.org/html/2106.00186) và [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/). Quy tắc foreground calibration cụ thể là giả thuyết của thực nghiệm này, đã đánh giá bằng đối chứng và subset mới.

## Kiểm tra và sửa biểu diễn lượng tử khi export

Hướng đã chọn: giữ checkpoint, range, weight và decoder; chỉ sửa cách biểu diễn làm tròn để khớp fake quant CPU. Kiểm tra riêng quantizer và convolution trước khi kết luận từ output toàn mạng. Không thêm training hoặc dùng lại holdout để chọn cấu hình. Đích kiểm chứng vẫn là relative L2 ≤2% và chênh sAP10 tuyệt đối ≤0,5 điểm; không thay gate theo kết quả.

Lượt export bổ sung dùng artifact local của research phía trên. Không cần tải dữ liệu hoặc thêm dependency. Chạy từ root repo, tuần tự:

```powershell
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/export_parity.py
.venv-day09/Scripts/python.exe daily/2026-10-15/experiments/qat/src/verify_export_parity.py
```

`export_parity.py` từ chối ghi đè `results/export-parity/rounding.json` đã có. Để chạy đối chiếu riêng, dùng `--artifact-dir <thư-mục-artifact-mới>` và `--result-dir <thư-mục-kết-quả-mới>`; verifier mặc định kiểm bộ kết quả đã công bố ở `export-parity/`. Các model/output/profiler trace được ignore; JSON nhỏ ghi metric, hash và giới hạn được lưu trong repo.

Exporter thay `QuantizeLinear` bằng nhân nghịch đảo scale FP32, `Round` về số chẵn, cộng zero-point, clip và cast về dtype gốc. Per-channel output được broadcast theo NCHW axis 1. Constant gốc, node không phải quantizer, checkpoint và decoder giữ nguyên. Model vẫn có 41 ConvInteger UINT8×UINT8→INT32 với boundary float; không phải full integer.

Verifier dùng PyTorch fake quant làm oracle cho các ca điểm giữa, lân cận FP32, clipping và scale theo kênh; kiểm cùng đầu vào của 43 quantizer và 41 convolution ở ảnh development đầu tiên; đối chứng thêm backend PyTorch trên đủ 16 ảnh; rồi tái chạy ONNX toàn tập development, tính metric, kiểm profiler và hash baseline/selection/holdout. Sửa làm tròn giảm relative L2 từ 5,9066% xuống 5,6565%, **vẫn không đạt gate 2%**. Không chạy benchmark hoặc đánh giá độc lập mới trong lượt này.

Hai script mới đã qua `ruff check`, `ruff format --check`; verifier xác nhận output ONNX tái lập và 41 ConvInteger được thực thi. Môi trường runtime đã khóa không có Ruff: dùng lệnh `ruff` của môi trường công cụ khi kiểm style, không cài thêm vào profile runtime.
