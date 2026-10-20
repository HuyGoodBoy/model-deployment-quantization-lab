# Thực nghiệm ngày 20/10/2026

Đã chuyển được model được giao sang LiteRT 2.x và kiểm chứng trên pipeline ONNX chung. Các thử nghiệm AEQ chọn lọc chưa cho phương án INT8 vừa nhanh vừa giữ chất lượng. FP32 8 luồng là cấu hình tăng tốc có ích trên CPU đã đo. **Phần yêu cầu Claude tối ưu graph còn thiếu do chưa đăng nhập**, không coi toàn bộ task tuần đã hoàn thành.

## Môi trường và nguyên tắc so sánh

CPU Intel Core i7-12700H, 20 logical CPU, Docker Linux/WSL2 x86-64; các lần benchmark chạy tuần tự. Không đo Android, GPU, NPU, RAM hay năng lượng.

| Nhánh | Runtime/thư viện thực dùng | Cấu hình |
|---|---|---|
| Zipformer | onnx2tf 2.6.9, LiteRT 2.1.2, ORT 1.26.0, Python 3.12.15 | CPU 4 luồng; ORT sequential, 1 inter, tắt spinning |
| Profile M-LSD ONNX | Windows ORT 1.30.0 | CPU 4 intra/1 inter, tắt spinning, warm16 rồi profile8 |
| M-LSD LiteRT | LiteRT 2.2.0, AI Edge Quantizer 0.9.0, Python 3.11.17 | CPU/XNNPACK, 4 luồng; thử thêm 1/2/4/8 luồng riêng |

Không so latency tuyệt đối giữa Windows ORT và Linux LiteRT để quy riêng cho INT8. Image ID, CPU, Python và `pip freeze` nằm trong [environments](environments/). Dockerfile Zipformer pin base digest và toàn bộ package lock. Native FlatBuffer backend không chuyển qua TensorFlow. Đầu vào/đầu ra ứng dụng M-LSD luôn float; adapter lượng tử/giải lượng tử dùng scale/zero-point thực khi model có I/O INT8, và chi phí adapter có trong timing.

## Zipformer: bản thường lỗi gì và đã sửa gì

Nguồn chính xác: [HF model được giao](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17), revision `d42f2d9f7ca24806fb667456a18a9f1b60f70d16`. Ba file FP32 được kiểm SHA256 với LFS upstream, không thay bằng model cùng tên/kiến trúc khác. [Source manifest](experiments/zipformer/results/source_manifest.json).

Encoder có `x=[N,39,80]` và 35 state, hop32. Decoder context2, vocab500, y INT64; joiner nhận hai vector512. Giữ nguyên frontend KNF80, 16kHz, dither0, snip_edges=false, high_freq=-400, thêm 0,3s silence; audio8k resample về16k. Greedy tối đa một symbol/frame; reset state mỗi utterance, carry đủ35 state giữa chunk.

| Bản/thành phần | Kết quả thực thi | Xử lý |
|---|---|---|
| Convert nguyên trạng: encoder | Allocate lỗi CONCAT, kích thước128 so với1; converter suy layout input thành `[1,80,39]` | Cố định đúng batch1 trên từng trục state, giữ chunk39; `-kat` toàn bộ input để giữ layout |
| Convert nguyên trạng: decoder | CLI trả0 và tạo `.tflite`, nhưng allocate lỗi custom `ONNX_CONV` | Strict lowering chỉ ra `unsupported_grouped_convolution`: Conv rank3, group128, weight `[512,4,2]` |
| Decoder sửa | Native Conv2D allocate/invoke được | NCW → NC1W, weight `[512,4,1,2]`, giữ group128/weight/stride/dilation/padding; squeeze về shape cũ |
| Joiner | Convert và invoke được từ đầu | Không sửa toán học |
| State counter | LiteRT trả cached_len INT32 nhưng input vẫn INT64 | Adapter chỉ widen INT32 → INT64 cho đúng tên cached_len; các dtype mismatch khác báo lỗi |

Không cast toàn bộ INT64. Giá trị counter khớp exact trên15 fixture thực và cuối cả73 câu; chưa thử session liên tục dài đến giới hạn INT32. Widen đầu vào không sửa được overflow nếu xảy ra bên trong graph, nên chưa khẳng định an toàn cho mọi độ dài.

[onnx2tf](https://github.com/PINTO0309/onnx2tf) có fallback custom ops trong đường CLI đã dùng. Vì vậy tạo được file và exit0 không chứng minh model chạy trên LiteRT chuẩn. Đã lưu [runtime validation raw](experiments/zipformer/results/runtime_validation_raw.json), [static](experiments/zipformer/results/runtime_validation_static.json), [strict error đúng ModelProto](experiments/zipformer/results/strict_decoder_error_modelproto.json) và [patch Conv2D](experiments/zipformer/results/decoder_conv2d_repair.json). `strict_decoder_error.json` cũ là lỗi cách gọi probe Graph thay vì ModelProto, không dùng làm nguyên nhân model thất bại.

### Chất lượng trước/sau

Khóa trước thử tối ưu toàn bộ73 câu validation public LibriSpeech dummy, một speaker, 1.150 từ, revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`. Không lọc theo model. Đây là tập nhỏ; không gọi là WER LibriSpeech đầy đủ hoặc chứng minh dữ liệu chưa xuất hiện trong pretraining.

| Cấu hình | S/D/I | WER | Token so ONNX | State counter cuối73 câu |
|---|---|---:|---|---|
| ONNX FP32 | 75/13/3 | 91/1150 = 7,9130% | Mốc | Mốc |
| LiteRT bản sửa Conv2D | 75/13/3 | 7,9130% | Khớp73/73 | Exact |
| LiteRT decoder lookup | 75/13/3 | 7,9130% | Khớp73/73 | Exact |

Cả3 audio đi kèm model, gồm audio8k, cũng khớp token. So encoder cùng15 input/state thực, đủ36 output/fixture: float allclose rtol/atol1e-4; counter nguyên exact dù dtype khác. Decoder sửa qua18 context; lookup qua64 context với token biên và seed cố định. Sai số lớn nhất decoder so ONNX lần lượt khoảng3,34e-6 và4,29e-6.

Pipeline chung dùng context khởi tạo `[0,0]` theo [ví dụ Icefall ONNX](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/onnx_pretrained.py). [Sherpa greedy](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc) dùng `[-1,0]` và còn xử lý unk như blank. Thử thay duy nhất context trong pipeline ONNX làm transcript khớp native Sherpa trên cả3 audio; xem [diagnosis](experiments/zipformer/results/native_context_diagnosis.json). Vì vậy các số WER/RTF chính là so conversion trên pipeline Icefall-style chung, không tuyên bố tương đương mặc định native Sherpa hoặc cải thiện native recognizer.

### Thử tối ưu graph và tốc độ

Phép sửa tương thích đã cố định shape, cho phép converter simplify/fold constants và chọn layout đúng. Thử thêm tối ưu đại số: embedding cố định và grouped Conv tuyến tính được tính trước thành hai bảng500×512; Gather+Add thay Conv, giữ ReLU/projection. Đây là patch **Codex**, không phải Claude. Dung lượng decoder tăng từ2.098MB lên3.102MB. Không chọn bản lookup vì decoder chậm hơn và RTF không cải thiện.

Benchmark v2: cùng CPU4 luồng, 5 vòng xoay/đảo thứ tự; mỗi component16 warm-up +32 lượt đo/vòng, tổng160 mẫu. Chỉ tạo một bộ runtime tại một thời điểm. RTF gồm file/audio frontend, reset state và greedy, lấy median3 vòng trên **3 audio smoke**, không phải73 câu WER. Model load và cache metadata nằm ngoài timing.

| Cấu hình | Encoder median/p95 ms | Decoder median/p95 ms | RTF median |
|---|---:|---:|---:|
| ONNX | 15,888 / 17,968 | 0,09597 / 0,12009 | 0,06175 |
| LiteRT sửa Conv2D | 10,053 / 10,825 | 0,01780 / 0,58188 | 0,03959 |
| LiteRT lookup | 9,903 / 11,208 | 0,02618 / 0,64207 | 0,04042 |

RTF thấp hơn là nhanh hơn. Bản sửa giảm khoảng35,9% thời gian pipeline so ONNX trên phép đo này, đổi cả runtime/kernel chứ không phải tác động riêng một rewrite. P95 decoder LiteRT cao hơn ONNX dù median thấp; không chỉ báo median rồi suy rộng mọi lượt đều nhanh. Encoder giữa hai bản LiteRT là cùng artifact, chênh timing là biến thiên đo. Bản benchmark đầu chưa prewarm metadata input_specs, lượt đầu ONNX tính cả parse encoder88,8MB; **không dùng số đó**, giữ JSON cũ làm lịch sử và dùng [matched_benchmark_v2.json](experiments/zipformer/results/matched_benchmark_v2.json).

## M-LSD: profile QDQ đã fuse và AEQ chọn lọc

Dùng cùng checkpoint QAT: một graph QDQ cũ và một graph tắt fake quant chạy FP32; không dùng teacher weight khác làm control. QDQ vẫn có vấn đề export parity đã ghi ngày19; profile hôm nay không chứng minh vấn đề đó đã được giải quyết.

ORT thực thi22 `QLinearConv` và19 Conv float. Ghép đủ41 lớp theo weight/output identity. Các lớp chậm hơn control:

| Lớp | Kernel QDQ | FP32 ms | QDQ ms | Diễn giải |
|---|---|---:|---:|---|
| block13.conv1.0 | FusedConv float | 5,4830 | 8,0785 | Đã dùng kernel float; không thuộc nhóm thử đổi kernel INT8 → FP32 |
| features2.conv0.0 | FusedConv float | 2,0568 | 3,8315 | Layout/biên fusion khác FP32 NCHWc |
| features0.0 | QLinearConv | 1,3010 | 2,1183 | Chọn thử float |
| features1.conv0.0 | QLinearConv | 0,7681 | 1,1130 | Chọn thử float |
| features3.conv1.0 | QLinearConv | 0,9025 | 1,1816 | Chọn thử float |

Profile gồm overhead và biên activation/layout khác, không phải microbenchmark cô lập phép nhân integer. Kernel float vẫn có thể nhận tensor đã qua Q/DQ; bỏ lượng tử hóa tại đó là một thay đổi khác, chưa đánh giá hôm nay. ORT profile chỉ đề xuất scope cần thử; không chứng minh kernel tương ứng LiteRT cũng chậm. [CSV đủ41 lớp](experiments/mlsd/results/ort_conv_comparison.csv).

[AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer) nhận **LiteRT float**, không sửa trực tiếp ONNX QDQ. Đã convert cùng QAT-weight FP32, parity relative L2 `1,3796e-5` (0,00138%) so ONNX. Sau đó PTQ mới với32 ảnh Wireframe, không giữ scale QAT cũ. Chọn recipe trên16 ảnh York development rồi khóa trước khi chạy54 ảnh hồi quy cũ.

Đã kiểm graph thực: static41 Conv có weight/output INT8; selective38 INT8 +3 FP32 đúng lớp chọn. Scope AEQ ghép output name và **dấu chấm phẩy cuối**; recipe thiếu dấu này lần đầu không khớp, bị loại bằng kiểm dtype trước benchmark. Đây là xác nhận graph/dtype, không phải profile chi tiết từng kernel XNNPACK.

| Bản LiteRT 2.2, 4 luồng | Median/p95 ms | Size bytes | sAP10 dev16 | sAP10 hồi quy54 |
|---|---:|---:|---:|---:|
| FP32 cùng QAT weights | 29,233 / 34,771 | 2.483.636 | 15,3230% | 13,2033% |
| AEQ static | 14,661 / 17,185 | 792.192 | 2,8522% | 2,3967% |
| AEQ selective3float | 16,139 / 18,556 | 795.856 | 3,0852% | 2,6284% |
| AEQ dynamic | 43,441 / 48,691 | 723.120 | 14,7161% | 13,1471% |

Size đọc lại từ exports JSON là nguồn chuẩn. Timing5 vòng xoay/đảo,16 warm-up+16 đo/vòng,80 mẫu/bản, có adapter I/O, không profile. sAP10 là AP phát hiện đoạn thẳng theo protocol subset đã dùng từ ngày13, **không phải phần trăm ảnh dự đoán đúng**. 54 ảnh là hồi quy đã dùng, không phải holdout độc lập mới.

Selective chậm hơn static khoảng10,1% và chất lượng cả hai rất thấp. Dynamic giữ chất lượng gần float trên hồi quy nhưng chậm hơn float. Quy tắc nội bộ không chọn được alternative vừa nhanh hơn static vừa đạt ngưỡng dev; `selected=static` nghĩa là giữ baseline thử nghiệm, **không phải duyệt triển khai**. Không có bản INT8 mới đủ cơ sở thay FP32.

### Nghiên cứu range đầu ra và hướng tăng tốc thêm

Static/selective dùng chung output scale2,3351 cho cả center logit và4 displacement. Chỉ quantize/dequantize đầu ra FP32 với scale này đã khiến sAP dev15,3230% →2,3885%; đây là một nguồn sai số đủ lớn. Không quy toàn bộ lỗi full graph cho output boundary. Thử bổ sung giữ final Conv và output transpose float cùng3 lớp đã chọn vẫn chỉ đạt3,0139% trên development; các lượng tử hóa phía trước vẫn gây sai số. Thử này khai báo riêng sau candidate lock, **chỉ xem development**, không dùng54 ảnh để điều chỉnh/chọn lại; không benchmark vì chưa qua chất lượng. [Output diagnosis](experiments/mlsd/results/aeq_output_diagnosis.json), [exploratory protocol](experiments/mlsd/results/aeq_float_output_protocol.json).

Thử thêm thay số luồng, giữ **cùng artifact FP32**:

| Luồng LiteRT | Median/p95 ms |
|---:|---:|
| 1 | 102,269 / 106,093 |
| 2 | 53,183 / 61,134 |
| 4 | 28,379 / 30,781 |
| 8 | 22,079 / 23,614 |

5 vòng xoay/đảo,80 lượt/điều kiện; sau khi chọn tốc độ, kiểm lại tensor16 ảnh **exact**, sAP15,3230% giữ nguyên. Cấu hình8 luồng giảm22,2% median so4 luồng trong cùng phép đo. Đây là tăng tốc nhờ tài nguyên CPU, không phải INT8; chưa đo pin/nhiệt/điện năng hoặc Android. [Thread tuning](experiments/mlsd/results/float_thread_tuning.json).

## Công cụ, Claude và giới hạn công việc

- Codex thực hiện bằng PowerShell, Python/NumPy/ONNX, Docker, onnx2tf native backend, ONNX Runtime profiler, LiteRT Interpreter, AI Edge Quantizer và Ruff. Astra lên kế hoạch và review bằng chứng/code theo yêu cầu người dùng.
- Đã gọi thật Claude Code2.1.235: `claude -p --output-format json --permission-mode plan --tools Read,Grep,Glob --no-session-persistence`. Trả mã1, `Not logged in`, API tokens0, modelUsage rỗng. **Không có model inference hay tool Read/Grep/Glob nào của Claude được thực hiện**; các tool trên chỉ là danh sách cho phép. [Prompt](tools/claude_prompt.txt), [execution evidence](experiments/zipformer/results/claude_execution.json). Cần đăng nhập rồi chạy lượt riêng, lưu output/patch và kiểm chứng tiếp; chưa báo đã làm xong yêu cầu Claude.
- onnx2tf lần đầu tự simplify ghi đè input M-LSD ngày19. Đã backup bản bị sửa trong artifact ngày20, tái export từ checkpoint/code cũ và chỉ phục hồi sau khi hash **trùng byte** `ce7b26475cb5eee7e9f5b34d873b51e7401c3cc9d9584e415b5a186e0f49dd1e`. Chạy lại converter trên bản sao cho `.tflite` byte-identical, xác nhận nguồn không đổi; [evidence](experiments/mlsd/results/float_conversion_reproduction.json). Từ đây chỉ convert bản sao.
- [Verifier](tools/verify.py) tái tính WER từ transcript, toàn bộ8 kết quả sAP dev/hồi quy, sample median/p95, kiểm SHA256 nguồn/model/output, integer state exact và timestamp khóa chọn trước hồi quy. `passed=true` là bằng chứng nhất quán, không có nghĩa model INT8 qua gate triển khai.

## Việc tiếp theo và commit

Giữ Zipformer bản sửa Conv2D và FP32 M-LSD8 luồng cho cấu hình CPU này. Sau khi có Claude login, yêu cầu phân tích encoder graph và lưu tool thực dùng; các patch mới phải qua tensor/state/streaming parity rồi WER/timing. Với M-LSD, ưu tiên tách range center/displacement và thử precision theo cả vùng head, calibration đại diện hơn; cần xác nhận trên dữ liệu độc lập/thiết bị đích trước triển khai.

Windows task `Codex-Commit-2026-10-20-1700` đã thực hiện commit `20/10` lúc17:00 Asia/Saigon từ snapshot đã khóa. Theo yêu cầu tiếp theo, bản đó đã được push, rồi REPORT.md được viết lại thành báo cáo chi tiết; các số liệu và artifact thí nghiệm giữ nguyên. Bản chỉnh sửa tài liệu được gộp vào commit `20/10` thay thế, không tạo thêm commit báo cáo riêng. [Schedule ban đầu](tools/schedule.json), [integration guard test](tools/schedule_guard_verification.json).
