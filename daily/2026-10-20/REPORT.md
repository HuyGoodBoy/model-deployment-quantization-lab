# Báo cáo ngày 20/10/2026 — Zipformer LiteRT và tối ưu M-LSD

Người thực hiện: Bùi Gia Huy.

Em đã chuyển đủ encoder, decoder và joiner của Zipformer 20M sang LiteRT 2.1.2. Bản convert trực tiếp gặp lỗi layout và grouped convolution; sau khi sửa, token và WER giữ nguyên so với ONNX trên tập đã khóa. Bản sửa chạy nhanh hơn ONNX trong phép đo cùng pipeline; thử tối ưu decoder bằng bảng tra cứu không đem lại lợi ích nên chưa chọn.

Với M-LSD, em profile từng convolution của bản QDQ đã fuse, rồi thử giữ các lớp chậm ở FP32 bằng AI Edge Quantizer. Recipe chọn lọc đã áp dụng đúng, nhưng chưa cải thiện đồng thời tốc độ và chất lượng. Kết quả tăng tốc có ích là giữ nguyên model FP32 và tăng từ 4 lên 8 luồng CPU: median giảm 22,2%, output trên development không đổi.

## 1. Môi trường và cách đo

CPU i7-12700H, 20 logical CPU. Các lượt Linux chạy trong Docker/WSL2; profile ONNX M-LSD chạy trên Windows. Chưa đo Android, GPU hoặc NPU trong ngày này.

| Thực nghiệm | Thư viện chính | Cấu hình |
|---|---|---|
| Zipformer ONNX → LiteRT | onnx2tf 2.6.9, ORT 1.26.0, LiteRT 2.1.2 | CPU 4 luồng, batch 1 |
| Profile M-LSD QDQ/FP32 | ORT 1.30.0 trên Windows | 4 intra/1 inter, sequential, tắt spinning |
| M-LSD lượng tử hóa và benchmark | AI Edge Quantizer 0.9.0, LiteRT 2.2.0 trên Linux | CPU/XNNPACK, 4 luồng; thử thêm 1/2/8 luồng riêng |

Benchmark chạy tuần tự, không có thí nghiệm khác cạnh tranh tài nguyên. Benchmark component và M-LSD có warm-up, xoay/đảo thứ tự qua 5 vòng, lưu cả median và p95; RTF toàn pipeline ASR đo 3 vòng riêng. Không dùng chênh lệch Windows ORT với Linux LiteRT để kết luận riêng về hiệu quả INT8. Phiên bản package, image và cấu hình máy nằm trong [environments](environments/).

## 2. Chuyển Zipformer 20M sang LiteRT 2.x

### Model và pipeline tham chiếu

Model được giao là [sherpa-onnx-streaming-zipformer-en-20M-2023-02-17](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17), revision `d42f2d9f7ca24806fb667456a18a9f1b60f70d16`. Em dùng đúng ba file ONNX FP32 và kiểm SHA256 với nguồn trước khi chạy; giữ nguyên model nguồn, sửa trên bản sao.

- Encoder nhận đặc trưng `[N,39,80]` cùng 35 state, mỗi chunk tiến 32 frame.
- Decoder dùng context 2 token, vocabulary 500, input INT64; joiner kết hợp hai vector 512 chiều.
- Frontend dùng 80-bin fbank, 16 kHz, dither 0, thêm 0,3 giây silence cuối; audio 8 kHz được resample. Greedy tối đa một token mỗi frame, carry đủ state giữa chunk và reset giữa các câu.

Đường chuyển dùng native FlatBuffer backend của onnx2tf. Kiểm tra gồm tạo file, allocate, invoke, sai số tensor, streaming nhiều chunk, transcript/WER và tốc độ; không coi CLI trả thành công là model đã chạy được.

### Lỗi convert thông thường và cách xử lý

| Thành phần | Lỗi quan sát được | Cách xử lý và kết quả |
|---|---|---|
| Encoder | Converter tự suy input thành `[1,80,39]`; allocate lỗi CONCAT với kích thước 128 so với 1 | Cố định batch 1 đúng trên từng trục state, giữ chunk 39 và dùng `-kat` để giữ layout input; allocate/invoke được |
| Decoder | Có file `.tflite` và exit code 0 nhưng chứa custom `ONNX_CONV`, LiteRT chuẩn không allocate được | Strict conversion xác định grouped Conv rank 3, group 128, chưa được lowering hỗ trợ |
| Decoder sau sửa | Cần giữ nguyên phép tính của Conv1D | Mở thêm chiều không gian: NCW → NC1W, weight `[512,4,2]` → `[512,4,1,2]`; giữ group/weight/stride/padding rồi squeeze output. Kết quả dùng native Conv2D |
| Joiner | Không gặp lỗi thực thi | Convert và invoke được, không cần sửa toán học |
| State counter | Output `cached_len` của LiteRT là INT32, input vẫn yêu cầu INT64 | Chỉ widen INT32 → INT64 tại đúng counter trước chunk tiếp theo; không cast INT64 hàng loạt |

Decoder sửa được kiểm trên 18 context; max absolute error so ONNX khoảng `3,34e-6`. Encoder được so trên 15 input/state lấy từ audio thực, đủ 36 output mỗi fixture: float đạt allclose với rtol/atol `1e-4`, counter nguyên khớp exact. Counter cuối của cả 73 câu cũng khớp, nhưng chưa kiểm session liên tục dài đến giới hạn INT32.

Các bước chuẩn bị shape/layout cho phép converter simplify và fold constants. Em chưa tách riêng đóng góp tốc độ của từng bước này. Lỗi, lệnh convert và kết quả thực thi được lưu trong [raw validation](experiments/zipformer/results/runtime_validation_raw.json), [strict decoder error](experiments/zipformer/results/strict_decoder_error_modelproto.json) và [patch Conv2D](experiments/zipformer/results/decoder_conv2d_repair.json).

### Chất lượng trước và sau chuyển đổi

Tập đánh giá được khóa trước khi thử tối ưu: toàn bộ 73 câu của LibriSpeech dummy validation, một người đọc, tổng 1.150 từ tham chiếu. WER được tính bằng tổng lỗi thay/xóa/chèn từ chia tổng từ tham chiếu, không lấy trung bình phần trăm từng câu.

| Model | Thay / xóa / chèn từ | WER | Token so với ONNX |
|---|---:|---:|---|
| ONNX FP32 | 75 / 13 / 3 | 91/1.150 = 7,9130% | Mốc tham chiếu |
| LiteRT sửa Conv2D | 75 / 13 / 3 | 7,9130% | Khớp 73/73 câu |
| LiteRT decoder bảng tra cứu | 75 / 13 / 3 | 7,9130% | Khớp 73/73 câu |

Cả ba audio đi kèm model, gồm audio 8 kHz, cũng khớp token. Đây là đánh giá trên tập nhỏ một speaker, không phải WER LibriSpeech đầy đủ hoặc chứng minh chất lượng trên dữ liệu độc lập với pretraining.

Pipeline chung dùng context khởi tạo `[0,0]` theo ví dụ ONNX Icefall. Native Sherpa dùng `[-1,0]`, dẫn đến transcript khác ở đầu câu. Khi chỉ đổi context sang `[-1,0]`, transcript ONNX khớp native trên cả ba audio thử. Vì vậy các số chính so sánh trước/sau conversion trên cùng pipeline Icefall-style; không gọi là cải thiện recognizer native Sherpa. [Kết quả chẩn đoán](experiments/zipformer/results/native_context_diagnosis.json).

### Thử tối ưu graph decoder và đo tốc độ

Em thử tính trước embedding và grouped Conv tuyến tính thành hai bảng 500×512. Khi chạy, Gather + Add thay cho Conv, còn ReLU và projection giữ nguyên. Bản này qua kiểm tra 64 context, max error khoảng `4,29e-6`, giữ nguyên WER nhưng decoder tăng dung lượng từ 2,098 lên 3,102 MB.

Mỗi component được warm-up 16 lần rồi đo 32 lần/vòng, tổng 160 mẫu. RTF là thời gian toàn pipeline chia thời lượng audio; bao gồm frontend, reset state và greedy, loại thời gian load model và chuẩn bị cache metadata. RTF trong bảng là median 3 vòng trên **ba audio thử**, khác tập 73 câu dùng tính WER.

| Cấu hình | Encoder median / p95 (ms) | Decoder median / p95 (ms) | RTF median |
|---|---:|---:|---:|
| ONNX FP32 | 15,888 / 17,968 | 0,09597 / 0,12009 | 0,06175 |
| LiteRT sửa Conv2D | 10,053 / 10,825 | 0,01780 / 0,58188 | 0,03959 |
| LiteRT decoder bảng tra cứu | 9,903 / 11,208 | 0,02618 / 0,64207 | 0,04042 |

Bản LiteRT sửa giảm khoảng 35,9% thời gian toàn pipeline so ONNX trong cấu hình đã đo. Đây là kết quả đổi runtime/kernel và graph cùng nhau, chưa quy riêng cho một rewrite. P95 decoder LiteRT vẫn cao hơn ONNX dù median thấp hơn.

**Chưa chọn bản bảng tra cứu:** decoder chậm hơn, file lớn hơn và RTF không cải thiện. Encoder của hai bản LiteRT là cùng artifact; chênh nhỏ giữa hai số đo encoder là biến thiên đo. Bảng dùng [benchmark v2](experiments/zipformer/results/matched_benchmark_v2.json), đã đưa việc parse/cache metadata ra ngoài timing; lần đo đầu được giữ làm lịch sử, không dùng để kết luận.

## 3. M-LSD: operator nào chậm sau khi QDQ đã fuse?

Em so graph QDQ với graph chạy float từ **cùng weight QAT**, tắt fake quant ở control FP32; không dùng teacher có weight khác. Profile sau 16 lượt warm-up, lấy 8 lượt cuối, ghép đủ 41 convolution theo identity của weight/output.

Profiler xác nhận bản QDQ thực thi **22 QLinearConv integer và 19 Conv float**. Các lớp có overhead đáng chú ý:

| Lớp | Kernel trong QDQ | FP32 trung bình (ms) | QDQ trung bình (ms) | Nhận xét |
|---|---|---:|---:|---|
| block13.conv1.0 | FusedConv float | 5,4830 | 8,0785 | Đã dùng kernel float, vẫn chậm hơn control |
| features2.conv0.0 | FusedConv float | 2,0568 | 3,8315 | Biên fusion/layout khác FP32 NCHWc |
| features0.0 | QLinearConv | 1,3010 | 2,1183 | Chọn thử giữ FP32 |
| features1.conv0.0 | QLinearConv | 0,7681 | 1,1130 | Chọn thử giữ FP32 |
| features3.conv1.0 | QLinearConv | 0,9025 | 1,1816 | Chọn thử giữ FP32 |

Thời gian trên gồm overhead và các biên fusion khác nhau, không phải microbenchmark cô lập phép nhân integer. Một lớp dùng kernel float vẫn có thể nhận tensor đã qua Q/DQ; chưa đánh giá việc bỏ Q/DQ tại toàn bộ các lớp float này. Kết quả không ủng hộ kết luận model chậm chỉ vì “chưa fuse”. [Bảng đủ 41 lớp](experiments/mlsd/results/ort_conv_comparison.csv).

## 4. Dùng AI Edge Quantizer giữ lớp chậm ở float

AI Edge Quantizer xử lý LiteRT float, không chỉnh trực tiếp model ONNX QDQ. Em convert graph FP32 cùng weight QAT sang LiteRT trước: relative L2 so ONNX là `1,3796e-5`, tương đương 0,00138%. Sau đó chạy PTQ mới với 32 ảnh Wireframe calibration; không giữ các scale QAT cũ.

Từ profile ONNX, em chọn ba Conv integer chậm hơn control để thử float trên LiteRT. Đây là giả thuyết chuyển sang backend khác, nên vẫn phải đo lại. Đã kiểm graph sau export: bản static có 41 Conv weight/output INT8; bản selective có **38 Conv INT8 và đúng 3 Conv FP32**. Recipe đầu thiếu dấu `;` cuối scope không khớp và đã bị loại bằng kiểm dtype trước benchmark.

### Tốc độ và sAP10

Giữ cùng input, decoder đoạn thẳng và protocol sAP10. Dùng 16 ảnh YorkUrban development để chọn, khóa quyết định rồi mới chạy 54 ảnh hồi quy đã dùng trước đây. sAP10 là AP phát hiện đoạn thẳng, không phải tỷ lệ ảnh dự đoán đúng; tập này không phải benchmark paper đầy đủ.

Benchmark 5 vòng, mỗi vòng 16 warm-up và 16 lượt đo, tổng 80 mẫu/model. Timing gồm adapter quantize/dequantize I/O nếu cần; không gồm load model, tiền xử lý ảnh hay giải mã đoạn thẳng.

| Model LiteRT 2.2, 4 luồng | Median / p95 (ms) | Dung lượng (bytes) | sAP10 dev16 | sAP10 hồi quy54 |
|---|---:|---:|---:|---:|
| FP32 cùng weight QAT | 29,233 / 34,771 | 2.483.636 | 15,3230% | 13,2033% |
| AEQ static | 14,661 / 17,185 | 792.192 | 2,8522% | 2,3967% |
| AEQ selective: 3 lớp float | 16,139 / 18,556 | 795.856 | 3,0852% | 2,6284% |
| AEQ dynamic | 43,441 / 48,691 | 723.120 | 14,7161% | 13,1471% |

Giữ ba lớp float làm median tăng khoảng 10,1% so static; sAP chỉ tăng nhẹ và vẫn rất thấp so FP32. Dynamic giữ sAP gần float trên hồi quy nhưng chậm hơn. Chưa có phương án INT8 mới vừa nhanh vừa giữ chất lượng để thay FP32.

Quy tắc chọn không tìm được alternative nhanh hơn static và đủ ngưỡng sAP development, nên registry giữ `selected=static` làm baseline thí nghiệm. Điều này **không có nghĩa static được duyệt triển khai**. Tập 54 ảnh là hồi quy cũ, không phải holdout độc lập mới; không chọn lại model sau khi xem kết quả này.

## 5. Nghiên cứu nguyên nhân mất sAP và hướng tăng tốc thêm

### Range INT8 chung của đầu ra

Static/selective dùng một output scale khoảng **2,3351** cho cả center logit và bốn kênh displacement. Để kiểm tra riêng ảnh hưởng của biên này, em lấy output FP32 và chỉ áp dụng quantize/dequantize theo scale thực, giữ nguyên các phần còn lại.

| Thử trên 16 ảnh development | sAP10 |
|---|---:|
| Output FP32 gốc | 15,3230% |
| Chỉ thêm INT8 round-trip tại output FP32 | 2,3885% |
| Toàn graph AEQ static | 2,8522% |
| Thử giữ final Conv/output float cùng ba lớp đã chọn | 3,0139% |

Bước lượng tử quá lớn ở output là một nguồn sai số đủ nghiêm trọng. Tuy nhiên, chỉ giữ lớp cuối/output ở float chưa khôi phục sAP, cho thấy còn sai số ở các lượng tử hóa phía trước; chưa quy toàn bộ lỗi cho output boundary. Nhánh giữ output float là thăm dò riêng **chỉ trên development**, không dùng 54 ảnh để chỉnh tiếp và chưa benchmark vì chưa đạt chất lượng.

### Tối ưu số luồng trên cùng model FP32

Em giữ nguyên artifact FP32, chỉ đổi số luồng LiteRT; cùng 5 vòng xoay/đảo, 80 mẫu/điều kiện:

| Luồng CPU | Median (ms) | P95 (ms) |
|---:|---:|---:|
| 1 | 102,269 | 106,093 |
| 2 | 53,183 | 61,134 |
| 4 | 28,379 | 30,781 |
| 8 | **22,079** | **23,614** |

Bản 8 luồng giảm **22,2% median** so 4 luồng trong cùng phép đo. Kiểm lại 16 ảnh development cho tensor exact và sAP10 vẫn 15,3230%. Đây là tăng tốc nhờ cấu hình CPU, không phải lợi ích của INT8. Chưa đo điện năng, nhiệt độ, pin hoặc hiệu quả trên Android. [Kết quả thread tuning](experiments/mlsd/results/float_thread_tuning.json).

## 6. Kiểm chứng, phần chưa xong và việc tiếp theo

Đã tái tính WER từ transcript, sAP của cả tám kết quả development/hồi quy, median/p95 từ sample gốc; kiểm raw profiler trace, optimized graph, SHA256 model/output và thứ tự khóa chọn trước hồi quy. Kiểm chứng bằng chứng đạt, không đồng nghĩa các model INT8 đạt chất lượng triển khai. Vấn đề export parity của ONNX QDQ đã ghi ngày 19 vẫn chưa được giải quyết trong ngày này.

Trong lần convert M-LSD đầu, onnx2tf tự simplify ghi đè input. Em đã lưu bản bị sửa riêng, tái export từ checkpoint/code cũ và chỉ phục hồi nguồn sau khi SHA256 trùng hoàn toàn manifest ngày 19. Convert lại trên bản sao cho LiteRT byte-identical với bản đã đo. Các lần sau chỉ convert bản sao; [bằng chứng phục hồi và tái lập](experiments/mlsd/results/float_conversion_reproduction.json).

Yêu cầu dùng Claude tối ưu graph **chưa thực hiện được** vì CLI chưa đăng nhập: lượt gọi trả `Not logged in`, không có API inference hoặc tool execution. Các bản sửa graph ở trên đã được thử độc lập, không gán cho Claude. Công cụ thực nghiệm đã dùng gồm onnx2tf native backend, ONNX/NumPy, ORT profiler, LiteRT Interpreter và AI Edge Quantizer; [lệnh gọi và trạng thái Claude](experiments/zipformer/results/claude_execution.json) được giữ riêng.

Tiếp theo em giữ Zipformer bản sửa Conv2D và M-LSD FP32 8 luồng cho CPU này. Cần thử precision theo cả vùng head hoặc tách range center/displacement, mở rộng calibration phù hợp; đồng thời kiểm dữ liệu độc lập, session ASR dài và thiết bị đích. Phần tối ưu graph bằng Claude còn cần thực hiện sau khi đăng nhập, rồi kiểm lại tensor/state, WER và tốc độ trước khi chọn patch.

[Nghiên cứu chi tiết](RESEARCH.md) · [Source và kết quả Zipformer](experiments/zipformer/) · [M-LSD](experiments/mlsd/) · [Kiểm chứng số liệu](VERIFICATION_final.json) · [Cách chạy](RUNNING.md)
