# Báo cáo ngày 22/10/2026

Người thực hiện: Bùi Gia Huy.

Hôm nay em tập trung giảm chi phí I/O của M-LSD và kiểm tra cách gọi model ZipFormer. M-LSD cải thiện tốc độ khi đưa quantize input vào model, giữ nguyên output trên các ảnh đã kiểm tra. Với ZipFormer, cách gọi trực tiếp chưa giúp toàn pipeline nhanh hơn rõ ràng nên em giữ cách gọi hiện tại.

## M-LSD ([mã nguồn](experiments/mlsd/src/))

- Evidence: [Thay đổi graph M1](experiments/mlsd/results/model_M1.json) · [Kiểm tra rounding](experiments/mlsd/results/quantize_probe.json) · [Chất lượng 8 luồng](experiments/mlsd/results/configuration_validation.json) · [Hồi quy 54 ảnh](experiments/mlsd/results/regression.json) · [So sánh tốc độ](experiments/mlsd/results/speed_selection.json) · [Phân rã I/O](experiments/mlsd/results/io_profile.json).

### 1. Đưa quantize input vào model

Ở lần đo trước, quantize input bằng NumPy tốn khoảng 1,6 ms. Em thử hai phương án trên C1, giữ nguyên weight, scale/zero-point và cách chọn các lớp INT8/FP32:

- **M1:** nhận input float NCHW và thêm QUANTIZE trước tensor INT8 cũ.
- **M2:** từ M1, thêm TRANSPOSE để model trả trực tiếp output float NCHW.

Khi sửa graph, em phát hiện C1 lưu 44 buffer bên ngoài phần FlatBuffer chính. Đóng gói lại trực tiếp sẽ thiếu dữ liệu weight. Em đã chuyển đầy đủ các buffer này vào model và kiểm từng buffer khớp bản gốc. Em thêm **C1 đóng gói lại** làm đối chứng riêng để tách ảnh hưởng của đóng gói khỏi việc sửa I/O; output của đối chứng khớp hoàn toàn C1 trên 16 ảnh development. [Kết quả đối chứng](experiments/mlsd/results/roundtrip_control.json).

### 2. Chất lượng

Native QUANTIZE khớp input INT8 của adapter cũ trên đủ **32 ảnh calibration và 16 ảnh development**, tương ứng 50.331.648 giá trị ở mỗi chế độ kiểm. Với 792 giá trị quanh điểm làm tròn, clipping và zero, có 50 giá trị lệch 1 LSB; tất cả nằm trong các trường hợp làm tròn đã kiểm, không có sai khác lớn hơn. Vì vậy em tiếp tục kiểm output và sAP của model đầy đủ, không suy tính tương đương từ probe riêng.

| Bản | sAP10 development 16 ảnh | sAP10 hồi quy 54 ảnh |
|---|---:|---:|
| FP32 tham chiếu | 15,3230% | 13,2033% |
| C1 | 14,8765% | 13,1121% |
| **M1** | **14,8765%** | **13,1121%** |
| M2 | 14,8765% | 13,1121% |

M1/M2 cho output khớp hoàn toàn C1 trên cả hai tập. Em kiểm thêm đủ 16 ảnh ở 8 luồng: output của cả năm bản đều khớp kết quả 4 luồng. sAP10 development của M1/M2 giảm **0,4466 điểm** so FP32, đạt ngưỡng giảm tối đa 0,5 điểm đã đặt trước. Bản M1 có dung lượng **1.905.168 bytes**, gần như C1 **1.905.424 bytes**.

FP32/C1 trong bảng dùng kết quả tham chiếu đã có; hôm nay em chạy M1/M2 và đối chiếu tensor với C1. Tập 54 ảnh đã dùng trước đây, dùng để kiểm hồi quy sau khi chọn bản theo development và tốc độ. sAP10 ở đây là điểm trên tập con theo cùng decoder, không phải điểm toàn bộ benchmark trong paper.

### 3. Tốc độ

Em đo trên CPU i7-12700H, Linux Docker/WSL2, LiteRT 2.2.0/XNNPACK. Mỗi cấu hình 4/8 luồng có hai phiên process riêng; mỗi phiên đổi thứ tự qua 5 vòng, mỗi vòng 16 warm-up và 16 lần đo, tổng 80 mẫu/model. Các bản cùng nhận float NCHW và trả float NCHW liên tục trong bộ nhớ; thời gian gồm adapter và copy, không tính load model, tiền xử lý ảnh hoặc decoder đoạn thẳng.

| Bản | 4 luồng S1 / S2 (ms) | 8 luồng S1 / S2 (ms) | p95 8 luồng S1 / S2 (ms) |
|---|---:|---:|---:|
| FP32 | 26,484 / 26,979 | 20,683 / 20,803 | 24,142 / 26,409 |
| C1 | 26,393 / 26,118 | 20,469 / 20,247 | 22,540 / 21,106 |
| C1 đóng gói lại | 26,250 / 26,265 | 20,329 / 20,269 | 22,532 / 20,994 |
| **M1** | **24,348 / 24,877** | **18,945 / 19,114** | **20,602 / 20,915** |
| M2 | 24,266 / 24,155 | 19,223 / 18,839 | 20,709 / 19,764 |

Ở 8 luồng, M1 giảm median **8,40% và 8,12% so FP32**, **7,44% và 5,60% so C1**. So với đối chứng chỉ đóng gói lại, M1 vẫn giảm **6,81% và 5,70%**, cho thấy lợi ích không chỉ đến từ đóng gói buffer.

Khoảng 95% ước lượng theo vòng đo của mức giảm so FP32 là **[5,39%; 11,45%]** và **[4,63%; 10,35%]**; so C1 là **[5,13%; 10,42%]** và **[3,16%; 7,39%]**. Các khoảng đều trên 0 trong hai phiên. Đây là ước lượng từ 5 vòng trên cùng máy/ngày, chưa chứng minh mức tăng tốc trên thiết bị khác.

M2 chậm hơn M1 **1,47%** ở phiên đầu và nhanh hơn **1,44%** ở phiên sau. Em chọn **M1**, chưa thêm TRANSPOSE output vì lợi ích bổ sung chưa nhất quán.

### 4. Chi phí I/O còn lại

Em đo riêng 24 mẫu trên tám ảnh sau warm-up để phân rã chi phí. Đường gọi này dùng set_tensor/invoke/get_tensor, khác benchmark SignatureRunner ở trên; chỉ dùng để chẩn đoán. Output được kiểm khớp đường gọi thông thường trên một ảnh cho mỗi bản/cấu hình luồng.

| Bản, 8 luồng | App quantize (ms) | Copy input (ms) | Native invoke (ms) | Output adapter (ms) | Tổng staged (ms) |
|---|---:|---:|---:|---:|---:|
| C1 | 1,620 | 0,104 | 18,517 | 0,243 | 20,510 |
| C1 đóng gói lại | 1,615 | 0,103 | 17,891 | 0,247 | 20,165 |
| M1 | 0,005 | 0,413 | 18,016 | 0,257 | 18,856 |
| M2 | 0,004 | 0,414 | 18,227 | 0,023 | 18,841 |

Với M1, công việc quantize chuyển vào native invoke, còn thời gian ở adapter input gần như hết. Đổi lại, copy input float tốn hơn input INT8. M2 giảm output adapter nhưng native invoke tăng trong phép đo này, phù hợp với việc chưa thấy lợi ích bổ sung nhất quán. Tổng gồm cả copy output; các cột là median riêng nên không cộng trực tiếp để suy tổng.

Em đã kiểm điều kiện dùng native profiler: image hiện tại chưa có compiler/CMake và Interpreter không có public API trả thời gian từng kernel. Hôm nay em chưa build runtime profiler mới. [Môi trường M-LSD](environments/mlsd.json) · [Phân tích hướng profiler](RESEARCH.md).

**Kết luận:** đưa quantize input vào model giúp C1 có lợi thế tốc độ rõ hơn trong cấu hình đã đo, đồng thời giữ nguyên kết quả trên các ảnh kiểm tra. Phép đo I/O vẫn chưa tách được thời gian từng kernel LiteRT; em chưa kết luận Q/DQ nào bên trong chạy chậm. Bước tiếp theo là kiểm M1 trên thiết bị đích và chuẩn bị native profiler trước khi chọn thêm lớp giữ float.

## ZipFormer ([source model](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17))

- Evidence: [Wrapper](experiments/zipformer/src/runtime22.py) · [Đối chiếu fixture](experiments/zipformer/results/fixture_parity.json) · [WER 173 câu](experiments/zipformer/results/regression.json) · [Stream 10 phút](experiments/zipformer/results/continuous_state.json) · [Benchmark phiên 1](experiments/zipformer/results/benchmark_s1.json) / [phiên 2](experiments/zipformer/results/benchmark_s2.json) · [Kết quả lựa chọn](experiments/zipformer/results/selection.json).

### 1. Thử giảm chi phí gọi runtime

Em giữ encoder FullyConnected đã chọn, decoder và joiner; không đổi graph hoặc weight. Bản **Z0** dùng SignatureRunner đã cache metadata. Bản **Z1** dùng trực tiếp set_tensor/invoke/get_tensor với shape cố định để bỏ bước resize/allocate lặp lại của SignatureRunner. Cách chuyển dtype counter, khởi tạo state, frontend và greedy decoding được giữ nguyên.

Z1 khớp hoàn toàn Z0 trên **15 bộ input/state encoder, mỗi bộ 36 output**, 18 input decoder và 30 input joiner. Token và final counter giữ nguyên trên cả **173 câu**:

| Tập hồi quy | S / D / I | Số từ tham chiếu | WER |
|---|---:|---:|---:|
| 73 câu đã có | 75 / 13 / 3 | 1.150 | 7,9130% |
| 100 câu, 20 người đọc | 101 / 9 / 7 | 2.295 | 5,0980% |

Em kiểm thêm **600 giây, 1.875 chunk liên tục** bằng audio lặp từ một file thử, mỗi wrapper tự giữ state xuyên suốt. Encoder output, token và final counter khớp hoàn toàn. Bài này kiểm đường chạy liên tục, không phải tập WER mới; chưa đối chiếu toàn bộ cached-state tensor tại từng chunk hoặc đánh giá đa người nói trong stream dài.

### 2. Tốc độ và lựa chọn

Em đo LiteRT 2.1.2, CPU 4 luồng, hai process session, mỗi phiên 5 vòng đổi thứ tự. Mỗi component có 160 mẫu; trước mỗi vòng pipeline, cả encoder/decoder/joiner và một lượt decode hoàn chỉnh đều được warm-up. RTF gồm đọc audio, frontend và greedy decoding trên cùng ba audio thử, không tính load model.

| Wrapper | Encoder median S1 / S2 (ms) | RTF median S1 / S2 |
|---|---:|---:|
| **Z0: SignatureRunner** | **6,339 / 6,051** | **0,03001 / 0,02957** |
| Z1: gọi trực tiếp | 6,455 / 6,060 | 0,02962 / 0,02897 |

Joiner giảm median từ **0,0247 xuống 0,0159 ms** ở phiên đầu và **0,0285 xuống 0,0153 ms** ở phiên sau, nhưng encoder vẫn chiếm phần lớn thời gian. RTF toàn pipeline chỉ giảm **1,30% và 2,05%**; khoảng 95% theo vòng đo là **[-10,27%; 10,03%]** và **[-10,35%; 12,29%]**, đều chứa 0.

**Kết luận:** em giữ **Z0**, chưa thay wrapper chỉ dựa trên chênh lệch nhỏ này. Cách warm-up hôm nay khác lần trước nên chỉ so hai bản cùng phiên hôm nay. Tiếp theo em sẽ ưu tiên đo encoder và pipeline trên thiết bị đích thay vì tiếp tục tối ưu phần gọi Python.
