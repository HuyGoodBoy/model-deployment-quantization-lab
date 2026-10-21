# Báo cáo ngày 21/10/2026

Người thực hiện: Bùi Gia Huy.

Hôm nay em tiếp tục tối ưu encoder ZipFormer và tìm cách cải thiện tốc độ M-LSD sau lượng tử hóa. Với ZipFormer, bản FullyConnected cải thiện rõ tốc độ, còn fuse thêm bias chưa giúp toàn pipeline nhanh hơn. Với M-LSD, bản bỏ các kênh đầu ra không dùng và giữ head FP32 giữ được sAP gần FP32, nhưng mức tăng tốc ở 8 luồng còn nhỏ.

## ZipFormer ([source model](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17))

- Evidence: [Tối ưu encoder](experiments/zipformer/results/encoder_fc_rewrite.json) · [Fuse bias](experiments/zipformer/results/refinement_bias_rewrite.json) · [WER và streaming state](experiments/zipformer/results/refinement_bias_evaluation.json) · [Benchmark](experiments/zipformer/results/refinement_benchmark.json) · [Khởi tạo và RAM](experiments/zipformer/results/refinement_coldstart_memory.json).

### 1. Tối ưu graph encoder

Em thay 112 BatchMatMul có weight cố định bằng FullyConnected, giữ nguyên 40 BatchMatMul của attention động. Weight được transpose từ [K,N] sang [N,K]; input/output, 35 state, decoder và joiner không đổi. Phần này vẫn chạy FP32.

Sau đó em thử gộp bias vào FullyConnected để bỏ phép ADD phía sau. Trong 112 lớp, có 92 trường hợp đáp ứng điều kiện về shape, activation và số nhánh sử dụng output. Sau khi gộp, số ADD trong graph giảm từ 241 xuống 149.

| Encoder | BatchMatMul | FullyConnected | Dung lượng |
|---|---:|---:|---:|
| Baseline ngày 20 | 152 | 0 | 78.638.936 bytes |
| FC | 40 | 112 | 78.639.872 bytes |
| FC + bias | 40 | 112 | 78.635.088 bytes |

Bản FC tăng nhẹ 936 bytes do cấu trúc lưu graph; tổng dữ liệu trong buffer không tăng. Cả hai bản tối ưu đều qua kiểm tra 36 output trên 15 bộ input/state thực với rtol/atol 1e-4; counter nguyên khớp hoàn toàn. Việc gộp bias có thể thay đổi làm tròn FP32 nên em kiểm thêm transcript và streaming state.

### 2. Tốc độ và chất lượng

Em đo trên CPU i7-12700H, Linux Docker/WSL2, LiteRT 2.1.2, 4 luồng, batch 1 và chunk 39. Ba bản chạy lần lượt, đổi thứ tự qua 5 vòng; mỗi vòng warm-up 16 lần rồi đo 32 lần, tổng 160 mẫu/bản.

| Encoder | Median / p95 (ms) | RTF toàn pipeline |
|---|---:|---:|
| Baseline | 8,637 / 9,670 | 0,03913 |
| FC | **6,146 / 6,877** | **0,02952** |
| FC + bias | 5,990 / 6,568 | 0,02975 |

Bản FC giảm **28,8%** median encoder và **24,6%** thời gian toàn pipeline so baseline. Fuse thêm bias giảm median encoder khoảng **2,54%**, nhưng kết quả còn dao động và RTF chưa cải thiện. Em giữ bản FC, chưa chọn bản fuse bias.

RTF đo trên cùng ba audio thử, lấy median 3 vòng, gồm frontend và greedy decoding, không tính load model. Các bản dùng cùng decoder, joiner và context khởi tạo [0,0]. Encoder được warm-up trước; lần gọi đầu decoder/joiner vẫn được tính ở cả ba bản.

| Tập regression | Người đọc | Từ tham chiếu | S / D / I | WER FC | WER FC + bias |
|---|---:|---:|---:|---:|---:|
| 73 câu đã có | 1 | 1.150 | 75 / 13 / 3 | 7,9130% | 7,9130% |
| 100 câu test-clean subset | 20 | 2.295 | 101 / 9 / 7 | 5,0980% | 5,0980% |

Token và final counter khớp trên đủ 173 câu. Em cũng kiểm stream liên tục **79,5 giây, 249 chunk**: không reset state giữa các đoạn lặp, token/counter vẫn khớp và encoder output nằm trong sai số cho phép.

Tập 100 câu gồm 20 người đọc, khoảng 14,2 phút, lấy từ LibriSpeech test-clean và không trùng ID hoặc người đọc của tập 73 câu. Đây là đánh giá trên một phần dữ liệu, chưa phải WER toàn test-clean. Hai tập được dùng lại để kiểm bản fuse bias. Bài kiểm stream hiện tại cũng chưa đánh giá session rất dài và giới hạn counter INT32.

### 3. Thời gian khởi tạo và RAM

Em đo thêm 5 lần cho mỗi encoder, mỗi lần khởi chạy trong một process riêng để tránh ảnh hưởng bộ nhớ của model trước.

| Encoder | Init median (ms) | First invoke median (ms) | Peak RSS median / max (MiB) |
|---|---:|---:|---:|
| Baseline | 1.092,6 | 21,61 | 204,93 / 205,09 |
| FC | 1.077,3 | 17,85 | 204,88 / 205,07 |
| FC + bias | 1.136,9 | 18,87 | 204,75 / 204,95 |

Peak RSS khoảng **205 MiB** ở cả ba bản, chưa thấy tăng đáng kể trong phép đo này. Số liệu gồm cả runtime và process Python, chưa phải mức RAM riêng của model hoặc kết quả trên Android. Thời gian khởi tạo còn dao động; cache file của hệ điều hành được giữ nguyên.

**Kết luận:** bản FC cải thiện tốc độ và giữ nguyên kết quả nhận dạng trên các tập đã kiểm tra. Em giữ bản này để kiểm tiếp trên thiết bị đích; chưa chọn fuse bias vì chưa thấy lợi ích toàn pipeline.

## M-LSD ([mã nguồn](experiments/mlsd/src/))

- Evidence: [Sai số activation](experiments/mlsd/results/activation_diagnosis.json) · [Kiểm tra prune FP32](experiments/mlsd/results/refinement_prune_parity.json) · [C1](experiments/mlsd/results/refinement_C1.json) / [C2](experiments/mlsd/results/refinement_C2.json) / [C3](experiments/mlsd/results/refinement_C3.json) · [Đánh giá 54 ảnh](experiments/mlsd/results/refinement_regression.json) · [Phân rã I/O](experiments/mlsd/results/refinement_io_profile.json).

### 1. Phân tích sai số và các phương án thử

Em giữ nguyên checkpoint, 32 ảnh Wireframe calibration, 16 ảnh YorkUrban development và decoder. Khi so activation trên bốn ảnh, relative L2 tại ba lớp đầu tăng từ **0,72% → 10,53% → 35,87%**. Sai số đã tích lũy từ backbone nên chưa thể kết luận chỉ một lớp gây mất chất lượng.

Conv cuối có 16 kênh nhưng decoder chỉ dùng kênh 7 cho center và kênh 8–11 cho displacement. Hai nhóm có biên độ khác nhau; khi dùng chung scale INT8, độ phân giải của center bị giảm. Em đã thử tách hai nhánh để dùng scale riêng: sAP10 development tăng từ **2,85% lên 12,00%**, vẫn thấp hơn FP32 **15,32%**.

Tiếp đó em bỏ 11 kênh không dùng, giữ một Conv 5 kênh và thử các mức giữ float khác nhau. Bản prune FP32 cho output khớp hoàn toàn với FP32 gốc trên 16 ảnh trước khi lượng tử hóa.

### 2. Chất lượng và dung lượng

| Bản | Conv INT8 / FP32 | Dung lượng (bytes) | sAP10 dev16 | sAP10 hồi quy54 |
|---|---:|---:|---:|---:|
| FP32 | 0 / 41 | 2.483.636 | 15,3230% | 13,2033% |
| AEQ static cũ | 41 / 0 | 792.192 | 2,8522% | — |
| A: toàn head float, output cũ | 30 / 11 | 1.910.560 | 14,8765% | 13,1121% |
| B: hai nhánh INT8 | 42 / 0 | 789.072 | 12,0048% | 10,8835% |
| **C1: prune 5 kênh, toàn head float** | **30 / 11** | **1.905.424** | **14,8765%** | **13,1121%** |
| C2: prune, chỉ Conv cuối/output float | 40 / 1 | 789.152 | 14,5570% | — |
| C3: prune, suffix block16 float | 38 / 3 | 1.007.072 | 14,7760% | — |

Em đặt ngưỡng trước khi thử: sAP10 development giảm không quá **0,5 điểm** so FP32. C1 giảm **0,4466 điểm**, đạt ngưỡng; C2 giảm **0,7660 điểm**, C3 giảm **0,5470 điểm**, chưa đạt. Vì vậy em chỉ benchmark đầy đủ và chạy thêm 54 ảnh cho C1.

Dấu — là các bản không chạy lại trên 54 ảnh: static giữ kết quả development cũ; C2/C3 dừng sau bước sàng lọc chất lượng.

C1 đạt **13,1121%** trên 54 ảnh, so FP32 **13,2033%**, thấp hơn **0,0912 điểm** và dung lượng giảm **23,3%**. Tập 54 ảnh đã dùng ở các lần trước, được dùng để kiểm hồi quy. Các điểm sAP trên đây theo cùng decoder và cách đánh giá trên tập con, không so trực tiếp với điểm toàn bộ benchmark của paper.

Các bản được lượng tử hóa lại bằng AI Edge Quantizer. Kết quả sAP gần FP32 chưa có nghĩa tensor khớp hoặc đã xử lý xong sai số của bản QAT/QDQ trước đó.

### 3. Tốc độ ở 4 và 8 luồng

Em đo bằng LiteRT 2.2.0/XNNPACK trên cùng máy Linux, hai phiên riêng cho mỗi cấu hình luồng. Mỗi phiên có 5 vòng, mỗi vòng 16 warm-up và 16 lần đo, tổng 80 mẫu/model.

Cả ba bản trả cùng output float NCHW liên tục trong bộ nhớ. Thời gian đo gồm quantize/dequantize input/output và chuyển layout, không tính load model, tiền xử lý ảnh hoặc decoder đoạn thẳng.

| Bản | 4 luồng S1 (ms) | 4 luồng S2 (ms) | 8 luồng S1 (ms) | 8 luồng S2 (ms) |
|---|---:|---:|---:|---:|
| FP32 | 27,203 | 27,372 | 21,189 | 21,166 |
| A | 27,044 | 26,178 | 21,284 | 21,403 |
| C1 | **26,408** | **26,058** | **20,982** | **20,814** |

Kết quả chi tiết: [4 luồng — phiên 1](experiments/mlsd/results/refinement_contiguous_benchmark_t4_s1.json), [phiên 2](experiments/mlsd/results/refinement_contiguous_benchmark_t4_s2.json); [8 luồng — phiên 1](experiments/mlsd/results/refinement_contiguous_benchmark_t8_s1.json), [phiên 2](experiments/mlsd/results/refinement_contiguous_benchmark_t8_s2.json).

Ở 4 luồng, C1 giảm median **2,92% và 4,80%** so FP32. Ở 8 luồng, mức giảm còn **0,98% và 1,66%**. Khoảng 95% ước lượng bằng bootstrap theo vòng đo ở 8 luồng vẫn chứa 0, nên chưa đủ để khẳng định mức tăng tốc này ổn định. [Phân tích độ dao động](REFINEMENT_ANALYSIS.json).

Em giữ C1 để đánh giá tiếp vì giảm dung lượng và giữ sAP gần FP32; chưa chọn thay FP32 8 luồng mặc định chỉ với chênh lệch tốc độ hiện tại.

### 4. Chi phí I/O và hướng tối ưu tiếp

Em tách bước quantize input, sao chép dữ liệu, invoke và xử lý output để xem phần nào làm mất lợi ích tăng tốc. Phép đo này dùng tám ảnh development, 24 mẫu/bản sau warm-up; kiểm output khớp với đường chạy thông thường trên một input cho mỗi model/cấu hình luồng.

| Bản, 8 luồng | App quantize (ms) | Native invoke (ms) | Output adapter (ms) | Tổng staged (ms) |
|---|---:|---:|---:|---:|
| FP32 | 0,004 | 20,041 | 0,023 | 20,735 |
| A | 1,619 | 19,063 | 0,024 | 21,060 |
| C1 | 1,610 | 18,072 | 0,255 | 20,167 |

Các cột là median riêng; tổng được tính trên từng lần chạy, không cộng median từng bước. Đây là phép đo phân rã chi phí, dùng đường gọi khác benchmark ở trên.

Native invoke của A nhanh hơn FP32, nhưng quantize input bằng NumPy tốn khoảng **1,6 ms**, làm mất phần lớn lợi ích. C1 còn tốn khoảng **0,26 ms** cho xử lý output, bao gồm chuyển NHWC về NCHW liên tục trong bộ nhớ.

Cả ba bản đều có một node DELEGATE trong thông tin graph sau allocate; hiện chưa có bằng chứng cho giả thuyết bị chia thành nhiều partition XNNPACK. Phần đo này chưa tách được thời gian từng kernel hoặc các Q/DQ bên trong LiteRT. [Profile ONNX QDQ ngày 20](../2026-10-20/experiments/mlsd/results/ort_profile_comparison.json) dùng runtime khác nên không lấy để quy nguyên nhân cho từng op LiteRT.

**Kết luận:** C1 là phương án giảm dung lượng và giữ chất lượng đáng kiểm tiếp, nhưng chưa có lợi thế tốc độ rõ ở 8 luồng. Tiếp theo em sẽ thử giảm chi phí quantize input ở phía native và đo từng kernel LiteRT trước khi chọn thêm lớp giữ float. Chưa đánh giá trên Android.
