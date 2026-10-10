# Báo cáo thực nghiệm ngày 09/10/2026

**Người thực hiện:** Huy. **Máy:** Windows x64, i7-12700H, RAM 16 GB; Linux Docker/WSL2 cho reproduce bổ sung. **Phạm vi:** CPU; chưa chạy M-LSD hoặc post-processing trên Android. Code và tài liệu thuật toán có hỗ trợ AI, Huy cần tự đọc lại trước khi trình bày.

## 1. Mục tiêu và kết quả bàn giao

- Đọc repo Đức tại revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`, giữ nguyên repo; chạy **model của Huy** trên runtime cũ và runtime mới tương ứng phiên bản Đức khai báo.
- Đối chứng DistilBERT: mixed static INT8, strict INT8 và hai giá trị attention-mask sentinel `1e4`, `1e2`.
- Chuyển **trọng số PyTorch M-LSD Tiny 512** sang Keras tương đương rồi TFLite; thử FP16 weights, dynamic range, static INT8 và graph chứa decoder.
- Prototype hình học Java có 5 ca kiểm tra; đề xuất contract Android và những phần cần port tiếp.

**Reproduce bổ sung:** Đã chạy source ResNet18 Đức trong Linux Docker/WSL2, đúng 10 release dependencies khai báo; 6/6 biến thể có output/latency. q1/q2/q3 nguyên script, q4 giữ nguyên hàm và tách từng variant để lưu lỗi/raw timing; source mount read-only và kiểm SHA256. Không cần cùng Mac để reproduce: phần cứng/OS/backend phải ghi riêng khi so tốc độ. [Bảng Mac Đức báo cáo ↔ Linux Huy thực đo và model Huy trên Linux](PEER-REPRODUCTION.md).

Matrix 24 phép đo ghép thành 12 cặp ở phần 3 vẫn là đối chứng model Huy trên hai runtime Windows đã đo trước đó. Bảng ResNet18 và model Huy trên Linux được tách riêng, không trộn protocol 10/100 của Đức với 30/200 của Huy.

## 2. Đối chiếu môi trường và giao thức

| Nội dung | Đức theo source/báo cáo | Huy trong phiên hôm nay |
| --- | --- | --- |
| Model ảnh | ResNet18 torchvision, logits | ResNet50 Keras, softmax |
| Model text | Không có trong bài đối chiếu | DistilBERT SST-2 |
| TFLite quantizer | LiteRT Torch → AI Edge Quantizer recipes | TensorFlow converter 2.15.1 |
| Runtime | ORT 1.30.0, LiteRT 2.2.0 | Cũ: ORT 1.20.1/TF 2.15.1; mới: ORT 1.30.0/LiteRT 2.2.0 |
| Phần cứng | Mac M1 Pro theo báo cáo | i7-12700H Windows |
| Calibration/evaluation | 100/300 ảnh | Model Huy giữ nguyên 100/100; M-LSD 32/10 ảnh |
| Thread/warm-up/runs | 4 / 10 / 100 | 4 / 30 / 200; ORT inter-op=1 |

Mỗi biến thể chạy process riêng, tuần tự; batch 1. Timer gồm API đồng bộ, copy tensor và quantize/dequantize I/O; loại load model, file và preprocessing. Benchmark lặp input đầu tiên; chưa đo phân phối nhiều loại input, nhiệt độ hay công suất. JSON lưu 200 mẫu latency, hash model/input/output. TFLite dùng delegate mặc định của Interpreter; log ghi XNNPACK nếu được tạo, chưa cố định mọi partition của delegate.

Nguồn Đức: [prepare](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q1_prepare.py), [quantizer](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q3_quant_tflite.py), [benchmark](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q4_evaluate.py). Claim range/ties trong [báo cáo Đức](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/BAO_CAO.md) chưa được xác minh từ raw output vì repo không chứa artifact đó.

## 3. Cùng model Huy, thay runtime

| Model / variant | ORT/LiteRT cũ ms | ORT/LiteRT mới ms | Mới/cũ | Accuracy cũ→mới | SNR cũ→mới dB |
| --- | --- | --- | --- | --- | --- |
| cv/onnx_dynamic | 30.608 | 25.465 | 0.832 | 73% → 73% | 40.339 → 40.339 |
| cv/onnx_fp32 | 32.768 | 27.234 | 0.831 | 73% → 73% | 112.398 → 112.398 |
| cv/onnx_static | 29.831 | 32.544 | 1.091 | 71% → 71% | 19.115 → 19.115 |
| cv/tflite_dynamic | 163.850 | 15.669 | 0.096 | 73% → 73% | 27.328 → 26.576 |
| cv/tflite_fp32 | 57.345 | 93.937 | 1.638 | 73% → 73% | 113.826 → 114.099 |
| cv/tflite_static | 24.907 | 14.906 | 0.598 | 74% → 74% | 20.585 → 20.585 |
| text/onnx_dynamic | 11.658 | 11.753 | 1.008 | 90% → 91% | 24.310 → 25.011 |
| text/onnx_fp32 | 24.567 | 21.528 | 0.876 | 91% → 91% | 124.677 → 128.327 |
| text/onnx_static | 95.143 | 94.119 | 0.989 | 92% → 92% | 10.142 → 10.142 |
| text/tflite_dynamic | 1054.815 | 39.582 | 0.038 | 90% → 90% | 21.119 → 21.270 |
| text/tflite_fp32 | 163.676 | 36.111 | 0.221 | 91% → 91% | 122.986 → 124.530 |
| text/tflite_static | 114.788 | 86.863 | 0.757 | 52% → 48% | 0.036 → 0.030 |

Tỷ lệ mới/cũ <1 nghĩa là nhanh hơn. Đây là thay **bộ software/runtime**: hai môi trường còn khác NumPy và dependency native, vì vậy chưa quy toàn bộ thay đổi cho một kernel hay một phiên bản ORT. Không đem trực tiếp latency ResNet18 Mac so với ResNet50 Windows để kết luận framework nhanh hơn. Cùng 4 thread cũng khác kiến trúc ARM/x86, SIMD, bộ nhớ, scheduling, fusion và khả năng delegate xử lý operator.

Một khác biệt phần mềm có cơ sở: TensorFlow công bố XNNPACK hỗ trợ dynamic-range Fully Connected/Conv2D và bật mặc định trong prebuilt binaries từ TF 2.17; baseline TF 2.15 nằm trước thay đổi này. Đây là **giả thuyết góp phần** giải thích dynamic chênh lệch, chưa phải kết quả profiling xác định từng kernel trong hai wheel đang dùng. [Thông báo TensorFlow](https://blog.tensorflow.org/2024/04/faster-dynamically-quantized-inference-with-xnnpack.html). FP32 chậm hơn ở runtime mới vẫn cần profile/thread-power đối chứng riêng.

Audit ResNet50 dynamic sau phép đo: old: 19 DELEGATE nodes; new: 1 DELEGATE nodes. [Audit cũ](experiments/01-reproduce/results/run-01/delegate_audit_old.json), [audit mới](experiments/01-reproduce/results/run-01/delegate_audit_new.json) lưu ranh giới tensor. Introspection dùng API private, có cả original nodes và delegate nodes, chưa xác nhận coverage hoặc timing từng kernel.

**Đối chứng do latency biến động:** DistilBERT static cùng file có số đo khác giữa nhóm ablation và runtime matrix. Chạy tiếp hai process liền nhau với cùng script/model/input, chỉ đổi cách đặt TF global threads; Interpreter vẫn 4 thread trong cả hai:

| TF global thread setting | Median ms | Accuracy | SNR dB |
| --- | --- | --- | --- |
| environment only | 123.401 | 52% | 0.036 |
| set API | 128.310 | 52% | 0.036 |

Source/JSON: [run-02-threadcheck](experiments/01-reproduce/results/run-02-threadcheck/). Matrix đặt TF thread qua environment; ablation đặt thêm API. Khác biệt protocol này và trạng thái máy là giới hạn khi so hai nhóm; không coi mọi thay đổi latency là do phiên bản thư viện. Chưa profile nhiệt độ/power hoặc lặp nhiều session.

Cặp kiểm tra liền nhau không tái hiện lợi thế latency 43 ms của nhóm ablation; đổi sang API đặt thread không khắc phục chênh lệch. Vì vậy chưa xác định nguyên nhân biến động giữa các phase. Các bảng latency là snapshot của phiên đo, cần kiểm soát power/nhiệt độ và lặp lại để đưa ra kết luận hiệu năng chắc hơn; không so chéo phase như cùng trạng thái máy.

Accuracy được tính trên 100 mẫu có nhãn thật. Sai số output không đồng nghĩa với đổi nhãn; JSON từng biến thể còn lưu agreement, MAE, max error, relative L2, cosine và SNR so với TensorFlow gốc. Các tỷ lệ này chỉ mô tả tập nhỏ đã chọn.

Ứng viên nhanh nhất trong các row runtime mới có accuracy không thấp hơn reference trên tập này: cv: `tflite_static` mới, 14.906 ms, accuracy 74%; ứng viên để đánh giá thêm, chưa kết luận deployment. text: `onnx_dynamic` mới, 11.753 ms, accuracy 91%; ứng viên để đánh giá thêm, chưa kết luận deployment. Không dùng chênh lệch 1–2 mẫu để kết luận accuracy cải thiện có ý nghĩa thống kê.

## 4. Dynamic/static ONNX Runtime và TFLite/LiteRT

`x ≈ scale × (q − zero_point)`. Dynamic tính range activation được hỗ trợ trong lúc inference; static dùng range từ representative/calibration data rồi cố định khi chạy. Cả hai đều có thể giữ operator float. [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

TFLite dynamic range thường lưu weights INT8, dùng hybrid kernels ở operator hỗ trợ; activation hoặc phần graph khác có thể vẫn float. Static calibration có thể cho graph mixed hoặc ép integer-only, tùy cấu hình converter. Token IDs/index/bias INT32 không phải float fallback. [LiteRT PTQ](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization).

Cùng tên phương pháp vẫn khác: operator được chọn, MinMax/histogram, per-tensor/per-channel, signedness, fusion, biểu diễn QDQ/QOperator và kernel/provider/delegate. Bài Huy ONNX text chọn MatMul/Gemm; TFLite có phạm vi khác. Đức quantize file TFLite bằng AI Edge Quantizer, không phải cùng TensorFlow converter của Huy. Phải audit graph thay vì suy từ nhãn “INT8”. Calibration không đại diện evaluation có thể ảnh hưởng **cả ONNX và TFLite**; chưa có chứng cứ ONNX miễn nhiễm.

Ở ONNX QDQ text của Huy, mask/Sub/Softmax nằm ngoài danh sách operator quantize; audit vẫn có 6 Softmax và nhiều operator float. TFLite static phủ rộng hơn, kể cả vùng mask với scale rất lớn. Vì vậy cùng “static” chưa phải hai graph có cùng rủi ro số học. QDQ giữ tên MatMul float trong graph lưu trữ, runtime có thể fuse kernel INT8; chỉ đếm tên node chưa đủ để kết luận dtype tính toán. [Metadata ONNX](../../quantization/results/text/conversion_onnx_static.json).

## 5. Phần riêng Huy: DistilBERT calibrated mixed INT8

**Calibrated mixed INT8 là static quantization cho phép graph mixed**, không phải phương pháp thứ ba nằm giữa dynamic và static. Static/dynamic nói lúc lấy scale; mixed/full nói phạm vi dtype/operator.

Cấu hình ngày trước dùng representative dataset, cho phép `TFLITE_BUILTINS_INT8` + `TFLITE_BUILTINS`, output INT8, IDs/mask INT32. Tensor float còn lại xác định là `distilbert/Cast [1,64]`; không có bằng chứng đã chủ động giữ LayerNorm hoặc toàn attention FP32.

Bản strict convert thành công; audit {'float32': 1, 'int8': 372, 'int32': 126}. Float còn ở CAST→QUANTIZE của mask, không phải toàn attention/LayerNorm FP32. Accuracy 52%, chưa chứng minh strict tốt hơn mixed. Vì vậy lý do chọn mixed trước đây là cho phép fallback để converter dễ chạy, **chưa phải lý do thực nghiệm chứng minh mixed cần thiết hoặc giữ chất lượng tốt hơn**. Cho phép mixed cũng không tự bảo vệ phần nhạy cảm khỏi bị quantize.

| Biến thể | Accuracy | F1 | SNR dB | MAE | Max error | Median ms | Scale lớn nhất |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 52% | 0.667 | 0.036 | 3.46346 | 4.84640 | 42.871 | 3.92e+27 |
| mixed_mask1e2 | 67% | 0.748 | 0.835 | 3.12495 | 4.92559 | 64.266 | 2.67e+03 |
| mixed_mask1e4 | 56% | 0.694 | 0.013 | 3.47366 | 5.04505 | 42.415 | 2.67e+03 |
| strict | 52% | 0.667 | 0.036 | 3.46346 | 4.84640 | 43.276 | 3.92e+27 |

Hướng thử mới: chỉ thay sentinel attention mask `1e30` bằng `1e4` hoặc `1e2`, giữ weights/input/calibration. Trên 100 evaluation, hai bản FP32 đã so với reference và cho output giống hệt (max error=0, accuracy 91%). Đây là kiểm tra trên tập đã dùng, chưa bảo đảm cho mọi sequence/input.

Mask 1e30 → 1e2: accuracy 52% → 67%, SNR 0.036 → 0.835 dB. Bản mask 1e2 vẫn thấp hơn FP32 91%, chưa đạt chất lượng để khuyến nghị deployment. Audit scale đi cùng output giúp đánh giá giả thuyết sentinel quá lớn làm resolution INT8 quanh attention quá thô. Các giá trị ablation được chọn trước phép đo; chưa tune trên tập test độc lập. Không gọi mọi chênh lệch là lỗi calibration của thư viện, hoặc kết luận đây là nguyên nhân duy nhất.

Evidence/code: [run.py](experiments/02-distilbert-int8/src/run.py), [kết quả](experiments/02-distilbert-int8/results/run-01/).

## 6. M-LSD Tiny: chuyển đổi và quantization

PyTorch revision `2312205254e66911703decf775f626995d260f17`; NAVER revision `453cafa09467d0272760578d35c1fda38e8895a5`. [Source/hash lock](experiments/03-mlsd/sources.lock.json). Không huấn luyện lại. Copy OIHW→HWIO, depthwise, BN eps, pad phải/dưới cho stride 2 và bilinear `align_corners=True`.

Input raw RGBA 512×512, alpha=1; model bridge normalize `x/127.5−1`. PyTorch nhận NCHW đã normalize. Reference raw là NHWC `[N,256,256,9]`. Bản TFLite chính thức trả `points [1,200,2]`, `scores [1,200]`, `vmap [1,256,256,4]`.

**Kiểm tra bridge:** SNR 96.931 dB, relative L2 0.00001424, max raw error 0.029602. `allclose(atol=rtol=1e-4)` ban đầu không đạt. Sau đọc mapping/architecture và đối chứng kernel, dùng gate công khai relative L2<1e-4 và max error<0,1 map unit; với displacement tương ứng <0,2 pixel ảnh 512. Đây là ngưỡng chất lượng số, không phải bit-equivalence và chưa chứng minh accuracy detect box. File validation giữ cả kết quả ngưỡng chặt và ngưỡng hình học.

| Biến thể / runtime | MiB | Median ms | p95 ms | SNR vs Torch | Max raw error | Line demo |
| --- | --- | --- | --- | --- | --- | --- |
| decoded_fp32 / 2.2.0 | 2.619 | 107.193 | 118.146 | khác output contract | — | 18 |
| decoded_fp32 / 2.15.1 | 2.619 | 51.183 | 103.511 | khác output contract | — | 18 |
| dynamic / 2.15.1 | 0.697 | 668.929 | 813.869 | 12.171 | 181.86096 | 20 |
| fp16 / 2.15.1 | 1.214 | 105.074 | 116.248 | 20.937 | 142.63797 | 18 |
| fp32 / 2.15.1 | 2.372 | 102.851 | 125.822 | 88.307 | 0.08914 | 18 |
| official / 2.2.0 | 2.376 | 110.938 | 122.798 | khác output contract | — | 24 |
| official / 2.15.1 | 2.376 | 105.265 | 111.809 | khác output contract | — | 24 |
| static / 2.15.1 | 0.756 | 125.828 | 157.894 | 3.158 | 507.82515 | 126 |

| Raw variant | Center SNR dB | Displacement SNR dB | Scale output |
| --- | --- | --- | --- |
| dynamic | 24.787 | 12.117 | 0.000000 |
| fp16 | 33.009 | 20.885 | 0.000000 |
| fp32 | 100.420 | 88.255 | 0.000000 |
| static | 11.052 | 3.121 | 5.646046 |

Scale 0 ở output float nghĩa là không có quantization I/O. Raw head có center logit và displacement khác miền giá trị trong cùng tensor; INT8 per-tensor output dùng chung scale, có thể làm center mất resolution. Đây là giả thuyết cần đối chứng tách head/quantization chọn lọc, không phải nguyên nhân duy nhất đã xác định. FP16 weights làm nhỏ file nhưng số liệu hiện tại cho thấy sai số tăng; chưa có cơ sở bảo đảm chất lượng detection tương đương FP32.

Chỉ đối chiếu tốc độ **decoded_fp32 ↔ official** cùng runtime/thread vì hai graph cùng trả decoder outputs. Raw FP32/FP16/dynamic/static có scope khác và chỉ so trong nhóm raw. Torch reference 1 thread, input đã normalize, nên latency Torch riêng không phải so công bằng với bảng 4 thread TFLite.

Sai số conversion/quantization luôn so với **cùng checkpoint PyTorch**, không dùng official khác checkpoint/graph làm chuẩn. Calibration: 32 ảnh train Imagenette, evaluation: 10 ảnh validation, không trùng hash, seed 20261009. Đây là ảnh thật nhưng chưa phải dataset có nhãn line/box; chưa có precision/recall, sAP hoặc IoU box. SNR toàn raw output có thể che sai số center/displacement; JSON lưu thêm metrics hai nhóm riêng. Line count chỉ là kiểm tra pipeline, không phải accuracy.

Evidence/code: [prepare_torch.py](experiments/03-mlsd/src/prepare_torch.py), [port_keras.py](experiments/03-mlsd/src/port_keras.py), [measure.py](experiments/03-mlsd/src/measure.py), [kết quả](experiments/03-mlsd/results/run-01/).

## 7. Detect box và hướng Android

Luồng source NAVER: decode line → Hough merge → giao điểm → kiểm tra khoảng cách/góc → ghép chu trình bốn cạnh → chấm điểm. Center NMS 3×3; Hough accumulator suppression 5×5. Tọa độ point là (y,x), displacement là (x,y).

[Tài liệu đọc thuật toán và contract Android](experiments/04-box-postprocess/ALGORITHM.md). [Học liệu HTML tương tác](../../docs/2026-10-09/hoc-ngay-2026-10-09.html). [Prototype Java](experiments/04-box-postprocess/src/BoxPostProcessor.java) xử lý giao điểm/chu trình, giới hạn 64 line, qua 5 ca hình học. Chưa port Hough merge và scoring tương đương NAVER. Thử CSV demo chỉ kiểm tra nối pipeline vì CSV chưa merge; latency JVM Windows không phải latency Android.

Smoke test desktop từ 18 line chưa merge: 4 box ứng viên, median 0.056 ms. Đây là kết quả prototype, không phải chất lượng NAVER hoặc tốc độ Android. [Evidence Java](experiments/04-box-postprocess/results/run-01/demo_boxes.json).

Đề xuất triển khai Kotlin/Java với primitive arrays và buffer tái sử dụng trước; chỉ chuyển C++/JNI khi profiler máy thật chứng minh bottleneck. Cần fixtures đối chiếu góc/box, kiểm tra song song/đoạn bằng 0/NaN và biến đổi ngược crop/letterbox.

## 8. Hướng tiếp và phần còn thiếu

1. Đối chứng sentinel đã chạy là hướng thử nghiệm bổ sung hôm nay. Tiếp theo dùng test set độc lập/sequence length khác để kiểm tra khả năng tổng quát; thử quantize chọn lọc attention hoặc QAT nếu PTQ vẫn giảm chất lượng.
2. Phân tích chênh lệch ResNet18 giữa Mac Đức báo cáo và Linux Huy; giữ raw timing, profile kernel riêng và lặp nhiều phiên. Không cần tái tạo phần cứng Mac để chạy code.
3. M-LSD: dùng calibration cùng miền ảnh đường thẳng và dataset có nhãn; đánh giá sAP/IoU sau decoding, thử nhiều subset mà không chọn theo test score.
4. Port Hough merge/scoring, đối chiếu fixtures với source, rồi APK benchmark trên Android thật; đo model/decoder/box/end-to-end riêng.
5. Huy tự đọc và viết lại phần thuật toán theo ý hiểu để đáp ứng yêu cầu mentor; tài liệu hỗ trợ không thay việc này.

## 9. Tái lập và kiểm chứng

[Hướng dẫn chạy](RUNNING.md), [dependency/environment](ENVIRONMENT.md), [matrix](tools/run_matrix.py). Runner chạy tuần tự và ghi exit code từng job. Số job đã có trạng thái: 40; thất bại: 0 (không có trong các job đã ghi). Không coi “chưa đo” là 0 ms.

Metric matrix Huy dùng float64: `SNR=10·log10(sum(ref²)/sum((ref−test)²))`; MAE/max đo trên output dequantized, cosine và relative L2 toàn tensor. Reproduce q4 Đức giữ thêm SNR float32 đúng source để đối chiếu, đồng thời ghi SNR float64. Median/p95 tính lại từ raw latency samples. Model/venv/cache/credentials không đưa vào Git. Bản chat cục bộ được ignore theo cấu hình repo. Bàn giao qua Git: code, báo cáo, source/dependency lock và JSON/CSV kết quả. Raw tensor, model/dataset, profiler trace, cache, credentials, tài liệu nội bộ và báo cáo chat giữ cục bộ; clone mới tạo lại artifact trước khi chạy verifier đầy đủ.
