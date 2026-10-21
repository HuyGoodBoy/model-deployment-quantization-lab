# Ghi chép nghiên cứu ngày 21/10

## ZipFormer

- Evidence: [Profile encoder](experiments/zipformer/results/encoder_profile.json) · [Rewrite](experiments/zipformer/results/encoder_fc_rewrite.json) · [Test streaming liên tục](experiments/zipformer/results/continuous_state_test.json) · [Manifest mới](experiments/zipformer/results/multispeaker_manifest.json).

### Vì sao chuyển BMM hằng sang FullyConnected?

Graph LiteRT ngày 20 có 152 BATCH_MATMUL, kể cả những phép linear có weight hằng `[K,N]`. Shape cố định và hằng RHS cho phép viết lại thành FC với weight `[N,K]`, giữ số chiều output. Trái lại, 40 BMM attention còn phụ thuộc tensor runtime nên không áp dụng quy tắc này.

Đây là tối ưu biểu diễn operator/weight layout, không giảm precision, không bỏ attention và không prune weight. Native model có 112 FC sau sửa, không fuse bias hoặc Add để hạn chế thay đổi số học. Các buffer cũ được compact lại, tránh model tăng gấp đôi do giữ cả hai bản weight.

[Quy tắc versioning TensorFlow v2.19](https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/tools/versioning/op_version.cc) ghi keep_num_dims từ FC version 5. Patch dùng 3 input slot, bias optional `-1`, FC v5; runtime LiteRT 2.1.2 allocate/invoke và parity thực tế là bước xác nhận cuối. Không suy ra hỗ trợ chỉ từ đọc schema.

Profile dùng Windows ORT 1.30.0, 4 threads, 8 warm-up + 8 lượt có ghi node events. Tổng MatMul khoảng 6,47 ms/run trong profile, lớn nhất trong các nhóm; số này chịu overhead profiler và không phải latency MatMul trên LiteRT. Native inventory ban đầu còn có 350 RESHAPE và 233 TRANSPOSE, nhưng hôm nay không tự bỏ chúng vì cần chứng minh điều kiện layout cho từng pattern.

Benchmark so hai encoder trên **cùng LiteRT 2.1.2**; decoder/joiner giữ nguyên. Encoder baseline có median từng vòng `[8,820; 8,750; 8,586; 8,879; 8,660]` ms, bản FC `[5,927; 6,369; 6,311; 5,833; 5,937]` ms. Cả 5 vòng đều cải thiện; giảm encoder 29,7% và pipeline RTF 23,4% là hai đại lượng khác nhau. Không quy toàn bộ cải thiện cho prepacking khi chưa profile nội bộ delegate.

### Kiểm thử chất lượng bổ sung

Model gốc được pin revision/SHA từ ngày 20; bản FC giữ input/output và 35 state của bản chuyển đổi đó. Kiểm 15 fixture bảo đảm36 output/fixture allclose1e-4, counter exact; hai lượt 73 câu giữ91/1150 lỗi và token/counter giống nhau.

Tập nhiều speaker lấy từ [OpenSLR12](https://www.openslr.org/12), license CC BY4.0. Archive test-clean có [MD5 công bố](https://www.openslr.org/resources/12/md5sum.txt) `32fa31d27d2e1cad72775fee3f4849a9`; đồng thời lưu SHA256 archive và từng file. Download bị ngắt đã được resume bằng HTTP Range có kiểm Content-Range, chỉ dùng dữ liệu sau khi checksum toàn archive đúng.

Chọn seed20261021: random20speaker rồi5câu/speaker, loại speaker/ID thuộc73câu cũ, không lọc theo model. Manifest khóa sau khi rewrite đã xong và trước lượt inference bổ sung. 100 câu dài853,15 giây,2.295 từ; cả hai bản117 lỗi, WER5,0980%, token/final counter khớp100/100. Không thay patch sau khi xem kết quả; không gọi đây là full test-clean WER hay bằng chứng dữ liệu chưa có trong pretraining.

Test stream79,5giây là lặp12lần một audio, không reset state giữa các lần. Nó kiểm đường state carry qua249chunk, không thêm dữ liệu đánh giá độc lập và không chứng minh cached_lenINT32 an toàn với mọi độ dài. Pipeline giữ context[0,0] theo Icefall-style của ngày20; không so tốc độ với native Sherpa có context khác.

### Công cụ thực dùng

Người dùng xác nhận không có Claude và yêu cầu dùng Codex thay thế ngày21. Các patch/kiểm tra ngày này do Codex thực hiện với Python, NumPy, FlatBuffers object API, ONNX Runtime profiler và LiteRT Interpreter. Không đổi attribution hoặc xóa lịch sử lỗi đăng nhập của ngày20; không có claim về tool/model execution của Claude.

## M-LSD

- Evidence: [CSV activation](experiments/mlsd/results/activation_summary.csv) · [Split FP32 parity](experiments/mlsd/results/split_float_parity.json) · [Recipes và I/O](experiments/mlsd/results/exports.json) · [Quyết định trước hồi quy](experiments/mlsd/results/selection.json) · [Phân tích độ biến động](experiments/mlsd/results/timing_analysis.json).

### Tách chẩn đoán với phép đo tốc độ

Tensor diagnostic dùng BUILTIN_WITHOUT_DEFAULT_DELEGATES, preserve_all_tensors và1thread để đọc được intermediate; benchmark dùng Interpreter CPU/XNNPACK4/8threads. Không lấy thời gian debug để kết luận hiệu năng triển khai. Hai chế độ có thể có sai số số học khác; diagnostic là bằng chứng định hướng vị trí sai lệch, không cô lập nguyên nhân một kernel.

Bốn ảnh có index0,1,8,15 được chọn trước trong development. Ghép41 Conv bằng output identity/shape; khi dequant tensor dùng actual scale/zero-point. Mean relative L2 ba lớp đầu lần lượt0,7234%,10,5316%,35,8724%, trong khi phần reference ngoài range của ba tensor này đều0%. Endpoint occupancy INT8 không được dùng thay cho bằng chứng clipping.

Conv cuối16filter, kênh7 làcenter,8:12 làdisplacement. Ví dụ ảnh0 ở Conv cuối, rangecenter khoảng[-33,12;4,54], displacement[-764,29;754,06]; scale chung2,3583 rõ ràng quá thô vớicenter. Tuy vậy sai số đã tích lũy trong backbone, nên không kỳ vọng chỉ sửaoutput sẽ giải quyết toàn bộ.

### Hai đối chứng có điều kiện rõ ràng

A giữ mọi op head ngoài backbone ở float, không chỉ ba Conv cuối: graph thực30 Conv weight/outputINT8 và11float. Nhánh B tách filter trước lượng tử hóa, resize độc lập rồi xuất hai output nativeNHWC. Center scale0,0961113, zero-point85; displacement scale2,3583019, zero-point0. Adapter dequant từng nhánh, transpose và concat sangNCHW5; toàn bộ chi phí này được tính trong latency.

Control splitFP32 cho output bit-exact với FP32 gốc trên16ảnh, chứng minh việc bỏ filter không dùng/tách nhánh không tự thay decoder hoặc đổi chất lượng trước PTQ. SplitINT8 có42Conv, không gọi nó là cùng41Conv với baseline. Range riêng làm devAP từ2,8522 lên12,0048 nhưng còn dưới15,3230 củafloat. A đạt14,8765, tức giảm0,4466điểm sofloat, qua gate0,5điểm đã khai báo.

Weight, dữ liệu32calibration/16development và decoder không đổi; input byte hash được ghi và verifier kiểm lại. Không train, không tăng calibration cùng lúc. Đây là PTQ mới, không chỉnh ONNX QDQ và không giữ scale QAT trước đó.

### Gate chất lượng, tốc độ và giới hạn

Primary rule chọn trên16dev: alternative phải trong0,5điểm sAP sofloat và nhanh hơn matchedfloat4threads. Chỉ A đạt; khóa chọn trước khi đọc54ảnh regression. B vẫn được công bố theo kế hoạch đối chứng, không chọn lại theo điểm hồi quy. Kết quả A13,1121 sofloat13,2033 trên54ảnh chỉ là hồi quy cũ.

AP gần nhau không có nghĩa tensor gần nhau: A relativeL2 toàn output46,22%,center12,14%,displacement46,50%. Decoder lọc/NMS/chọn đoạn, còn L2 tính cả bản đồ, nên hai metric đo hai mục tiêu khác nhau. Không lấy AP để báo gate export2% của QAT đã đạt; đây là model PTQ/policy khác.

Lợi thế median4threads chỉ3,80%. Bootstrap dùng5round làm cluster, lấy lại các round chung cho hai model,2.000resample,seed20261021; khoảng95% xấp xỉ[-1,75%;10,29%]. Không coi80lượt là80mẫu độc lập. Ở8threads, A chậm hơn2,66%, không nhanh hơn ở round median nào trong5vòng; khoảng xấp xỉ[-3,90%;-1,01%] cho tỷ lệ giảm latency. Những khoảng này mô tả biến thiên trong5vòng, không bảo đảm cho thiết bị/phiên khác.

Kết luận kỹ thuật: A phục hồi chất lượng và giảm23,1% dung lượng file, có giá trị như một trade-off; chưa là phương án tăng tốc tốt hơn FP32 8threads trên host này. Chưa đo RAM, năng lượng hoặc Android. Quy tắc chọn4threads và khuyến nghị cấu hình nhanh nhất8threads là hai câu hỏi khác nhau.

## Kiểm chứng và nguồn

[VERIFICATION.json](VERIFICATION.json) tái tính AP/WER/timing/uncertainty, kiểm weight transpose, dtype, dữ liệu, khóa chọn trước regression và source ngày20. File nguồn/report ngày20 giữ nguyên. Source_code hash trong từng result ghi phiên bản lúc lưu; code cuối đã được Ruff định dạng, snapshot cuối ghi hash phiên bản bàn giao.

Thư viện tham khảo: [AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer), [LiteRT Interpreter](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/python/interpreter.py). API và precision thực tế được đối chiếu thêm trực tiếp với package0.9.0/LiteRT2.2.0 đã pin, không lấy README phiên bản mới làm bằng chứng runtime của ngày này. Lệnh chạy: [RUNNING.md](RUNNING.md).

## Bổ sung sau debate: giả thuyết và kết quả thực đo

- Evidence: [Protocol khóa trước sweep](PROTOCOL_REFINEMENT.md) · [REFINEMENT_ANALYSIS.json](REFINEMENT_ANALYSIS.json) · [REFINEMENT_VERIFICATION.json](REFINEMENT_VERIFICATION.json).

### ZipFormer: fuse ADD không đồng nghĩa tự động nhanh hơn

Kiểm graph tìm được 92 FC có duy nhất consumer ADD, bias hằng 1D FP32 đúng shape, activation identity và không có public FC output. Fuse bằng slot bias FC, bỏ 92 ADD, giữ nguyên toàn bộ bytes buffer/tensor/signature. Allclose trên 15 fixture giữ nguyên counter; đánh giá 173 câu và stream 249 chunk cũng khớp token. Fusion đổi vị trí rounding nên không yêu cầu activation bit-exact.

Encoder median FC 6,1456 ms, fuse bias 5,9896 ms. Gain 2,54% có khoảng bootstrap [-1,03%; 6,20%]; RTF 0,029515 so 0,029751. Chưa có lợi ích pipeline để promote patch bias. Giữ FC cũ không phủ nhận tính hợp lệ của fusion; chỉ tách correctness với hiệu quả thực đo.

Kiểm serialization giải thích được 936 byte tăng ở rewrite FC đầu tiên: payload buffer baseline/FC bằng nhau, chênh nằm ngoài payload. Không có bằng chứng quy riêng toàn bộ chênh lệch cho options/bias. RSS encoder trong 15 process mới xấp xỉ 205 MiB, không thấy tăng rõ ở phạm vi này; không suy thành chứng minh prepacking hay RAM trên Android. Init gồm constructor/allocation/metadata, filesystem cache không flush.

### M-LSD: float output không còn cần split chỉ để có scale riêng

Prune một Conv còn 5 filter giữ exact output FP32 trên 16 ảnh. Nếu Conv cuối float, một output 5 kênh là đối chứng đơn giản hơn hai nhánh float rồi concat. Sweep đã khai báo tối đa ba recipe giữ calibration/dev unchanged:

- C1: 30 Conv INT8/11 float, dev 14,8765%, qua gate mất tối đa 0,5 điểm.
- C2: 40 INT8/1 float, dev 14,5570%, mất 0,7660 điểm, trượt.
- C3: 38 INT8/3 float, dev 14,7760%, mất 0,5470 điểm, vẫn trượt; không nới gate dù chỉ hụt 0,0470 điểm.

C2 phục hồi mạnh hơn static, nhưng không đủ để chọn. Chỉ C1 chạy timing đầy đủ và hồi quy; hồi quy 13,1121% so FP32 13,2033%, cùng điểm A cũ. Không tuyên bố tương đương chất lượng hoặc fresh holdout. Cũng không thay backbone features1.conv1 dựa trên L2 tích lũy, vì chưa cô lập nguyên nhân tại lớp đó.

Hai timing exploratory ban đầu trả C1 dưới dạng NCHW view, còn FP32/A contiguous. View không sai contract cũ, nhưng có thể đẩy locality/copy cost sang decoder bị loại khỏi timing. Phép đo cuối áp dụng ascontiguousarray cho tất cả, kiểm exact/C-contiguous, chạy lại hai process session 4/8 luồng và lưu prefix refinement_contiguous_. Không thay precision hoặc tune theo hồi quy trong lúc làm rõ contract.

Ở 4 luồng, C1 tốt hơn matched FP32 2,92% và 4,80%; hai khoảng bootstrap trong phiên đều trên 0. Ở 8 luồng nhanh hơn 0,98% và 1,66% theo median, nhưng hai khoảng đều chứa 0. Năm paired round vẫn ít; hai process cùng host/ngày không phải chứng minh across-day stability. Registry proposal C1 theo point gate là lựa chọn nghiên cứu, chưa phải promote model thay FP32 8 luồng.

### Diagnostic I/O không phải native node profile

Đường set_tensor/invoke/get_tensor được kiểm output exact với signature trên một input cho mỗi model/cấu hình luồng trước timing. SignatureRunner giữ reference native wrapper; cần giải phóng nó trước direct Interpreter.invoke để dùng safety guard bình thường, không bypass _ensure_safe. get_tensor trả bản copy; không giữ tensor() view khi invoke.

Ở 8 luồng, A native invoke khoảng 19,06 ms so FP32 20,04 ms, nhưng NumPy quantize input khoảng 1,62 ms ăn phần lợi ích. C1 native invoke 18,07 ms, input quantize 1,61 ms và toàn output adapter, gồm copy contiguous, khoảng 0,26 ms; chưa cô lập copy riêng. Tổng tính theo từng sample; không cộng median từng stage. Đây là diagnostic khác plumbing/tập ảnh so benchmark, không dùng làm số end-to-end thay thế.

Mỗi bản có một DELEGATE node theo inventory. _get_ops_details còn giữ node gốc sau delegation nên không phải active execution plan; không suy các Q/DQ gốc đều chạy kernel CPU. Pinned Python wrapper không expose profiler result/execution plan, image không có compiler/native benchmark sẵn; chưa build profiler C++. Không dùng số ORT ngày 20 làm timing kernel LiteRT.

Hướng tiếp theo có căn cứ là thử tối ưu quantize input ở native boundary/contract ứng dụng và đo lại; chưa có kết quả chứng minh gain. Tách từng yếu tố: policy float, prune, copy output, input quantization, runtime/thread. Không gọi 8 luồng chắc chắn tràn sang E-core; affinity guest WSL2 chưa chứng minh P-core vật lý.

Nguồn sơ cấp phục vụ kiểm phương pháp: [LiteRT benchmark/profiler](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/tools/benchmark/README.md), [Profiler interface v2.2](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/core/api/profiler.h), [XNNPACK backend](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md), [Intel i7-12700H](https://www.intel.com/content/www/us/en/products/sku/132228/intel-core-i712700h-processor-24m-cache-up-to-4-70-ghz/specifications.html). Runtime thực đo vẫn theo version/image đã pin, không lấy source main làm bằng chứng chắc về kernel cụ thể.
