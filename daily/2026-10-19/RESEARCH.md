# M-LSD: chủ động chọn vùng float, ngày 19/10

Câu hỏi: chủ động bỏ lượng tử hóa ở một cụm lớp có cải thiện chất lượng–latency so với QDQ ngày 16 không? Đây là thử nghiệm mới trong dự án, không tuyên bố phương pháp mới về học thuật. Thiết kế nằm ở [PROTOCOL.md](PROTOCOL.md).

## Điều đã xác nhận về fusion

QDQ baseline chạy 22 QLinearConv và 19 Conv float trên CPU ORT 1.30.0. Custom export ngày 15 chạy 41 ConvInteger, là artifact khác; không dùng câu “chưa fuse QDQ” để giải thích chung cả hai. Tắt toàn optimizer trên cùng QDQ cho 41 Conv float, nhưng phép so đó còn thay constant folding, layout và các fusion khác.

Đối chứng bổ sung giữ ALL và chỉ vô hiệu `QDQSelectorActionTransformer`: số QLinearConv giảm 22 → 0. Median của pass bật là **153.594 ms**, tắt là **177.546 ms**, p95 lần lượt 191.157/230.632 ms. Chênh median theo (1 − bật/tắt) là 13.49%. Đây là hiệu ứng của pass này trong cấu hình đo, không phải chi phí thuần của một kernel Conv và không bảo đảm fuse nhiều luôn nhanh. [Raw timing, trace và metric](experiments/selective/results/fusion_control.json).

Graph audit ghi nhận 19 Conv float gồm 3 Conv có Clip→Pad trước quantizer kế tiếp, 9 projection Conv có Add/residual hoặc nhiều consumer, và 7 Conv head có Relu→Resize/Concat/Add hoặc output Slice. Tất cả group=1. Đây là các pattern quan sát được, chưa chứng minh mỗi node không fuse do duy nhất một điều kiện. Nguồn ORT mô tả mẫu thay Conv bằng QLinearConv cần các DQ đầu vào và Q đầu ra; graph thực tế và kernel trace mới quyết định coverage.

## Chọn vùng và giữ đối chứng

Trên bốn ảnh development 0,5,10,15, từng Conv fake quant được so với Conv float cùng weight và **cùng input**, tránh quy sai số truyền từ trước thành độ nhạy riêng của lớp. Gom theo block, chọn tích thời gian Conv warm-profile × relative L2 cục bộ lớn nhất. Block được chọn: `model.model.block16`. Heuristic này chỉ chọn vùng thử; không chứng minh đó là nguyên nhân duy nhất của sai số toàn mạng.

- A: bỏ lượng tử input/weight tại block16, 3 Conv; giữ output quantizer.
- B: mở rộng sang block15 và block16, tổng 5 Conv; giữ output quantizer.
- Phần còn lại giữ nguyên checkpoint QAT, weight, bias, scale/zero-point và decoder. Không train, không fit calibration mới.
- Teacher FP32 phân biệt với `qat_weights_fp32`: control sau dùng weight QAT và tắt toàn bộ fake quant, giúp phân biệt tác động weight với precision.

Kernel float không đồng nghĩa không bị lượng tử hóa: Conv float của baseline có thể nhận input/weight đã Q/DQ. A/B chủ động bỏ lượng tử ở vùng đã chọn. Cả A và B chạy 19 Conv integer +22 float nhưng không cùng precision semantics. A vẫn 40 DQ/19 Q như baseline; B còn 36 DQ/17 Q. Chỉ B giảm số boundary Q/DQ theo phép đếm này; số node không phải số byte chuyển hay tổng thời gian.

## Chất lượng và tốc độ

| Model | Dev sAP10 (%) | Hồi quy sAP10 (%) | Median ms | p95 ms | Conv integer/float | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| teacher_fp32 | 15.1617 | 12.7802 | 85.804 | 103.455 | 0/41 (graph float) | 2502514 |
| custom_41_integer | 13.7334 | — | 311.538 | 402.633 | 41/0 | 760882 |
| qdq_all | 13.6884 | 11.9598 | 144.531 | 195.072 | 22/19 | 773207 |
| qdq_optimizer_off | 14.0390 | — | 188.577 | 242.131 | 0/41 | 773207 |
| selective_a | 13.9274 | 12.2811 | 141.153 | 169.642 | 19/22 | 992970 |
| selective_b | 14.0675 | 12.3090 | 133.265 | 170.254 | 19/22 | 1210938 |
| qat_weights_fp32 | 15.3230 | 13.2033 | 83.394 | 99.031 | 0/41 (graph float) | 2512757 |

Benchmark cùng runtime, batch1, 4 intra/1 inter threads, sequential, spinning tắt. Chỉ một session tồn tại mỗi lượt, 5 vòng xoay/đảo thứ tự, warm-up16 +timed16 mỗi model/vòng, 80 timing/model; profiler tắt. Không gồm load/preprocess/decoder. Median từng vòng được lưu; B thấp hơn QDQ ở 5/5 vòng. Các vòng là đo lặp trên cùng máy, không phải 80 thí nghiệm độc lập. Chênh tốc độ quan sát chưa bảo đảm tổng quát sang máy khác.

Lựa chọn đã khóa: **selective_b**, theo development và latency trước khi inference hồi quy. Giữ giới hạn giảm AP không quá 0,5 điểm và chỉ chọn A/B nếu median thấp hơn QDQ. Đây là lựa chọn nghiên cứu, không phải xác nhận qua release gate.

B tăng 0.3492 điểm sAP10 trên 54 ảnh hồi quy, bootstrap ảnh ghép cặp 1.000 lần seed20261019 cho CI95% [0.1015; 0.6987] điểm. Tập này đã dùng ngày16: không phải holdout mới, CI chỉ mô tả tập đã đo. Cả A/B được báo đủ, không chọn lại sau khi xem hồi quy.

## Sai lệch export và profiler

Relative L2 ONNX so PyTorch **cùng policy**: A 8.2541%, B 8.1634%; cả hai vượt gate 2%. Chênh sAP10 export cũng được lưu riêng. Sai lệch do đổi precision so QAT gốc là phép đo khác, không dùng thay gate export. QAT-weights FP32 export relative L2 0.001410%.

Warm profile baseline: mean model_run 162.444 ms; Conv integer 65.596, Conv float 53.358, Q/DQ/Transpose 28.596 ms. Đây là thời gian **có overhead profiler**, không lấy thay benchmark hoặc kết luận mọi chênh latency đều do chuyển đổi. Trace chỉ dùng 8 model_run cuối sau 16 warm-up, loại warm-up/startup. Không dùng trace một lượt cold ngày16 để phân bổ bottleneck.

## Kiểm chứng và giới hạn

Verifier tái tính metric, bootstrap, median/p95 và lựa chọn; kiểm hash nguồn cũ không đổi, weight/bias/qparams ONNX, kernel/provider/dtype và graph đã tối ưu. Chạy lại 16 ảnh cho ba model mới và 54 ảnh cho model đã chọn, yêu cầu output khớp tuyệt đối; baseline FP32/QDQ trên hồi quy phải khớp output ngày16. `status=passed` xác nhận bằng chứng nhất quán; gate model vẫn thất bại.

Timing hôm nay khác các ngày trước; không trộn các số để quy khác biệt cho fusion. Không dùng kết quả CPU ORT1.30 để kết luận GPU ORT1.20.2 hay Android. Các JSON giữ hash script khi chạy; `source_archive.json` giữ bản source trước dọn lint/format để kiểm provenance, source hiện tại dùng tái lập.

Kết luận: B là phương án nghiên cứu tốt hơn QDQ baseline trên cấu hình đã đo: median giảm 7.80%, AP hồi quy tăng 0.3492 điểm. Đổi precision vùng head và giảm Q/DQ diễn ra cùng nhau nên chưa cô lập đóng góp riêng từng yếu tố. Đối chứng weight QAT chạy float đạt 13.2033% và 83.394 ms, tốt hơn B cả chất lượng lẫn latency; B chỉ cải thiện so với QDQ cũ, không phải cấu hình tốt nhất toàn bộ. Model B nhỏ hơn bản float (1210938 so với 2512757 bytes), nhưng chưa đo RAM runtime. Model vẫn chậm hơn FP32 và chưa đạt gate export; bước sau cần xử lý parity rồi kiểm tra dữ liệu độc lập hoặc thiết bị đích. Link source ORT main chỉ minh họa quy tắc được đọc ngày19/10; coverage và hiệu ứng pass của phiên bản1.30.0 được xác nhận bằng graph/trace thực tế đã lưu.

Nguồn: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html), [Conv QDQ rule](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc).
