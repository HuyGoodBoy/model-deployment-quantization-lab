# Kế hoạch ngày 20/10/2026

Người dùng yêu cầu thực hiện toàn bộ, Astra lên kế hoạch trước và review sau. Báo cáo cuối theo mẫu “Anh ơi, em Huy báo cáo ngày 20/10 ạ”, khoảng 5–6 gạch đầu dòng.

## Zipformer → LiteRT 2.x

- Pin revision HF `d42f2d9f7ca24806fb667456a18a9f1b60f70d16`, dùng ba component FP32 encoder/decoder/joiner, tokens và audio/transcript đi kèm. Kiểm SHA256 công khai trước inference. Không gọi bản model khác là bản chuyển đổi của model này.
- Audit graph, shape/dtype/metadata, streaming state, blank/token context, chunk/hop. Chạy baseline ONNX trước. Giữ cùng frontend và greedy decoder khi so runtime, reset state giữa utterance, xử lý đoạn cuối và 8 kHz đúng cách.
- Thử convert nguyên trạng bằng onnx2tf rồi mới sửa lỗi cụ thể. Lưu lệnh, phiên bản, stdout/stderr, lỗi, patch và model hash. onnx2tf mới có backend FlatBuffer trực tiếp: không mặc định chuyển qua TensorFlow; ghi backend thực dùng. Runtime đích phải là LiteRT 2.x.
- Kiểm decoder/joiner trước, encoder sau. Các sửa shape cho batch1/chunk cố định phải đúng metadata streaming, không cast INT64 hàng loạt. Mọi bản sửa phải qua tensor parity và streaming nhiều chunk.
- Nhánh tối ưu dùng tối đa hai sửa có căn cứ: shape inference/constant folding, bỏ node dư hoặc rewrite pattern cụ thể. Giữ nguồn nguyên vẹn, lưu before/after và kiểm tensor, transcript/wer, latency/RTF. Không bắt buộc phải có speedup.
- Audio kèm model dùng smoke/parity. Nếu đánh giá WER, khóa tập audio và transcript trước khi chọn tối ưu, ghi số utterance/từ/phút và tính tổng edit-distance/tổng reference words. Không gọi WER trên vài audio smoke là độ chính xác toàn bộ model.
- Claude phải được gọi thật nếu người dùng đăng nhập: lưu tool/model/version, prompt, phản hồi, patch và kết quả kiểm chứng. Đang có Claude Code nhưng chưa đăng nhập; đã hỏi người dùng, tiếp tục baseline độc lập trong lúc chờ. Không gán công việc của Codex/Astra cho Claude.

## M-LSD: QDQ và AI Edge Quantizer

- So warm-profile FP32/QDQ trên cùng ONNX Runtime, CPU/thread/input. Ánh xạ Conv theo layer/shape, đo cùng đầu vào tại lớp cho microbenchmark nếu cần. Profiling time không thay inference-only latency.
- AI Edge Quantizer nhận LiteRT float, không sửa ONNX QDQ. Dùng LiteRT FP32 cùng checkpoint và kiểm parity trước; thử AEQ default và một recipe selective giữ float vùng được xác nhận chậm. Đây là PTQ mới, không giả vờ giữ QAT scale cũ.
- Regex scope phải khớp tensor/output thực. Kiểm graph/kernel/dtype sau quantization; chọn trên 16 ảnh development; 54 ảnh cũ chỉ kiểm tra hồi quy sau khóa chọn.
- Benchmark tuần tự, không chạy training/conversion khác đồng thời. Giữ số thread, input, warm-up và số lượt; báo median/p95, sAP10, size và parity/gate. Không so timing tuyệt đối khác OS/backend để quy riêng tác động precision.

## Kiểm chứng, review và commit

- Astra review bằng chứng và cách diễn giải sau khi chạy; không gọi kiểm chứng bằng chứng đạt là model đã qua gate chất lượng.
- Lưu README link, REPORT ngắn, RESEARCH/RUNNING và JSON/CSV; artifact/model/audio/trace lớn được ignore.
- Người dùng yêu cầu lịch commit một lần lúc **17:00 ngày 20/10/2026, Asia/Saigon**, tên `20/10`. Chỉ commit snapshot đã kiểm chứng/review; không tự push. Đăng ký Windows Scheduled Task nếu có thể; ghi rõ kết quả đăng ký và yêu cầu máy còn hoạt động. Không hứa đã đặt lịch nếu chưa xác nhận task tồn tại.

Nguồn: [model HF](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17), [onnx2tf](https://github.com/PINTO0309/onnx2tf), [AEQ](https://github.com/google-ai-edge/ai-edge-quantizer), [LiteRT](https://developers.google.com/edge/litert/overview).
