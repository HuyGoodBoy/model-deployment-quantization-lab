# Nghiên cứu ngày 22/10/2026

## M-LSD: giảm chi phí ở biên model

- Evidence: [Graph M1](experiments/mlsd/results/model_M1.json) · [Graph M2](experiments/mlsd/results/model_M2.json) · [Probe rounding](experiments/mlsd/results/quantize_probe.json) · [Đối chứng đóng gói](experiments/mlsd/results/roundtrip_control.json).

### Giả thuyết và cách kiểm

Phép đo trước cho thấy quantize input ngoài model tốn khoảng 1,6 ms. Em thử đưa công việc này vào LiteRT bằng QUANTIZE với đúng scale **0,007836945354938507**, zero-point **0**, giữ tensor INT8 cũ và toàn bộ graph phía sau. Đây là thay đổi nơi thực hiện quantize, không phải lượng tử hóa lại hoặc giữ thêm lớp ở float.

Theo [đặc tả quantization LiteRT](https://ai.google.dev/edge/litert/performance/quantization_spec), giá trị thực được biểu diễn bởi scale và zero-point. Tuy nhiên, cùng tham số chưa bảo đảm mọi phép làm tròn khớp bit giữa NumPy và kernel. [Reference quantize của TensorFlow](https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/kernels/internal/reference/quantize.h) dùng phép làm tròn sau khi chia scale; kernel delegate có thể dùng cách tính khác. Source này chỉ giúp đặt ca thử, không xác nhận kernel thực tế của wheel LiteRT 2.2.0.

Em kiểm trực tiếp wheel đang dùng, gồm chế độ mặc định và builtin không delegate. Trên 32 ảnh calibration cùng 16 ảnh development, hai chế độ đều khớp toàn bộ input INT8 với adapter NumPy. Trên 65.536 giá trị random hữu hạn trong [-4; 4], không có mismatch. Trên 792 giá trị quanh half-bin và hai giá trị float32 kề bên, clipping và zero, có 50 mismatch, tối đa 1 LSB ở mỗi chế độ. Các giá trị đều giải thích được trong các phương án chia/nhân reciprocal float32 và ties-even/ties-away đã kiểm.

Probe riêng không chứng minh kernel QUANTIZE trong graph đã delegate sẽ làm tròn giống hệt trên mọi input. Vì vậy, em kiểm tiếp model đầy đủ: output M1/M2 khớp C1 trên 16 ảnh development ở 4/8 luồng và 54 ảnh hồi quy ở 8 luồng; sAP giữ nguyên. Kết luận giới hạn trong các input/cấu hình đã thử.

### External buffer khi sửa FlatBuffer

[Schema LiteRT](https://github.com/tensorflow/tensorflow/blob/v2.21.0-rc0/tensorflow/compiler/mlir/lite/schema/schema.fbs) có Buffer.offset/size ngoài Buffer.data. C1 sử dụng 44 external buffer, tổng **1.749.472 bytes**. InitFromObj rồi Pack trực tiếp làm mất vùng payload nằm ngoài FlatBuffer chính: file thử chỉ còn 156.136 bytes thay vì 1.905.424 bytes. Em phát hiện và dừng trước inference; file này không được đưa vào đánh giá hoặc benchmark. [Ghi nhận lỗi đóng gói](experiments/mlsd/results/serialization_attempt.json).

Cách xử lý là kiểm giới hạn offset/size, chép payload vào Buffer.data, đặt offset/size về 0, rồi đọc lại file đã ghi để đối chiếu từng buffer. Các large custom option ngoài file bị từ chối để tránh bỏ sót dữ liệu. Bản chỉ đóng gói lại không thêm op được kiểm output trước khi dùng làm đối chứng.

M1 vẫn nhanh hơn đối chứng này 6,81% và 5,70% ở hai phiên 8 luồng. M2 thêm TRANSPOSE để chuyển NHWC sang NCHW trong model, nhưng lợi ích bổ sung không nhất quán nên chưa chọn. Số liệu này không đủ để kết luận QUANTIZE được fuse vào một kernel cụ thể.

### Phạm vi profiler

Em dùng phân rã wall time ở phía ứng dụng để tách quantize, set_tensor, invoke, get_tensor và output adapter. Đường gọi này dùng để chẩn đoán, không thay benchmark chính. Danh sách op từ `_get_ops_details()` có thể chứa cả op gốc sau delegation; số Q/DQ trong danh sách không phải số kernel Q/DQ thực sự chạy.

Wheel Interpreter đang dùng không có public API trả thời gian từng kernel. Em kiểm hướng build [benchmark C++ LiteRT](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/tools/benchmark/CMakeLists.txt) và [CMake của bản 2.2.0](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/tflite/CMakeLists.txt): cần compiler C++20, CMake cùng source/dependency TensorFlow. Image hiện tại chưa có toolchain này. Em dừng ở kiểm tra điều kiện build, chưa build runtime khác hoặc đo native per-kernel; tránh lấy profile ONNX Runtime để suy thời gian kernel LiteRT. [Hướng dẫn CMake chính thức](https://ai.google.dev/edge/litert/build/cmake).

## ZipFormer: fixed-shape wrapper

- Evidence: [Wrapper Z1](experiments/zipformer/src/runtime22.py) · [Fixture](experiments/zipformer/results/fixture_parity.json) · [Stream liên tục](experiments/zipformer/results/continuous_state.json) · [Lựa chọn](experiments/zipformer/results/selection.json).

Em đối chiếu [source Interpreter và SignatureRunner](https://github.com/google-ai-edge/LiteRT/blob/v2.1.2/tflite/python/interpreter.py). SignatureRunner thực hiện resize và allocate trong mỗi lần gọi. Với input/state có shape cố định, có thể cache mapping tên→tensor index, dùng set_tensor/invoke/get_tensor và bỏ phần chuẩn bị lặp lại này.

Wrapper thử nghiệm chỉ chấp nhận đúng shape và dtype đã kiểm. Em giải phóng SignatureRunner trước khi dùng Interpreter.invoke để không giữ reference tới native wrapper; không tắt safety guard hoặc giữ tensor view qua invoke. Phần chuyển counter INT32→INT64 dùng lại quy tắc cũ.

Hai wrapper chạy cùng graph nên em yêu cầu bit-exact trên fixture, không dùng ngưỡng sai số rộng để che lỗi mapping. Sau fixture là regression 173 câu và stream liên tục 600 giây. Stream được tạo bằng lặp một audio, giữ state riêng của mỗi wrapper; chưa đo tất cả state tensor ở mỗi chunk. Kiểm này không thay đánh giá WER trên dữ liệu mới hoặc kiểm counter ở mọi độ dài session.

Kết quả cho thấy joiner có giảm chi phí gọi, nhưng encoder không nhanh hơn và mức giảm RTF toàn pipeline chưa tách khỏi dao động. Vì vậy em giữ SignatureRunner, chưa chuyển cách gọi mặc định.

## Cách diễn giải phép đo

Hai phiên là hai process trên cùng máy và cùng ngày. Bootstrap lấy lại mẫu theo 5 vòng đo, giữ cặp control/candidate trong cùng vòng; không coi các lần gọi trong một vòng là quan sát độc lập. Khoảng 95% chỉ là ước lượng với ít vòng, chưa phải bảo đảm cho thiết bị đích.

Ngưỡng sAP và quy tắc chọn tốc độ được giữ nguyên sau khi bắt đầu đo. Kiểm chất lượng 8 luồng thực hiện sau loạt timing và trước chọn bản cuối. Tập 54 ảnh được chạy sau khi chốt lựa chọn, không dùng để chọn lại M1/M2. Các dữ liệu này và 173 câu ZipFormer đã có từ trước, được dùng để kiểm hồi quy.
