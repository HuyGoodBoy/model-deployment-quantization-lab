# Thực nghiệm ngày 22/10/2026

## M-LSD

- Mục tiêu: giảm khoảng 1,6 ms quantize input ngoài model và phần output adapter của C1. Giữ checkpoint, weight, scale/zero-point, 32 ảnh calibration tham chiếu, 16 ảnh development và decoder.
- Đối chứng: FP32 và C1 ngày 21, cùng input float NCHW và output float C-contiguous NCHW 5 kênh. Thêm C1_roundtrip chỉ đổi external buffer sang embedded để kiểm ảnh hưởng đóng gói; tổng tối đa 5 model gồm 3 control và 2 candidate. Không lượng tử hóa lại hoặc thay precision.
- Tối đa hai model sửa I/O: M1 thêm input float và QUANTIZE tới tensor INT8 cũ; M2 thêm TRANSPOSE output NHWC→NCHW vào M1. Chỉ tạo M2 khi M1 qua kiểm cấu trúc/rounding và chất lượng.
- Trước inference đầy đủ, kiểm native QUANTIZE bằng model nhỏ với đúng scale/zero-point: 32 calibration + 16 development, các half-bin và nextafter hai phía, clipping, zero và random finite. Báo mismatch/max LSB so adapter NumPy. Sai khác có giải thích tối đa 1 LSB được đánh giá sAP tiếp; sai khác lớn hơn hoặc không giải thích được thì dừng. Không thay scale để ép khớp. Không khẳng định probe riêng chứng minh kernel trong model delegated có rounding giống hệt.
- Gate chất lượng: sAP10 development không thấp hơn FP32 quá 0,5 điểm; dữ liệu/tensor finite, đúng shape/dtype. Chọn trước khi chạy hồi quy 54 ảnh đã dùng, không điều chỉnh theo hồi quy.
- Chỉ model qua gate mới benchmark đầy đủ: CPU/LiteRT 2.2.0, 4/8 luồng, hai process session mỗi cấu hình, mỗi session 5 vòng đổi thứ tự, 16 warm-up + 16 timing/vòng. Cùng float application I/O contiguous; gồm adapter/copy, loại load/preprocess/line decoder. Không chạy experiment khác cùng timing.
- So cùng phiên với FP32/C1; tái tính median/p95 và bootstrap paired round, không coi 80 lượt là 80 mẫu độc lập. Candidate phải nhanh hơn C1 và FP32 8 luồng ở cả hai session mới được đề xuất, kèm độ bất định; không gọi hai process cùng host/ngày là bằng chứng across-device/day.
- Quy tắc chọn được ghi trước timing: lower bound khoảng paired-round 95% phải trên 0 so FP32 và C1 ở cả hai session 8 luồng. Chọn trung bình hai median thấp nhất; ưu tiên M1 nếu M2 không giảm thêm ít nhất 1% và có lower bound dương ở cả hai session. Khoảng bootstrap chỉ xấp xỉ từ 5 round, không phải chứng nhận triển khai.
- Phân rã application input/copy/invoke/output riêng sau benchmark. Nếu chưa có native profiler tương thích, chỉ báo đúng phạm vi I/O và delegate inventory, không gán thời gian từng kernel từ ORT hoặc danh sách node tĩnh.

C1 sử dụng Buffer.offset/size để trỏ dữ liệu ngoài phần FlatBuffer chính. Trước khi đóng gói graph mới phải đọc đầy đủ byte từ offset, chuyển thành Buffer.data và kiểm lại từng buffer; thao tác InitFromObj→Pack trực tiếp không giữ dữ liệu ngoài. C1_roundtrip phải qua kiểm output với C1 gốc trước khi đánh giá các candidate. Thay đổi đối chứng này được ghi trước inference/chất lượng/benchmark.

## ZipFormer

- Mục tiêu: giảm chi phí gọi model trên encoder FC ngày 21, decoder/joiner ngày 20. Không sửa graph, weight, state, frontend, context [0,0] hoặc greedy rule.
- Đối chứng Z0: SignatureRunner đã cache metadata. Candidate duy nhất Z1: public set_tensor/invoke/get_tensor với metadata cache, giải phóng SignatureRunner sau khi lấy mapping, giữ safety guard; không giữ tensor() view qua invoke. Chỉ dùng đúng subgraph 0 đã kiểm.
- Gate: 15 fixture encoder/36 output; decoder/joiner fixture nếu có; trên 173 câu cũ token và final counter khớp; ghi WER để đối chiếu, không gọi là holdout mới. Cùng model nên yêu cầu exact output cho wrapper trên cùng kernel/cấu hình, nếu không thì điều tra trước timing.
- Kiểm stream khoảng 10 phút ghép từ audio đã có, không reset state giữa các đoạn; so token/counter/output, kiểm finite. Đây là kiểm độ ổn định, không phải WER dữ liệu mới hoặc chứng minh counter INT32 không tràn ở mọi session.
- Sau gate, benchmark Z0/Z1 cùng LiteRT 2.1.2 CPU4threads, hai process session, mỗi session 5 vòng đổi thứ tự, component 16 warm-up + 32 mẫu/vòng. Pipeline mỗi vòng cùng ba smoke audio, warm-up đủ encoder/decoder/joiner cho cả hai; loại load/metadata, gồm frontend/read/greedy. Không so trực tiếp RTF mới với số ngày 21 có cách warm-up khác.
- Chỉ chọn Z1 khi token/state không đổi và lợi ích pipeline được lặp lại; chênh nhỏ còn bất định thì giữ Z0. Không mở thêm quantization/precision search ZipFormer trong ngày này.
- Trước timing chốt tiêu chí: median RTF thấp hơn và lower bound paired-round bootstrap 95% dương ở cả hai session mới chọn Z1 trên cấu hình đã đo; không suy sang host hoặc ngày khác.

## Đầu ra

REPORT.md chia theo project, Evidence đầu mục, trình bày kết quả, điều kiện đo, hạn chế và việc tiếp theo. RESEARCH.md giải thích giả thuyết/phương pháp; RUNNING.md ghi lệnh tái lập. JSON/tensor lưu model/input hash, cấu hình, raw sample và thời điểm khóa lựa chọn; giữ nguyên nguồn/kết quả ngày trước.
