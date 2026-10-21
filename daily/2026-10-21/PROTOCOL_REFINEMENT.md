# Bổ sung thực nghiệm sau phản biện — 21/10/2026

Người dùng đã duyệt phương án sau debate với Astra; chỉ làm local, **không commit/push**. Giữ nguyên kết quả lượt đầu và toàn bộ file ngày 20. Kế hoạch này được ghi trước khi tạo/chạy các candidate mới.

## M-LSD

- Giữ cùng checkpoint, 32 calibration, 16 development và decoder. Hồi quy 54 ảnh đã được dùng trước đây, không phải test độc lập.
- Prune Conv cuối còn đúng filter 7..11, một Conv 5 kênh và một output; kiểm control FP32 với baseline trên 16 ảnh trước PTQ. Đây là đối chứng khác với hai nhánh INT8 cần scale riêng.
- Tối đa ba recipe PTQ mới: C1 giữ toàn head float; C2 chỉ Conv cuối và phần output sau nó float; nếu C1 qua và C2 trượt thì C3 giữ suffix bắt đầu từ block16 float. Đo và ghi dtype thực, không suy precision từ tên recipe.
- Gate sàng lọc: sAP10 development không thấp hơn FP32 quá 0,5 điểm. Không gọi gate này là kiểm định non-inferiority. Candidate trượt không chạy benchmark/hồi quy đầy đủ. Không chọn theo hồi quy.
- Candidate qua gate mới đo matched 4/8 luồng với FP32 và A hiện có; timing gồm adapter ứng dụng. Chạy hai session/process cho mỗi cấu hình; đề xuất thay FP32 chỉ khi median 8 luồng thấp hơn matched FP32 ở cả hai session. Đây vẫn là sàng lọc bằng point median trên cùng máy/ngày, không chứng minh tăng tốc tổng quát hoặc có ý nghĩa thống kê.
- Profile riêng execution plan/delegate và chi phí application I/O/Q-DQ của FP32/A/candidate. Dùng profiler thật nếu runtime cung cấp; phân biệt phân rã wall-clock, node profile và chế độ không delegate. Không quy nguyên nhân cho E-core hoặc partitioning khi chưa có evidence. Không thay precision backbone dựa duy nhất vào L2 tích lũy.

Làm rõ contract trước phép đo cuối: hai lượt exploratory `refinement_benchmark_t4_s1`/`t8_s1` cho phép C1 trả NCHW view trong khi FP32/A trả buffer contiguous. NCHW view hợp lệ, nhưng để kiểm soát cùng buffer layout và không đẩy copy sang decoder bị loại khỏi timing, phép đo cuối áp dụng `np.ascontiguousarray` cho **tất cả** biến thể, kiểm output exact/C-contiguous và chạy hai session mới có tiền tố `refinement_contiguous_`. Không lấy exploratory để chọn model; không tuyên bố contiguous là yêu cầu deployment đã được xác nhận.

## ZipFormer

- Bắt đầu từ encoder FC đã kiểm chứng, chỉ fuse FC→ADD bias hằng nếu output FC có một consumer, không phải graph output, dtype/shape/broadcast và activation đều hợp lệ. Không mặc định cả 112 node đều fuse được. Fusion vẫn có thể đổi vị trí rounding FP32; không sửa giá trị bias/weight và không hứa bit-exact.
- Giữ nguyên public I/O, state, decoder, joiner, frontend. Kiểm tensor/state trên 15 fixture; kiểm lại WER/token trên 73 và 100 câu đã có, cùng stream liên tục 79,5 giây. Hai tập đã biết kết quả trước patch mới nên là regression, không gọi fresh holdout.
- Đo matched baseline/FC/FC+bias cùng phiên, tách encoder và RTF. Chỉ giữ bản mới nếu parity qua và có lợi ích được đo; nếu không, giữ FC cũ và báo kết quả âm.
- Bổ sung init/allocation/first invoke và RSS trong process mới cho từng model; không chạy hai model chung process đo peak. Phân biệt RSS process và dung lượng file; không khẳng định prepacking là nguyên nhân tốc độ chỉ từ rewrite.
- Kiểm payload buffer để giải thích chênh 936 byte của rewrite cũ; không quy riêng overhead cho options/bias nếu chưa tách contribution.

## Báo cáo và kiểm chứng

- REPORT.md chi tiết, chia ZipFormer/M-LSD, mỗi mục Evidence đầu tiên, sau đó trạng thái quyết định. Báo cáo chat ngắn; thông tin công cụ ở cuối bản chi tiết.
- Lưu JSON/tensor/command/environment/hash và lý do skip. Giữ nguyên JSON cũ; kết quả mới có tiền tố `refinement_`. Kiểm lại metric từ output/raw timing, hash nguồn cũ, lint và link Markdown; cập nhật snapshot cuối.
