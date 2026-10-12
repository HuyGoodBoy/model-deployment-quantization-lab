# Báo cáo 12/10

Người làm: Bùi Gia Huy

Hôm nay em làm tiếp phần post-processing M-LSD để chuẩn bị đưa lên Android.

1. Phần đã bổ sung

Em thêm Hough merge và cách chấm điểm theo source NAVER vào Java. Một cạnh có thể bị detect thành nhiều đoạn nhỏ nên cần gộp trước. Sau đó tìm giao điểm, lọc góc và nối bốn cạnh bằng ID đường chung để tạo box.

Đã đối chiếu Python và Java trên 12 bộ input, gồm 3 bộ lấy từ output model đã lưu. Đoạn sau gộp, góc và tập box khớp nhau; sai số các thành phần điểm lớn nhất khoảng 4,4e-16. Thử thêm 3 ca bật trọng số overlap/length cũng khớp. Bốn ca input rỗng/lỗi đều được Java xử lý an toàn.

2. Hướng thử thêm

Em thử lưu danh sách góc theo ID đường để chỉ xét những góc có thể nối nhau, thay vì duyệt cả nhóm rồi kiểm tra. Hai cách cho cùng kết quả trên các ca đã thử.

Median toàn bộ post-processing trên Windows, i7-12700H, OpenJDK 21, một luồng xử lý; warmup 1.000 lần, đo 400 lần mỗi cách:

| Input | Duyệt cả nhóm | Danh sách kề |
|---|---:|---:|
| Output model em convert | 0,0304 ms | 0,0330 ms |
| Output model chính thức | 0,0473 ms | 0,0446 ms |
| Lưới 8 ngang × 8 dọc | 0,1778 ms | 0,1647 ms |

Với lưới nhiều cạnh, số lần kiểm tra kết nối giảm từ 6.400 xuống 2.368; riêng bước ghép giảm từ 0,0262 xuống 0,0132 ms. Với input ít góc, chi phí dựng danh sách có thể làm chậm hơn. Vì vậy em chưa coi đây là cách luôn nhanh hơn.

3. Phần tiếp theo

Tích hợp decoder/camera và đo trên Android thật. Số trên chỉ đo hậu xử lý JVM, chưa gồm inference và chưa phải accuracy detect box trên dataset có nhãn.

Code và kết quả: [mlsd-postprocess](experiments/mlsd-postprocess/). Chi tiết kiểm tra: [verification.json](experiments/mlsd-postprocess/results/verification.json), [thời gian từng lượt](experiments/mlsd-postprocess/results/timings.csv). Cách chạy: [RUNNING.md](RUNNING.md). Phần giải thích: [ALGORITHM.md](experiments/mlsd-postprocess/ALGORITHM.md).

Kết quả rà lại các thí nghiệm trước đó được giữ ở [results/verification.json](results/verification.json).
