# M-LSD: phần Java bổ sung ngày 12/10

Nguồn đối chiếu: [pred_squares của NAVER](https://github.com/navervision/mlsd/blob/453cafa09467d0272760578d35c1fda38e8895a5/utils.py).

1. Gộp đoạn

Một cạnh trên của tờ giấy có thể bị detect thành ba đoạn. Hough gom theo vị trí và góc của đường, thay vì đòi ba đoạn phải chạm nhau. Bản Java dùng ô khoảng cách 1, góc 2 độ và cửa sổ 5×5 như source. Mỗi cụm lấy các đầu mút ngoài cùng để tạo đoạn đại diện.

2. Tìm góc và ghép box

Kéo dài hai đoạn để tìm giao điểm, kiểm tra góc 60–120 độ và giao điểm có gần đầu mút không. Góc còn giữ ID của hai đường tạo ra nó. Hai góc liền nhau phải có chung một ID đường, nối đủ bốn cạnh rồi quay về góc đầu mới thành ứng viên box.

Ví dụ A có ID (L3,L0), B có (L0,L1): cạnh A–B dùng L0. Nếu B đổi thành (L4,L1) thì cạnh này không nối được dù tọa độ B trông vẫn hợp lý.

3. Cách thử mới: danh sách kề

Lưu sẵn mỗi đường đi qua những góc nào. Khi đang ở A, chỉ tìm trong các góc có chung đường với A. Cách cũ vẫn duyệt cả nhóm góc rồi mới kiểm tra ID. Hai cách phải cho cùng box và điểm; danh sách kề cần thêm thời gian dựng, nên ít góc chưa chắc nhanh hơn.

4. Chấm điểm và kiểm chứng

Đã đưa đủ các thành phần overlap, góc đối, diện tích, lệch tâm và độ dài cạnh đối vào Java. Mặc định overlap và length có trọng số 0; thử thêm ba ca bật hai trọng số này để tránh bỏ sót lỗi.

Bản ngày 12 giữ epsilon và ép INT32 tại giao điểm để đối chiếu NAVER, khác prototype ngày 09 giữ double. Java chủ động bỏ cặp gần song song, NaN/Inf, đoạn bằng 0 và chỉ số ngoài bảng Hough. Đây là thay đổi để xử lý input lỗi, không cam kết giống source ở mọi trường hợp biên.

Lõi nhận đoạn đã decode trên map 256×256. Phần decoder tensor và camera Android chưa tích hợp trong thí nghiệm này. Kết quả kiểm chứng chỉ nói hai bản khớp trên fixture đã thử, không phải accuracy detect box trên dataset có nhãn.
