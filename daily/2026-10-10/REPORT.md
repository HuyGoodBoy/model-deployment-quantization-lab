# Báo cáo 10/10

Người làm: Bùi Gia Huy

Hôm nay chủ yếu đọc thuật toán M-LSD và xem lại phần quantization của DistilBERT, chưa chạy thêm thí nghiệm.

## 1. M-LSD ghép đoạn thẳng thành box thế nào

Model trả tâm đoạn thẳng và độ lệch tới hai đầu mút. Lấy tâm cộng độ lệch sẽ được đoạn thẳng. Sau đó lọc các đoạn có score thấp hoặc quá ngắn.

Phần ghép box trong [`utils.py` của NAVER](https://github.com/navervision/mlsd/blob/453cafa09467d0272760578d35c1fda38e8895a5/utils.py) làm tiếp:

- **Gộp đoạn:** một cạnh có thể bị dự đoán nhiều lần. Source gom theo góc và khoảng cách tới gốc `d = |c|/sqrt(a²+b²)`, với đường `ax+by=c`.
- **Tìm góc:** giải giao điểm của hai đường, giữ góc 60–120° và giao điểm gần đầu đoạn. Hai đường kéo dài cắt nhau ở xa thì không coi là góc box.
- **Ghép bốn cạnh:** hai góc kề nhau phải có chung ID đường. Nối đủ bốn góc thành vòng kín mới tạo được box.
- **Chọn box:** chấm điểm theo diện tích, góc, độ phủ cạnh và vị trí trong ảnh. Một số thành phần có trọng số mặc định bằng 0 nên không ảnh hưởng điểm.

Bản Java hiện còn thiếu bước gộp Hough và dùng cách chấm điểm đơn giản hơn. Vì vậy chạy ra box chưa có nghĩa là đã port đúng thuật toán gốc.

## 2. Mixed INT8 trong DistilBERT

Mixed INT8 có calibration vẫn là static. Khác biệt là mixed cho phép giữ phần float khi cần, còn bản strict chỉ yêu cầu bộ operator INT8. Cho phép float không có nghĩa converter tự giữ attention ở FP32.

Mask dùng số âm rất lớn để bỏ qua padding trước Softmax. Nếu vùng này bị quantize, khoảng giá trị quá rộng làm mỗi bước INT8 lớn hơn, các giá trị nhỏ dễ bị làm tròn mất.

Kết quả hôm trước: mixed và strict đều đạt 52% accuracy. Giảm độ lớn mask từ `1e30` xuống `1e2` được 67%, vẫn thấp hơn FP32 91%. Như vậy sửa mask có giúp, nhưng chưa giải quyết hết vấn đề.

## 3. Việc làm tiếp

- Bổ sung Hough merge và scoring vào Java, so từng bước với source NAVER rồi mới tích hợp Android.
- Kiểm tra đổi tọa độ từ model về ảnh camera, nhất là khi có crop/letterbox.
- Với DistilBERT, thử giữ phần nhạy cảm ở float và đánh giá lại trên tập độc lập.

Công thức và ví dụ chi tiết: [RESEARCH.md](RESEARCH.md).
