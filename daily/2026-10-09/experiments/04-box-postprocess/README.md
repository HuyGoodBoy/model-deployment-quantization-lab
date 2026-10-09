# Detect box và hướng port Android

**Ngày:** 09/10/2026. **Trạng thái:** có hướng nghiên cứu, chưa port hoặc đo Android.

Nguồn đọc: [utils.py chính thức](https://github.com/navervision/mlsd/blob/master/utils.py). Phần giải thích cần Huy tự đọc, ghi lại theo ý hiểu và giải thích được các điều kiện hình học.

## Nội dung cần diễn giải

1. Từ center points, scores và displacement map khôi phục hai đầu mỗi đoạn thẳng.
2. Gộp các đoạn có biểu diễn đường gần nhau.
3. Tính giao điểm, xử lý đường song song và kiểm khoảng cách tới đoạn.
4. Phân loại góc hợp lệ, nối góc có cạnh chung thành chu trình bốn cạnh.
5. Chấm điểm ứng viên và chuyển tọa độ từ map về ảnh gốc.

## Hướng port

Tách hàm hình học nhận đoạn thẳng/điểm số và trả bốn đỉnh cùng score. Cân nhắc Kotlin cho bản đối chiếu đầu hoặc C++/JNI cho buffer và tính toán hình học. Đo post-processing độc lập với inference; giới hạn candidate và tái dùng bộ nhớ khi đã xác minh tính đúng.

Ca kiểm chứng cần có: không có đoạn, đường song song, đoạn độ dài 0, giao điểm ngoài đoạn, hình chữ nhật đã biết, nhiễu và tọa độ ảnh không vuông. Nếu cải tiến điều kiện/thuật toán so với upstream, ghi riêng là thay đổi, không gọi tương đương trước khi đối chiếu.

`src/` và `results/` được tạo khi triển khai. Chưa có code Android M-LSD; module `android/` hiện tại phục vụ MobileNetV2 FP32.

- [Task/tiêu chí](../../TASKS.md)
- [Báo cáo](../../REPORT.md)
- [Model M-LSD](../03-mlsd/README.md)
