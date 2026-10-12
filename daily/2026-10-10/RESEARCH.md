# Giải thích thuật toán 10/10

Người làm: Bùi Gia Huy

File gốc: [utils.py của NAVER](https://github.com/navervision/mlsd/blob/453cafa09467d0272760578d35c1fda38e8895a5/utils.py), hai hàm pred_lines và pred_squares.

1. Lấy đoạn thẳng — pred_lines

Bản TFLite có decoder trả ra ba thứ:

- 200 điểm có thể là tâm đoạn thẳng.
- Score của từng điểm.
- Bản đồ độ dời từ mỗi điểm tới hai đầu đoạn.

Các bước:

1. Lấy tọa độ tâm cộng với độ dời để ra hai đầu mút. Ví dụ tâm (50,40), độ dời (-15,0) và (15,0) thì đoạn là (35,40)→(65,40).
2. Chỉ giữ đoạn có score > score_thr và độ dài > dist_thr. Mặc định của pred_lines là 0,1 và 20. pred_squares dùng ngưỡng score riêng là 0,06.
3. Nhân tọa độ với 2 vì output 256×256 nhỏ bằng nửa input 512×512, rồi nhân tỷ lệ để về kích thước ảnh gốc. Nếu ảnh có crop/letterbox thì cần đổi ngược cả phần đó.

2. Ghép box — pred_squares

Đầu vào cũng lấy đoạn thẳng từ tâm và độ dời như trên. Phần ghép box làm trên tọa độ bản đồ 256×256, cuối cùng mới đổi về ảnh gốc.

1. Gộp đoạn: một cạnh thật có thể bị đoán thành nhiều đoạn chồng nhau. Hough gom theo góc và khoảng cách tới gốc, rồi dựng một đoạn bao các đầu mút. Code chia ô khoảng cách 1 đơn vị, góc 2° và tìm cụm mạnh trong vùng 5×5.
2. Tìm giao điểm: giải hai phương trình đường thẳng cho mọi cặp đoạn, kéo dài đường nếu cần. Ví dụ x=20 và y=10 gặp nhau ở (20,10).
3. Lọc góc: giữ góc lớn hơn 60° và nhỏ hơn 120°, giao điểm phải gần đầu mút của cả hai đoạn. Hai đường kéo dài cắt nhau rất xa thì chưa đủ để coi là góc box. Sau đó chia góc thành trên-trái, trên-phải, dưới-phải, dưới-trái.
4. Ghép bốn cạnh: hai góc kề nhau phải có chung ID đường. Ví dụ góc trên-trái và trên-phải cùng dùng cạnh trên. Ghép đủ vòng 0→1→2→3→0 thì được một tứ giác.
5. Chấm điểm: ưu tiên tứ giác có góc đối gần bằng nhau, diện tích lớn và gần tâm ảnh. Code còn tính độ phủ và độ giống chiều dài cạnh, nhưng mặc định hai trọng số này bằng 0. Sắp xếp giảm dần để chọn box đứng đầu.
6. Đổi tọa độ: nhân 2 rồi đổi theo kích thước ảnh gốc như phần đoạn thẳng.

3. Viết lại để chạy Android

Chia làm hai phần:

- Trong .tflite: model và decoder gồm sigmoid, maxpool 3×3, top-k. Bản decoded_fp32 đã có phần này, trả points, scores, vmap.
- Trong Java/Kotlin: lấy hai đầu mút, lọc đoạn, gộp Hough, tìm góc, ghép và chấm điểm box.

Bản Java hiện có phần giao điểm và ghép box, nhưng còn thiếu Hough merge. Cách sinh box và scoring cũng khác NAVER nên chưa phải bản port đầy đủ.

Những chỗ cần chú ý:

1. Không có đoạn/góc/box thì trả danh sách rỗng.
2. Hough đang dùng TensorFlow maxpool/top-k và NumPy unique. Cần viết lại bằng mảng, vòng lặp hoặc HashMap.
3. NumPy chấp nhận chỉ số âm, Java thì không; phải kiểm tra biên khi duyệt vùng lân cận.
4. Bỏ đoạn dài bằng 0 và cặp gần song song trước khi chia. NAVER thêm epsilon rồi ép giao điểm về INT32, còn Java giữ tọa độ thực; cần so ảnh hưởng của khác biệt này.
5. Ghép bốn góc có nhiều vòng lặp lồng nhau, nhiều góc có thể làm chậm dù chỉ có 200 tâm ban đầu.
6. Tâm lưu (y,x), độ dời lưu (x,y). Camera có crop, padding, xoay hoặc lật thì phải đổi tọa độ tương ứng.

Cách kiểm tra: lưu output model và kết quả Python trên cùng vài ảnh, đưa đúng output đó cho Java rồi so đoạn sau gộp, góc, box và thứ hạng. Sau khi khớp mới đo trên Android.

4. Mixed INT8 trong DistilBERT

- Dynamic lấy khoảng activation khi chạy ở operator được hỗ trợ; static lấy trước bằng dữ liệu calibration.
- Mixed cho phép một số phần giữ float. Trong bài này mixed có calibration nên vẫn là static.
- Chọn mixed để converter có thể giữ float khi cần, không có nghĩa attention tự được giữ FP32. Bản strict yêu cầu bộ operator INT8, nhưng token ID/index vẫn có thể là INT32.

Ví dụ câu phim hay [PAD] [PAD]: mask dùng số âm rất lớn để phần padding gần như không được chú ý sau Softmax. Nếu vùng này bị quantize, số quá lớn có thể kéo khoảng giá trị rộng ra. INT8 chỉ có 256 mức nên khoảng rộng sẽ làm mỗi bước lớn hơn, dễ mất các khác biệt nhỏ.

Kết quả ngày 09/10 trên 100 mẫu: mixed và strict đều 52% accuracy. Giảm độ lớn mask xuống 1e4 được 56%, xuống 1e2 được 67%, còn FP32 là 91%. Giảm mask có giúp nhưng chưa giải quyết hết vấn đề; cần thử trên tập khác.

Code đối chiếu: [Keras/decoder](../2026-10-09/experiments/03-mlsd/src/port_keras.py), [Java](../2026-10-09/experiments/04-box-postprocess/src/BoxPostProcessor.java), [DistilBERT](../2026-10-09/experiments/02-distilbert-int8/src/run.py). Số liệu đầy đủ: [báo cáo 09/10](../2026-10-09/REPORT.md).
