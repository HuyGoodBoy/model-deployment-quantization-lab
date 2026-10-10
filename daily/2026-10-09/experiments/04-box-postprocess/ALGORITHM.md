# M-LSD: từ output model đến bốn góc

Tài liệu mô tả thuật toán post-processing M-LSD để phát hiện box và hướng port sang Android.

Nguồn đối chiếu: [`pred_lines` và `pred_squares`](https://github.com/navervision/mlsd/blob/453cafa09467d0272760578d35c1fda38e8895a5/utils.py). Revision được cố định; không coi prototype Java là bản port tương đương source này.

## 1. Decoder đoạn thẳng

Model Tiny nhận RGBA 512×512. Bản PyTorch trả bản đồ 256×256×9: channel 0 là center logit, channel 1–4 là `(dx_start, dy_start, dx_end, dy_end)`. Các channel còn lại không dùng cho phép decode đoạn thẳng này.

Áp sigmoid lên center, giữ cực đại trong cửa sổ **3×3**, lấy 200 vị trí mạnh nhất. Với center `(x,y)`, hai đầu mút là `(x+dx_start,y+dy_start)` và `(x+dx_end,y+dy_end)`. Loại center có score ≤0,1 hoặc độ dài displacement ≤20. Nhân tọa độ với 2 để trở lại ảnh 512. `points` có thứ tự **(y,x)**, displacement có thứ tự **(x,y)**; nhầm hai thứ tự này sẽ xoay/sai đoạn thẳng.

TFLite chính thức đã chứa sigmoid, NMS và top-k trong graph, trả `points`, `scores`, `vmap`. Vì vậy chỉ so latency graph raw PyTorch với graph chính thức sẽ khác phạm vi công việc.

## 2. Gộp các đoạn cùng đường

Một cạnh có thể được model dự đoán nhiều lần. Source mô tả đường `ax+by=c`, lấy khoảng cách đến gốc `d=abs(c)/sqrt(a²+b²)` và góc `theta`. Quantize Hough theo 1 đơn vị khoảng cách và 2 độ, đếm số đoạn trong mỗi ô. Source dùng max-pool **5×5 trên accumulator Hough** để chọn cụm; cửa sổ này khác NMS 3×3 của center heatmap.

Gom các endpoint quanh cụm, lấy biên x/y để tạo đoạn đại diện. Cách lấy min/max độc lập này là heuristic của source, không phải phép fit đường bằng least squares. Khi port cần quyết định giữ nguyên hành vi hay thay thuật toán, rồi kiểm chứng riêng.

## 3. Giao điểm và điều kiện góc

Với hai đường, định thức `D=a1*b2-a2*b1`. Khi |D| nhỏ, đường gần song song và giao điểm không đáng tin. Nếu D hợp lệ: `x=(c1*b2-c2*b1)/D`, `y=(a1*c2-a2*c1)/D`.

Source tính góc từ giao điểm đến tâm hai đoạn, giữ góc trong khoảng 60–120 độ. Điều kiện khoảng cách kiểm tra giao điểm nằm gần endpoint đến mức nào: phân biệt phần bên trong và ngoài đoạn với `inside_ratio`/`outside_ratio`. Không chỉ cần hai đường cắt nhau là tạo được góc box.

## 4. Ghép chu trình

Source phân loại góc thành bốn hướng 0,1,2,3. Góc liền kề phải chia sẻ cùng ID đường; ghép 0→1→2→3→0 tạo ứng viên bốn cạnh. Đây là tứ giác gần chữ nhật, không phải chỉ bounding box song song trục. Vòng lặp lồng nhau có thể tạo nhiều ứng viên; giới hạn số line và dùng danh sách kề giúp giảm chi phí trên điện thoại.

## 5. Chấm điểm

Source kết hợp độ phủ đoạn trên chu vi, tương đồng góc đối diện, diện tích chuẩn hóa, khoảng cách đến tâm ảnh và tương đồng độ dài cạnh đối diện. Tâm ảnh có trọng số trừ; các thành phần khác có trọng số cộng. Sau đó sắp xếp giảm dần. Prototype hiện tại chỉ dùng diện tích + mean confidence nên **chưa có thứ hạng tương đương** thuật toán NAVER.

## 6. Contract Android đề xuất

`ByteBuffer RGBA → LiteRT → points/scores/vmap → decoder → line merge → box geometry → float corners[4][2]`.

- Chạy phần Java/Kotlin với primitive arrays tái sử dụng, tránh tạo object trong từng frame. Phần lõi `detect` không dùng TensorFlow hoặc OpenCV.
- Giữ tọa độ float trong giao điểm, kiểm tra NaN/Inf, đoạn dài bằng 0 và định thức gần 0. Source có ép giao điểm sang INT32; đổi hành vi này cần phép so đối chứng.
- Nếu Java chưa đạt yêu cầu sau đo trên máy thật, port phần merge/intersection sang C++/JNI, dùng buffer trực tiếp; không mặc định JNI sẽ nhanh hơn.
- Khi camera dùng letterbox/crop, phải giữ phép biến đổi ngược. Không chỉ nhân 2 rồi vẽ trực tiếp lên ảnh camera khác kích thước.
- Đo riêng model, decoder, box và toàn pipeline trên **Android thật**; số đo JVM Windows không đại diện điện thoại.

## 7. Prototype và giới hạn đã kiểm tra

[`src/BoxPostProcessor.java`](src/BoxPostProcessor.java) nhận **đoạn thẳng đã gộp**; giới hạn 64 line, kiểm tra giao điểm/góc, chu trình bốn line, convex/area/bounds và trả top 20. Các ca kiểm tra: hình chữ nhật đã biết, song song, đoạn bằng 0, input rỗng và giao điểm ở quá xa.

CSV từ demo M-LSD hiện là đoạn thẳng **chưa qua Hough merge**. Đưa CSV đó vào Java chỉ kiểm tra nối pipeline và robustness, không chứng minh tương đương `pred_squares`.

Việc tiếp theo: viết Hough merge độc lập, lưu fixture từ source chính thức, so ID cạnh/góc/điểm/thứ hạng, rồi mới tích hợp Android. Huy tự thử bốn đoạn `(10,10)→(110,10)→(110,70)→(10,70)→(10,10)` và giải thích vì sao được một box diện tích 6.000.
