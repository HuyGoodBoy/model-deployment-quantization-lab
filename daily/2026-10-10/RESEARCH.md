# Giải thích thuật toán 10/10

Người làm: Bùi Gia Huy

## 1. Model tìm đoạn thẳng thế nào

Ví dụ ảnh chụp một tờ giấy. Model dự đoán tâm đoạn thẳng và độ lệch tới hai đầu đoạn.

Tâm ở `(50,40)`, đầu trái lệch `(-15,0)`, đầu phải lệch `(15,0)`. Cộng lại sẽ được hai đầu `(35,40)` và `(65,40)`.

Một vùng có thể có nhiều điểm tâm gần nhau. NMS giữ điểm có score cao nhất trong vùng 3×3. Sau đó lấy 200 điểm mạnh nhất rồi bỏ đoạn yếu hoặc quá ngắn. Trong `pred_lines`, ngưỡng là score > 0,1 và độ dài > 20. Score này được đưa về khoảng 0–1 bằng sigmoid, không phải cứ score cao là đúng.

Output nằm trên bản đồ 256×256, còn ảnh đưa vào là 512×512 nên phải nhân tọa độ với 2. Đoạn trên thành `(70,80)→(130,80)`. Một chỗ dễ nhầm là tâm lưu `(y,x)`, còn độ lệch lưu `(x,y)`.

## 2. Ghép các đoạn thành box

### Gộp đoạn

Một cạnh tờ giấy có thể bị model đoán thành nhiều đoạn chồng nhau. Nếu giữ hết thì dễ ra nhiều box trùng.

Hough gom đoạn theo hướng và khoảng cách tới gốc. Ví dụ `y=10` và `y=11` cùng nằm ngang, gần nhau nên có thể được gộp. Còn `x=10` nằm dọc nên không gộp chung chỉ vì cũng cách gốc 10 pixel.

Với đường `ax+by=c`, khoảng cách tới gốc là `d=|c|/sqrt(a²+b²)`. Code chia ô theo 1 đơn vị khoảng cách và 2° góc, tìm cụm mạnh trong vùng 5×5 rồi lấy các đầu mút để dựng đoạn đại diện.

### Tìm góc

Đường trên `y=10` và đường trái `x=20` gặp nhau ở `(20,10)`. Đây là cách lấy một góc.

Với đường bất kỳ, giải hai phương trình đường thẳng. Công thức dùng:

```text
D = a1*b2 - a2*b1
x = (c1*b2 - c2*b1)/D
y = (a1*c2 - a2*c1)/D
```

Hai đường song song thì D bằng 0. Gần song song cũng dễ làm giao điểm chạy rất xa. Bản Java loại những cặp này trước khi chia; NAVER thêm `1e-10` vào D rồi ép tọa độ về INT32. Hai cách này cần so lại khi port.

Có giao điểm rồi vẫn phải lọc. Ví dụ cạnh trên chỉ dài tới `x=80`, còn đường kia ở `x=500`: kéo dài thì cắt nhau, nhưng không coi ngay đó là góc tờ giấy. Code kiểm tra giao điểm có gần đầu đoạn không, đồng thời giữ góc lớn hơn 60° và nhỏ hơn 120°.

### Ghép bốn góc và chọn box

```text
A ------ cạnh trên ------ B
|                         |
cạnh trái             cạnh phải
|                         |
D ------ cạnh dưới ------ C
```

A và B nối được vì cùng dùng ID cạnh trên. Làm tiếp B→C→D→A để khép kín bốn cạnh. Code chia góc thành bốn hướng để tìm vòng này. Box có thể xoay, không cần bốn cạnh bằng nhau.

Nếu có nhiều box thì chấm điểm theo diện tích, góc, độ phủ cạnh, độ dài cạnh và vị trí trong ảnh. Mỗi phần có trọng số riêng; mặc định phần độ phủ và độ dài có trọng số 0 nên không góp điểm.

Bản Java hiện còn thiếu Hough merge, cách ghép và chấm điểm cũng khác NAVER. Nó chạy ra box nhưng chưa phải bản port đầy đủ.

## 3. Mixed INT8 và mask của DistilBERT

INT8 có 256 mức. Chia khoảng 0–1 thì mỗi bước khoảng 0,0039; chia khoảng 0–100 thì mỗi bước thành 0,392. Khoảng rộng hơn sẽ khó giữ những khác biệt nhỏ. Đây là ví dụ để hiểu sai số làm tròn.

- Dynamic: tính khoảng activation khi chạy ở operator được hỗ trợ.
- Static: dùng dữ liệu calibration để lấy khoảng trước, rồi cố định khi chạy.
- Mixed: cho phép có phần giữ float. Mixed trong bài này có calibration nên vẫn là static.

Chọn mixed không có nghĩa attention tự được giữ FP32. Bản strict chỉ yêu cầu bộ operator INT8, nhưng token ID/index vẫn có thể là INT32.

Ví dụ câu `phim hay [PAD] [PAD]`: hai `[PAD]` chỉ để đủ chiều dài. Mask cộng số âm rất lớn vào điểm attention của chúng để sau Softmax, chúng gần như bị bỏ qua.

Nếu phần chứa số này bị quantize, khoảng giá trị có thể bị kéo quá rộng. Khi đó các điểm nhỏ dễ bị làm tròn thành giống nhau.

Kết quả ngày 09/10 trên 100 mẫu:

| Bản TFLite | Accuracy | SNR (dB) |
| --- | --- | --- |
| Mixed ban đầu | 52% | 0.036 |
| Strict | 52% | 0.036 |
| Mixed, độ lớn mask 1e4 | 56% | 0.013 |
| Mixed, độ lớn mask 1e2 | 67% | 0.835 |

Giảm mask giúp INT8 tốt hơn, nhưng vẫn thấp hơn FP32 91%. Hai bản FP32 đổi mask vẫn cho output giống bản gốc trên tập này. Vậy mask là chỗ cần xem tiếp, chưa thể nói nó là nguyên nhân duy nhất.

## 4. Chạy trên Android

Luồng sẽ là: ảnh camera → model → đoạn thẳng → gộp đoạn → ghép box → vẽ lên ảnh. Phần ghép có thể viết Java/Kotlin và dùng lại mảng qua từng frame.

Cần chú ý tọa độ. Ví dụ ảnh 640×480 thu xuống 512×384, rồi thêm 64 pixel ở trên và dưới. Góc `(80,104)` trên ảnh 512×512 phải đổi lại:

```text
x = 80/0,8 = 100
y = (104-64)/0,8 = 50
```

Nếu lấy điểm từ bản đồ 256×256 thì nhân 2 trước bước trên. Camera xoay/lật cũng phải đổi theo.

Tiếp theo cần làm nốt Hough merge và scoring, so đoạn/góc/box với NAVER bằng cùng đầu vào rồi đo trên điện thoại. Chưa có số đo Android mới.

Nguồn: [NAVER utils.py](https://github.com/navervision/mlsd/blob/453cafa09467d0272760578d35c1fda38e8895a5/utils.py), [bản Keras](../2026-10-09/experiments/03-mlsd/src/port_keras.py), [bản Java](../2026-10-09/experiments/04-box-postprocess/src/BoxPostProcessor.java), [code DistilBERT](../2026-10-09/experiments/02-distilbert-int8/src/run.py), [kết quả 09/10](../2026-10-09/REPORT.md).
