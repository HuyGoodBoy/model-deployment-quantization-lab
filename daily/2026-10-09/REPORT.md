# Báo cáo 09/10

Người làm: Bùi Gia Huy

Em chạy trên CPU i7-12700H, RAM 16 GB. Phần reproduce dùng Linux trong Docker/WSL2; các phép đo trước đó chạy trên Windows. Chưa chạy Android.

## 1. Chạy lại code của Đức và so với môi trường của em

### Code của Đức

Em chạy lại ResNet18 với 100 ảnh calibration và 300 ảnh đánh giá, giữ nguyên source Đức. q1–q3 chạy nguyên script; q4 giữ hàm đánh giá và benchmark, tách từng biến thể để lưu kết quả riêng. Đã chạy được 6/6 biến thể.

Cả hai dùng 4 thread, batch 1, warm-up 10 lần rồi đo 100 lần lấy median. Cột Mac là số Đức đã báo cáo, cột Linux là số em chạy lại.

| Biến thể | Đức báo trên Mac (ms) | Em đo trên Linux (ms) | Accuracy Linux | Agree với PyTorch | SNR (dB) |
| --- | --- | --- | --- | --- | --- |
| ONNX FP32 | 14.90 | 10.177 | 64.67% | 100.00% | 120.95 |
| ONNX dynamic | 9.50 | 40.724 | 65.67% | 93.33% | 20.51 |
| ONNX static | 3.70 | 10.999 | 67.33% | 96.33% | 24.59 |
| TFLite FP32 | 15.30 | 11.920 | 64.67% | 100.00% | 117.27 |
| TFLite dynamic | 4.70 | 5.122 | 67.33% | 95.33% | 27.72 |
| TFLite static | 4.30 | 4.940 | 61.00% | 85.00% | 20.87 |

Agree là tỷ lệ dự đoán giống PyTorch gốc, khác với accuracy là tỷ lệ đúng nhãn. Chất lượng output nhìn chung gần kết quả Đức, nhưng tốc độ thay đổi khá nhiều, rõ nhất ở ONNX dynamic.

Em profile riêng ONNX dynamic trên Linux: `ConvInteger` chiếm 93,3% thời gian các node, còn `DynamicQuantizeLinear` chỉ 1,3%. Như vậy phần chậm chính trong lần chạy này là kernel convolution INT8, không phải bước tính range. Chưa có profile trên Mac để khẳng định nguyên nhân của toàn bộ chênh lệch.

Em cũng kiểm tra TFLite static: output chỉ biểu diễn được tới 14,58 trong khi logit PyTorch lên tới 32,52, nên có clipping. Đây là kết quả kiểm tra range; em chưa làm thí nghiệm thay EMA bằng min/max như phần của Đức.

### Model của em trên môi trường đó

Em chạy lại ResNet50 và DistilBERT, giữ nguyên file model và input. Nhóm này dùng 4 thread, warm-up 30 lần, đo 200 lần; accuracy đo trên 100 mẫu có nhãn.

- Windows cũ: ORT 1.20.1 / TensorFlow Lite 2.15.1.
- Windows mới và Linux: ORT 1.30.0 / LiteRT 2.2.0.

| Model / biến thể | Windows cũ (ms) | Windows mới (ms) | Linux (ms) | Accuracy Linux | SNR Linux (dB) |
| --- | --- | --- | --- | --- | --- |
| ResNet50 / onnx dynamic | 30.608 | 25.465 | 22.822 | 73% | 40.34 |
| ResNet50 / onnx fp32 | 32.768 | 27.234 | 22.602 | 73% | 112.40 |
| ResNet50 / onnx static | 29.831 | 32.544 | 23.621 | 70% | 18.89 |
| ResNet50 / tflite dynamic | 163.850 | 15.669 | 11.845 | 73% | 26.58 |
| ResNet50 / tflite fp32 | 57.345 | 93.937 | 25.332 | 73% | 114.10 |
| ResNet50 / tflite static | 24.907 | 14.906 | 10.169 | 74% | 20.58 |
| DistilBERT / onnx dynamic | 11.658 | 11.753 | 9.324 | 91% | 25.01 |
| DistilBERT / onnx fp32 | 24.567 | 21.528 | 21.421 | 91% | 128.33 |
| DistilBERT / onnx static | 95.143 | 94.119 | 31.624 | 92% | 10.17 |
| DistilBERT / tflite dynamic | 1054.815 | 39.582 | 7.550 | 90% | 21.27 |
| DistilBERT / tflite fp32 | 163.676 | 36.111 | 20.993 | 91% | 124.53 |
| DistilBERT / tflite static | 114.788 | 86.863 | 24.954 | 48% | 0.03 |

TFLite dynamic thay đổi nhiều nhất. Ví dụ DistilBERT từ 1054,815 ms ở Windows cũ xuống 39,582 ms ở Windows mới và 7,550 ms trên Linux. XNNPACK ở các bản mới có thêm hỗ trợ dynamic-range Conv2D/Fully Connected, nên có thể góp phần vào chênh lệch này; em chưa profile đủ để quy toàn bộ thay đổi cho XNNPACK.

Cùng máy nhưng khác OS, bản thư viện và native kernel vẫn có thể chạy khác nhau. Ngoài ra, giữa các phiên đo có biến động latency: DistilBERT static từng khoảng 43 ms, nhưng kiểm tra lại được 123–128 ms; đổi cách đặt thread chưa giải thích được chênh lệch. Vì vậy các số trên là kết quả từng phiên, cần đo lặp và kiểm soát nhiệt độ/power trước khi kết luận chắc về tốc độ.

Chi tiết môi trường, sai số và profile: [PEER-REPRODUCTION.md](PEER-REPRODUCTION.md).

## 2. Dynamic/static của ONNX Runtime và TFLite khác nhau thế nào

Quantization ánh xạ số thực sang số nguyên theo `x ≈ scale × (q − zero_point)`. Range càng rộng thì mỗi bước INT8 càng lớn, sai số làm tròn tăng; range quá hẹp thì giá trị ngoài khoảng bị kẹp.

| Phương pháp | ONNX Runtime | TFLite/LiteRT |
| --- | --- | --- |
| Dynamic | Weight quantize trước; range activation ở operator hỗ trợ được tính khi chạy | Weight thường lưu INT8; kernel hybrid có thể quantize activation khi chạy, các phần khác vẫn float |
| Static | Lấy range từ calibration trước khi chạy | Cũng dùng calibration; có thể cho phép float fallback hoặc yêu cầu các operator INT8 |

Cùng tên dynamic/static chưa có nghĩa là hai model làm giống nhau. Còn phải xem operator nào được quantize, weight dùng một scale cho cả tensor hay từng kênh, cách gộp range calibration và runtime có kernel tối ưu hay không.

Trong bài DistilBERT của em, ONNX static chỉ chọn MatMul/Gemm; phần mask và Softmax nằm ngoài phạm vi đó. TFLite static quantize rộng hơn, có cả vùng liên quan đến mask với scale rất lớn. Vì thế không thể chỉ nhìn nhãn “static INT8” rồi cho rằng hai bên chịu cùng sai số.

Đức dùng AI Edge Quantizer, còn em dùng TensorFlow converter 2.15.1. Kết quả về EMA trong bài Đức cần gắn với đúng thư viện/cấu hình đó, không áp dụng chung cho mọi cách convert TFLite. Calibration không đại diện dữ liệu chạy thật có thể gây sai ở cả ONNX lẫn TFLite.

## 3. DistilBERT: calibrated mixed INT8 và static INT8

**Mixed INT8 ở đây vẫn là static quantization.** “Calibrated” nghĩa là đã dùng dữ liệu calibration để lấy scale; “mixed” nghĩa là cho phép một số phần giữ float. Hai cách gọi nói về hai việc khác nhau: lúc lấy scale và phần nào được chạy INT8.

Ban đầu em chọn mixed để converter có thể giữ float ở phần chưa hỗ trợ INT8. Tuy nhiên, kiểm tra model cho thấy không phải toàn bộ attention hay LayerNorm được giữ FP32; tensor float còn lại là phần CAST của mask.

Em thử thêm bản strict, chỉ cho phép bộ operator INT8. Bản này convert được, có 372 tensor INT8, 126 INT32 và 1 FLOAT32 ở CAST→QUANTIZE của mask. Cả mixed và strict đều đạt 52% accuracy. Như vậy, hiện chưa có bằng chứng mixed giữ chất lượng tốt hơn strict trong model này. INT32 của token ID/index cũng không phải float fallback.

### Thử giảm giá trị rất lớn trong attention mask

Attention mask dùng một số âm rất lớn để vị trí padding có xác suất gần 0 sau Softmax. Giá trị cỡ `1e30` có thể kéo range quá rộng, làm các giá trị nhỏ hơn khó phân biệt khi đưa về INT8.

Em giữ nguyên weights, input và calibration, chỉ đổi độ lớn giá trị mask từ `1e30` sang `1e4` hoặc `1e2`. Trước khi quantize, hai bản FP32 mới cho output giống hệt reference trên 100 mẫu, accuracy vẫn 91%.

| Biến thể | Accuracy | SNR (dB) | Median (ms) | Scale lớn nhất |
| --- | --- | --- | --- | --- |
| Mixed ban đầu | 52% | 0.036 | 42.871 | 3.92e+27 |
| Mixed, mask 1e2 | 67% | 0.835 | 64.266 | 2.67e+03 |
| Mixed, mask 1e4 | 56% | 0.013 | 42.415 | 2.67e+03 |
| Strict INT8 | 52% | 0.036 | 43.276 | 3.92e+27 |

Đổi mask sang `1e2` giúp accuracy từ 52% lên 67%, nhưng vẫn thấp hơn FP32 91%. Kết quả ủng hộ việc range của mask quá lớn là một phần vấn đề; chưa chứng minh đây là nguyên nhân duy nhất. Bước tiếp theo là kiểm tra trên tập độc lập và thử giữ những phần nhạy cảm ở float.

Code và kết quả: [02-distilbert-int8](experiments/02-distilbert-int8/).

## 4. Convert M-LSD PyTorch sang TFLite và thử quantize

### Cách convert và kiểm tra output

Em chuyển M-LSD Tiny 512 từ PyTorch sang Keras rồi convert TFLite, dùng cùng checkpoint, không train lại. Khi chuyển cần đổi thứ tự chiều weight convolution, xử lý depthwise convolution, padding và resize bilinear cho khớp PyTorch.

Input là RGBA 512×512, alpha bằng 1, normalize bằng `x/127.5 − 1`. Output thô được đưa về cùng thứ tự NHWC để so.

Bản Keras so với PyTorch đạt SNR 96.931 dB, relative L2 0.00001424, max error 0.029602. Kiểm tra `allclose` với ngưỡng `1e-4` ban đầu không đạt. Bản chuyển đổi đạt ngưỡng nới hơn là relative L2 < `1e-4` và max error < `0.1`; đây mới là kiểm tra sai số số học, chưa phải đánh giá chất lượng detect line/box.

### Tốc độ so với TFLite chính thức

Bản chính thức có sẵn decoder, trả `points`, `scores`, `vmap`. Em thêm decoder vào bản convert rồi mới so tốc độ, cùng 4 thread, warm-up 30 lần và đo 200 lần:

| Runtime | Bản em convert, có decoder (ms) | TFLite chính thức (ms) |
| --- | --- | --- |
| 2.15.1 | 51.183 | 105.265 |
| 2.2.0 | 107.193 | 110.938 |

Ở TF Lite 2.15.1, bản em convert nhanh hơn khoảng 2 lần trong phiên đo này; ở LiteRT 2.2.0 thì hai bản gần nhau. Hai model không dùng cùng checkpoint/graph, nên bảng này so tốc độ, không dùng để kết luận chất lượng tương đương.

### Quantize phần model trả output thô

Em dùng 32 ảnh train để calibration, 10 ảnh validation để so output với cùng checkpoint PyTorch. Nhóm này chưa có decoder trong graph nên không so trực tiếp latency với bảng trên.

| Biến thể | Dung lượng (MiB) | Median (ms) | SNR so với PyTorch (dB) | Max error |
| --- | --- | --- | --- | --- |
| dynamic | 0.697 | 668.929 | 12.171 | 181.86096 |
| fp16 | 1.214 | 105.074 | 20.937 | 142.63797 |
| fp32 | 2.372 | 102.851 | 88.307 | 0.08914 |
| static | 0.756 | 125.828 | 3.158 | 507.82515 |

FP32 giữ output gần PyTorch nhất. FP16 và INT8 làm file nhỏ hơn nhưng sai số tăng; dynamic và static chưa đem lại lợi thế tốc độ trong môi trường đã đo.

Output chứa cả center logit và displacement. Bản static dùng chung output scale khoảng 5,646; bước này có thể quá lớn với center. SNR center là 11,052 dB, displacement là 3,121 dB. Em cần thử tách hai head hoặc quantize chọn lọc để kiểm tra giả thuyết này. Chưa có dataset nhãn line/box, nên chưa báo precision/recall hay IoU; số đoạn detect được chỉ dùng kiểm tra pipeline.

Code và kết quả: [03-mlsd](experiments/03-mlsd/).

## 5. Thuật toán detect box và hướng chạy Android

Model dự đoán đoạn thẳng; phần post-processing mới ghép chúng thành box. Luồng chính gồm:

1. **Decode đoạn thẳng:** áp sigmoid lên center heatmap, giữ cực đại trong cửa sổ 3×3 và lấy top 200. Từ center `(x,y)`, cộng displacement để có hai đầu mút. Loại dự đoán yếu/ngắn, rồi nhân tọa độ với 2 để về ảnh 512. `points` là `(y,x)`, displacement là `(x,y)` nên phải đổi đúng thứ tự.
2. **Gộp đoạn cùng đường:** một cạnh có thể bị dự đoán thành nhiều đoạn. Biểu diễn đường bằng `ax + by = c`, lấy khoảng cách tới gốc `d = |c| / sqrt(a²+b²)` và góc để gom vào ô Hough. Các đoạn có khoảng cách/góc gần nhau được gộp thành một đoạn đại diện. Source dùng suppression 5×5 trên bảng Hough, khác NMS 3×3 ở bước center.
3. **Tìm góc:** với hai đường, tính `D = a1*b2 − a2*b1`. Nếu D gần 0 thì hai đường gần song song, bỏ cặp đó. Ngược lại, tính giao điểm `x = (c1*b2 − c2*b1)/D`, `y = (a1*c2 − a2*c1)/D`. Chỉ giữ góc khoảng 60–120° và giao điểm đủ gần đầu đoạn, tránh lấy giao điểm của hai đường kéo dài quá xa.
4. **Ghép bốn cạnh:** chia góc thành bốn hướng rồi tìm chu trình 0→1→2→3→0. Hai góc kề nhau phải dùng chung ID đường để bảo đảm chúng nối bằng cùng một cạnh. Kết quả là tứ giác gần chữ nhật, có thể bị xoay.
5. **Chấm điểm:** kết hợp mức phủ cạnh, độ giống góc/cạnh đối diện, diện tích và khoảng cách tới tâm ảnh, rồi xếp hạng ứng viên.

Em đã viết prototype Java cho phần giao điểm và ghép chu trình, qua 5 ca kiểm tra hình học. Demo từ 18 đoạn chưa gộp cho 4 box ứng viên, median 0,056 ms trên JVM Windows. Prototype chưa có Hough merge và scoring đầy đủ như NAVER, số này cũng chưa phải tốc độ Android.

Hướng port là dùng Java/Kotlin với mảng số và buffer tái sử dụng; giữ tọa độ float khi tính giao điểm, xử lý đường song song/đoạn bằng 0. Sau khi hoàn thiện merge và scoring, đối chiếu kết quả với source chính thức rồi tích hợp LiteRT. Khi vẽ lên camera cần đổi ngược tọa độ crop/letterbox. Nếu đo trên điện thoại thấy phần này chậm mới cân nhắc C++/JNI.

Chi tiết: [ALGORITHM.md](experiments/04-box-postprocess/ALGORITHM.md), [prototype Java](experiments/04-box-postprocess/src/BoxPostProcessor.java).

## 6. Việc làm tiếp

- DistilBERT: kiểm tra lại mask trên dữ liệu độc lập, thử quantize chọn lọc để tránh giảm accuracy quá nhiều.
- M-LSD: dùng calibration gần ảnh đường thẳng hơn, thử tách center/displacement và đánh giá trên dữ liệu có nhãn.
- Hoàn thiện Hough merge/scoring, rồi đo riêng model, post-processing và toàn pipeline trên Android thật.

[Cách chạy lại](RUNNING.md) · [Môi trường](ENVIRONMENT.md) · [Bài học HTML](../../docs/2026-10-09/hoc-ngay-2026-10-09.html). JSON/CSV chi tiết nằm trong từng thư mục thí nghiệm.
