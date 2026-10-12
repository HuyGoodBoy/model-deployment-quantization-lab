# Chạy lại thí nghiệm 12/10

Chạy từ root repo. Dùng môi trường baseline đã có; không cần tải lại model hay chạy inference.

```powershell
python scripts/setup.py --profile baseline
.venv/Scripts/python.exe daily/2026-10-12/experiments/mlsd-postprocess/verify.py
```

Cần JDK 11 trở lên trong JAVA_HOME hoặc PATH. Script cũng nhận JDK đã tải ở .cache/android-tools/java. Code Java được compile với --release 8; chỉ class MlsdPostProcessor là phần lõi để đưa vào Android. PostprocessBench dùng đọc/ghi file trên desktop.

Script kiểm tra hash của fixture và source NAVER revision 453cafa09467d0272760578d35c1fda38e8895a5. Source được tải vào cache ở lần chạy đầu. Phần sau ranh giới decode đoạn của pred_squares được giữ nguyên để làm đối chứng Python; chỉ đổi return thành locals để lấy thêm kết quả trung gian.

Hai bên nhận cùng đoạn trong tọa độ map 256×256, lọc score > 0,06 và độ dài > 20 trước đó. Các fixture nhỏ đã nằm trong Git. Muốn tạo lại ba fixture từ output model ngày 09, thêm --refresh-fixtures; lúc đó cần các file demo_*.npz cũ trên máy.

Mặc định warmup 1.000 lần và đo 400 lần cho mỗi cách ghép. Hai cách chạy xen kẽ, đổi thứ tự trước/sau ở mỗi lượt. Kết quả gồm Hough, tìm góc, ghép và chấm điểm; không gồm inference, đọc CSV hay đổi tọa độ về ảnh gốc.

- experiments/mlsd-postprocess/results/verification.json: đối chiếu kết quả, môi trường, hash và median/p95.
- experiments/mlsd-postprocess/results/timings.csv: thời gian từng lượt, từng bước.
- .cache/day12-postprocess/: class đã compile và output chi tiết từng fixture, không đưa vào Git.

So sánh tọa độ/ID cạnh và tập ứng viên trước, rồi so các thành phần điểm với sai số tối đa 1e-8. Thứ tự giữa các box bằng điểm không bắt buộc giống nhau. Bốn ca input lỗi kiểm tra Java trả rỗng an toàn; không tính là tương đương source Python.

Đã kiểm tra thêm phép đổi tọa độ map sang ảnh 640×480. Camera có letterbox/crop cần phép biến đổi ngược riêng. Đây chưa phải tích hợp hay benchmark Android.
