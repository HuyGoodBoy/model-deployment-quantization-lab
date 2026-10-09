# Chạy lại source Đức trên Linux / CPU

Không yêu cầu cùng Mac để reproduce. Giữ model, weights, dữ liệu/preprocess, release dependencies và protocol; ghi rõ OS/ISA/CPU/backend khi đối chiếu số trên hai máy.

## Chuẩn bị source và image

PowerShell tại root repo. Docker Desktop dùng Linux containers cần hoạt động. Không sửa cấu hình/ổ đĩa Docker nếu không hiểu dữ liệu hiện có. Source Đức chỉ đọc; mọi output ghi riêng.

```powershell
git -c core.autocrlf=false clone https://github.com/TruongDuke/Quantization.git .cache/duke-readonly
git -C .cache/duke-readonly checkout --detach 536b9ae1f9e1b3829ac4776a98fb05677df9b6aa
docker build -t huy-peer-day09:cpu environments/peer-linux
docker volume create huy-peer-day09-artifacts
New-Item -ItemType Directory -Force daily/2026-10-09/experiments/01-reproduce/results
```

Nếu đã có checkout đúng revision thì bỏ hai lệnh git. Wrapper kiểm SHA256 nguồn (chấp nhận LF/CRLF tương đương), sau chạy kiểm raw bytes không thay đổi. Dockerfile ghim digest Python 3.11 và đúng mười release versions repo Đức; Torch/torchvision dùng CPU wheels từ kho chính thức. `uv pip check` chạy khi build mới; image lịch sử trong metadata dùng công cụ cài tại thời điểm đo. Full transitive environment Mac chưa có lockfile để xác nhận giống hoàn toàn.

Cần archive `quantization/.cache/imagenette2-160.tgz` và cached weights `.cache/torch-peer/hub/checkpoints/resnet18-f37072fd.pth`. Chạy download baseline nếu chưa có archive. Lần đầu tải weights bằng torchvision ở máy có mạng, đặt `TORCH_HOME` vào `.cache/torch-peer`; sau đó có thể chạy container `--network none`. Không copy model/credential vào Git.

## Reproduce ResNet18

```powershell
docker run --rm --name huy-peer-day09-reproduce --network none `
  --mount 'type=bind,source=D:\vin\710,target=/workspace,readonly' `
  --mount 'type=volume,source=huy-peer-day09-artifacts,target=/workspace/daily/2026-10-09/experiments/01-reproduce/artifacts' `
  --mount 'type=bind,source=D:\vin\710\daily\2026-10-09\experiments\01-reproduce\results,target=/workspace/daily/2026-10-09/experiments/01-reproduce/results' `
  huy-peer-day09:cpu python daily/2026-10-09/experiments/01-reproduce/src/reproduce_peer.py --mode full --run-id run-04-peer-linux
```

Thay các host paths theo checkout của bạn. Volume Linux giúp tránh ghi các tensor NPY lớn qua bind mount Windows. Root source/data mount read-only; chỉ results và volume artifact được ghi. Không cần mở port hoặc gửi dữ liệu ra ngoài khi inference.

q1/q2/q3 chạy nguyên script. q4 dùng nguyên hàm runner/evaluate/benchmark, chạy từng variant để một lỗi operator không chặn cả bảng; interceptor thu `times` sau timer, không đổi thuật toán inference. JSON ghi rõ adaptation, status từng variant, hash, package inventory, accuracy/agreement/SNR, MAE/max error và raw 100 latency samples. Protocol source: CPU 4 thread / 10 warm-up / 100 lượt / batch 1. Không điền latency cho variant lỗi.

`run-04-peer-linux` là ID của phiên đã ghi trong báo cáo; khi làm phiên mới, dùng ID khác để không đè evidence. Có thể gọi riêng `--stage prepare|onnx|tflite|evaluate` với cùng mode/ID để tiếp tục; đọc trạng thái các stage trước.

## Model của Huy trên môi trường đã cài

Sau khi source conversion và benchmark Đức kết thúc, chạy tuần tự:

```powershell
docker run --rm --network none `
  --mount 'type=bind,source=D:\vin\710,target=/workspace,readonly' `
  --mount 'type=bind,source=D:\vin\710\daily\2026-10-09\experiments\01-reproduce\results,target=/workspace/daily/2026-10-09/experiments/01-reproduce/results' `
  huy-peer-day09:cpu python daily/2026-10-09/experiments/01-reproduce/src/run_huy_peer_environment.py
```

Script dùng ID riêng `run-04-huy-linux` và đúng protocol 4/30/200 của matrix Windows Huy để đối chiếu cùng file/input hash. Đây là phép đối chứng riêng, không ghép trực tiếp với benchmark ResNet18 4/10/100. Không chạy benchmark song song. Giữ cùng power mode, ghi nhiệt độ và lặp nhiều phiên nếu muốn kết luận ổn định.

Sau benchmark, `audit_peer.py` đo profile riêng và đếm mẫu hòa; `verify_peer_results.py` kiểm offline hash, metric và raw timing trong cùng volume, không invoke model. Lần chạy cũ bị trùng tên file raw FP32; `output_export_verification.json` ghi việc phục hồi/tách tên và xác nhận metric/latency chính không thay đổi. Collector hiện tại dùng tên riêng ONNX/TFLite nên không còn trùng. Helper phục hồi của phiên cũ đã được lưu cục bộ; collector hiện tại không cần bước này.

## Evidence và bài học

```powershell
.\.venv-day09\Scripts\python.exe daily/2026-10-09/tools/build_peer_report.py
git -C .cache/duke-readonly status --short
```

Generator đọc số Đức từ báo cáo ghim revision và số Huy từ JSON thực đo; cập nhật `PEER-REPRODUCTION.md`. Học liệu HTML giữ cục bộ trong docs, không thuộc Git. Không lấy số Mac làm số Huy đo. Các NPZ/NPY/model/cache vẫn ignore. Artifact Linux còn trong volume `huy-peer-day09-artifacts`; có thể dùng helper container + `docker cp` để xuất ra máy nếu cần.

Windows ONNX-only là fallback riêng, bỏ import và export TFLite trong AST adaptation q1; không gọi nó là full pipeline. Lần đó nằm ở `run-03-peer-windows`, source gốc vẫn nguyên vẹn.
