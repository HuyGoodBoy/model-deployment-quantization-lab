Anh ơi, em Huy báo cáo ngày 19/10 ạ:

- Đã nghiên cứu cách ONNX Runtime thực thi QDQ. Profiler xác nhận bản cũ đã fuse 22 Conv, còn 19 Conv float. Trong lượt đối chứng riêng, tắt pass QDQ làm median tăng 153,59 → 177,55 ms, nên chưa thể quy nguyên nhân chậm chỉ cho việc chưa fuse.
- Thử lượng tử hóa chọn lọc M-LSD: giữ một hoặc hai block head cuối ở float. Vùng thử được chọn bằng profiler sau warm-up và sai số Conv trên cùng đầu vào; không train hay calibration lại.
- Bản giữ hai block được chọn trên 16 ảnh development. Median giảm 144,53 → 133,26 ms, tương đương 7,8%, thấp hơn ở cả 5 vòng đo. sAP10 trên 54 ảnh hồi quy tăng 11,96% → 12,31%.
- Bản mới chạy 19 Conv integer và 22 Conv float, giảm 4 node DQ và 2 node Q. Dung lượng 1,21 MB so với 2,51 MB của bản QAT chạy float, nhưng tốc độ và sAP10 vẫn thấp hơn bản float.
- Đã kiểm chứng lại metric, timing và output. Sai lệch export còn 8,16%, chưa đạt ngưỡng 2%; 54 ảnh đã dùng trước đây nên chỉ là kiểm tra hồi quy. Tiếp theo em sẽ xác định vị trí bắt đầu lệch giữa PyTorch và ONNX; chưa đo Android.

Code và báo cáo: [model-deployment-quantization-lab](https://github.com/HuyGoodBoy/model-deployment-quantization-lab/tree/main/daily/2026-10-19).

[Nghiên cứu chi tiết](RESEARCH.md) · [Kết quả](experiments/selective/results/regression.json) · [Kiểm chứng](experiments/selective/results/verification.json) · [Cách chạy](RUNNING.md)
