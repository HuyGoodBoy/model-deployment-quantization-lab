# Báo cáo ngày 16/10/2026

Người thực hiện: Bùi Gia Huy.

Hôm nay em hoàn thiện export và đánh giá M-LSD trên 54 ảnh YorkUrban, DistilBERT trên 772 câu SST-2 mới đối với các lượt thực nghiệm đã ghi nhận. Dữ liệu được khóa và kiểm tra không trùng trước khi đánh giá.

M-LSD sau calibration đạt sAP10 12.18%, tăng 4.71 điểm so với bản integer cũ; FP32 đạt 12.78%. Bản QDQ đạt 11.96%, median 48.59 ms/ảnh so với FP32 27.56 ms. Bản này chạy 22 Conv integer và 19 Conv float, vẫn chưa nhanh hơn FP32.

DistilBERT đã export static W8A8 cho 38 Linear. Bản integer đúng 698/772 câu (90.41%), FP32 đúng 703/772 (91.06%). Median inference giảm từ 22.28 xuống 18.44 ms/câu; embedding, attention MatMul và normalization còn float.

Đã kiểm lại metric, profiler và timing. Tuy nhiên, relative L2 khi export trên development còn 8.88% với M-LSD QDQ và 5.55% với DistilBERT, đều vượt gate 2%. Chưa hoàn thành full INT8 và chưa đo Android.

Em bổ sung kiểm thử trên RTX 3050 Laptop 4 GB. Cùng ORT 1.20.2, M-LSD FP16 GPU đạt 5.83 ms/ảnh, sAP10 12.65%; DistilBERT FP32 GPU đạt 6.59 ms/câu, accuracy 91.06%; FP16 chưa nhanh hơn. Profiler cho thấy QAT integer vẫn có phép tính ở CPU, còn convolution của M-LSD QDQ chạy float trên GPU; chưa có GPU INT8 hoàn chỉnh.

[Tổng kết tuần](WEEKLY-REPORT.md) · [Kết quả M-LSD](experiments/finalize/results/mlsd_final.json) · [DistilBERT](experiments/finalize/results/text_final.json) · [Kiểm chứng](experiments/finalize/results/verification.json) · [Cách chạy](RUNNING.md)
