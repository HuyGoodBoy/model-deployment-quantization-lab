# Báo cáo tiến độ 14/10

Người làm: Bùi Gia Huy

Hôm nay em bắt đầu QAT cho M-LSD và DistilBERT. Mỗi model mới chạy thử 8 bước để kiểm tra pipeline, chưa xong task tuần này.

- M-LSD: dùng 32 ảnh Wireframe train, teacher FP32 tạo pseudo-label. Loss tách BCE cho center và Smooth L1 cho displacement. Batch 1; lượt này dùng 8 ảnh để cập nhật weight.
- DistilBERT: chuẩn bị 256 câu SST-2 train, đánh giá trên 100 câu validation riêng. Dùng CE theo nhãn kết hợp distillation từ teacher FP32, freeze embedding. Batch 2; lượt này dùng 16 câu để cập nhật weight.
- Train bằng PyTorch CPU, AdamW lr=1e-5. Fake quant weight INT8 per-channel và activation INT8 per-tensor; observer lấy range trên train/calibration rồi freeze.

Kiểm tra fake quant trước/sau lượt train: M-LSD sAP10 từ 7,60 lên 7,75%; DistilBERT accuracy từ 89% lên 90% (FP32 91%). Mức thay đổi còn nhỏ, chưa đủ kết luận QAT tốt hơn PTQ.

Hiện PyTorch trên máy chỉ có CPU. Attention/normalization và một số phép toán vẫn giữ float; chưa xuất model chạy INT8 thật. Phần còn lại của tuần là train lâu hơn, mở rộng đánh giá rồi export/audit INT8 và đo tốc độ. Đã lưu checkpoint để làm tiếp.

[Code và kết quả](experiments/qat/) · [M-LSD](experiments/qat/results/mlsd_progress.json) · [DistilBERT](experiments/qat/results/text_progress.json) · [Cách chạy](RUNNING.md)
