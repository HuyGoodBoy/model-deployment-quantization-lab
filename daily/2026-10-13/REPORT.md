# Báo cáo task 13/10

Người làm: Bùi Gia Huy. Chạy bổ sung ngày 14/10.

Em thử tiếp PTQ M-LSD, dùng 32 ảnh Wireframe train để calibration và 16 ảnh YorkUrban có nhãn để đánh giá. Giữ cùng checkpoint, không train lại.

| Bản | SNR output đang dùng | sAP10 trên subset |
|---|---:|---:|
| FP32 | 95,62 dB | 15,16% |
| FP16 | 24,72 dB | 15,02% |
| INT8, head chung | 5,92 dB | 0,57% |
| INT8, tách center/displacement | 5,94 dB | 9,93% |
| Mixed, giữ 6 Conv cuối và Resize đầu ra ở FP32 | 6,01 dB | 14,26% |

Tách head giúp center có scale riêng: bước lượng tử giảm từ 5,81 xuống 0,16. SNR chung gần như không đổi nhưng detect line tốt hơn rõ. Với bản mixed, chỉ giữ Conv ở float chưa đủ vì Resize vẫn đưa output về INT8; giữ cả Resize ở float mới đạt 14,26%. Bản này không phải full INT8.

FP16 giữ chất lượng detect khá gần FP32 trên subset này, nên chưa thể nói FP16 tệ chỉ từ SNR thấp. Thử đổi calibration và giữ một số weight FP32 chưa cải thiện rõ. Tổng cộng đã chạy 12 cấu hình; kết quả subset chưa đại diện toàn bộ dataset.

Tiếp theo em chuyển sang QAT để thử giảm sai số còn lại ở INT8.

[Code và kết quả](experiments/mlsd-ptq/) · [Tổng hợp](experiments/mlsd-ptq/results/summary.json) · [Cách chạy](RUNNING.md)
