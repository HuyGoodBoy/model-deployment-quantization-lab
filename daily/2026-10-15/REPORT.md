# Báo cáo ngày 15/10/2026 — QAT và kiểm tra export

Người làm: Bùi Gia Huy

Em train tiếp QAT đến 128 bước rồi kiểm tra vì sao M-LSD chưa tốt lên. Sau các lượt đối chứng, phần cải thiện rõ nhất là hiệu chỉnh range lượng tử ở output, không phải tăng số bước train. Lượt bổ sung cuối ngày đã sửa một khác biệt làm tròn khi export và xác định model còn nhạy với backend convolution; độ lệch tensor toàn mạng vẫn chưa đạt gate 2%.

| M-LSD, sAP10 | 16 ảnh chọn cấu hình | 32 ảnh mới |
|---|---:|---:|
| FP32 | 15,16% | 12,34% |
| PTQ tách head, full INT8 | 9,93% | 7,76% |
| QAT + ONNX integer cũ, mixed | 7,93% | 6,69% |
| QAT + range mới + ONNX integer, mixed | **13,73%** | **10,98%** |

- Range cũ bị kéo rộng bởi outlier ở nền. Em lấy range displacement ở vùng teacher có confidence cao trên 32 ảnh Wireframe train, tách scale theo kênh và chỉnh range center; bước lượng tử displacement giảm từ 7,03 xuống khoảng 0,49–0,87. Decoder và ngưỡng đánh giá giữ nguyên.
- Em thử riêng đổi range, đổi Smooth L1 và kết hợp cả hai, cùng checkpoint/optimizer và thêm 96 bước mỗi nhánh. Đổi loss riêng chưa cải thiện rõ. Cấu hình được chọn vẫn là weight QAT bước 32 cộng calibration mới; không lấy điểm tăng này làm bằng chứng train thêm hiệu quả.
- Trên 32 ảnh YorkUrban chưa dùng chọn cấu hình, bản integer tăng **4,29 điểm phần trăm**, khoảng tin cậy bootstrap 95% **[2,90; 6,36]**. Bản mới giữ khoảng 89% mức sAP của FP32 trên tập này. Kết quả vẫn là subset, không phải benchmark trong paper.
- Profiler xác nhận bản calibration dùng 41 Conv integer 8-bit, tích lũy INT32; bias, cộng, resize và các bước khác còn float. Đo lại chỉ một ORT session mỗi lượt, 4 thread: median FP32 79,42 ms, integer cũ 230,40 ms, integer sau calibration 231,62 ms/ảnh. Chất lượng tăng nhưng chưa có lợi về tốc độ. Các số thời gian này thuộc bản export trước sửa làm tròn.
- DistilBERT đã train 128 bước trên 256 câu SST-2, CE + distillation, freeze embedding; accuracy vẫn 90% trên 100 câu riêng. Training dùng PyTorch CPU, AdamW lr=1e-5.

**Bổ sung: sửa và kiểm chứng phép làm tròn khi export M-LSD.** Giữ nguyên checkpoint bước 32 đã chọn, toàn bộ scale/zero-point, weight, bias và decoder. Chỉ thay cách biểu diễn 43 bộ lượng tử activation/output để giữ thứ tự tính của fake quant PyTorch CPU: nhân nghịch đảo FP32 → làm tròn → cộng zero-point → clip → cast.

| Trên cùng 16 ảnh development | Relative L2 so với fake quant | sAP10 |
|---|---:|---:|
| ONNX integer sau calibration, trước sửa làm tròn | 5,9066% | 13,7334% |
| ONNX integer biểu diễn làm tròn mới | **5,6565%** | 13,7053% |

- Trên ảnh development đầu tiên, biểu thức chia scale lệch 37 giá trị trong khoảng 37,27 triệu giá trị qua 43 bộ lượng tử; biểu thức nhân nghịch đảo khớp fake quant ở toàn bộ giá trị đã kiểm tra. Kiểm thử ORT riêng ở điểm giữa hai mức lượng tử, các số FP32 lân cận, clipping và scale theo kênh đều khớp PyTorch.
- So riêng từng convolution bằng cùng đầu vào, relative L2 lớn nhất của 41 lớp sau sửa là **9,05×10⁻⁷**. Chỉ đổi backend convolution trong PyTorch, cùng weight và fake quant, cũng tạo relative L2 toàn mạng **5,4702%** trên 16 ảnh. Bằng chứng này cho thấy phép làm tròn và sai số số học qua nhiều lớp cùng cần được xem xét; chưa chứng minh đã xử lý toàn bộ nguyên nhân.
- Profiler bản thử mới vẫn chạy đủ **41 ConvInteger trên CPU**, không có Conv float. Sai số giảm **0,2501 điểm phần trăm**, nhưng gate 2% vẫn thất bại. sAP10 giảm nhẹ 0,0282 điểm so với bản ONNX trước sửa; không kết luận chất lượng detect line tăng thêm. Chưa benchmark tốc độ bản làm tròn mới và chưa đánh giá bản này trên tập ảnh độc lập.
- Verifier tái chạy đủ 16 ảnh, tính lại metric và kiểm tra hash: checkpoint, model baseline, quyết định chọn cấu hình và kết quả 32 ảnh trước đó giữ nguyên. **Kiểm chứng bằng chứng đạt; gate tương đương tensor chưa đạt.**

Task tuần còn xử lý độ nhạy số học khi export, thử dữ liệu/loss supervised rộng hơn, tối ưu backend và export DistilBERT. Chưa đo Android. Bản làm tròn mới là một thực nghiệm riêng, chưa thay thế bản deployment đã báo cáo trên 32 ảnh.

[Phân tích và đối chứng](RESEARCH.md) · [32 ảnh mới và khoảng tin cậy](experiments/qat/results/research/holdout_evaluation.json) · [Kết quả sửa export](experiments/qat/results/export-parity/rounding.json) · [Kiểm chứng export](experiments/qat/results/export-parity/verification.json) · [Cách chạy](RUNNING.md)
