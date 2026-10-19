# Thử nghiệm ngày 19/10: chọn vùng float cho M-LSD

Thiết kế đã được duyệt trong chat trước khi chạy. Mục tiêu là kiểm tra việc chủ động bỏ lượng tử hóa ở một vùng có cải thiện chất lượng–latency so với QDQ ngày 16 hay không. Không train, không fit lại scale, không đổi decoder.

1. Khóa hash checkpoint, model cũ, dữ liệu và môi trường. Giữ nguyên 16 ảnh development; 54 ảnh ngày 16 chỉ là tập kiểm tra hồi quy đã được xem trước đây.
2. Profile QDQ và ConvInteger sau 16 lượt warm-up, lấy 8 lượt development tiếp theo. Đếm kernel/provider/dtype từ trace và lưu graph đã tối ưu. Thời gian profiler chỉ dùng chẩn đoán, không thay latency đo khi tắt profiler.
3. Đo sai số lượng tử cục bộ của từng Conv trên cùng input từ 4 ảnh development cố định (0, 5, 10, 15), so fake-quant Conv với Conv float của cùng checkpoint. Gom theo block backbone/head. Chọn block có tích `thời gian Conv đã warm × relative L2 cục bộ` lớn nhất; đây là heuristic để chọn vùng thử, không phải bằng chứng block đó gây sai số toàn mạng.
4. Ứng viên A bỏ quant activation và weight ở block được chọn. B mở rộng A thêm block ngay trước và sau trong cùng backbone/head nếu có. Các weight float lấy từ checkpoint QAT; quantizer và scale ở phần còn lại, kể cả output, giữ nguyên. Việc này thay precision có chủ đích, không được gọi là chỉ fuse.
5. Đối chứng: teacher FP32, weight QAT tắt toàn bộ fake quant, custom 41 ConvInteger, QDQ ALL và cùng artifact QDQ với optimizer DISABLE_ALL. Bật/tắt toàn optimizer không cô lập tác động của riêng fusion.
6. Đánh giá 16 ảnh development và benchmark tuần tự, CPU ORT 4 intra/1 inter threads, spinning tắt, batch 1, 5 vòng xoay/đảo thứ tự, mỗi model/vòng warm-up 16 và đo 16. Lưu 80 raw timing/model, median, p95, graph và model hash.
7. Chọn A/B bằng development trước khi mở kết quả hồi quy: giữ mức giảm sAP10 so QDQ không quá 0,5 điểm; ưu tiên ứng viên có median thấp hơn QDQ và không bị ứng viên khác trội cả chất lượng lẫn tốc độ. Nếu không có, giữ QDQ baseline. Không buộc INT8 thắng FP32.
8. Tách gate export (ONNX so với PyTorch cùng precision policy: relative L2 ≤2%, |chênh sAP10| ≤0,5 điểm) với đánh đổi chất lượng do đổi precision (so QDQ/teacher). Không đổi gate cũ để làm kết quả đạt. Đánh giá trên 54 ảnh sau khi khóa lựa chọn, gọi đúng là hồi quy; bootstrap ảnh ghép cặp chỉ mô tả tập này.
9. Kiểm chứng lại hash, metric, raw timing, lựa chọn trước hồi quy và kernel thực thi. Báo cáo chính ngắn; chi tiết tái lập và các kết quả âm nằm ở RESEARCH.md/RUNNING.md/JSON.

Nguồn: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
