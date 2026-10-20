# Review Astra ngày20/10

Người dùng yêu cầu Astra lên kế hoạch rồi review sau thực nghiệm. Agent Astra đã đọc code, bản REPORT ban đầu, RESEARCH và bằng chứng; không sửa file hoặc chạy workload cạnh tranh benchmark. Sau đó REPORT.md được biên tập lại thành báo cáo chi tiết theo yêu cầu người dùng, giữ nguyên kết quả thí nghiệm; review dưới đây áp dụng cho số liệu và thí nghiệm, không tuyên bố đã review lại bản diễn đạt mới.

Các điểm được yêu cầu và đã xử lý:

- Prewarm input_specs ngoài pipeline timing; lưu riêng `matched_benchmark_v2.json`, giữ lần đo đầu làm lịch sử.
- Kiểm exact counter nguyên, không chỉ allclose; nêu giới hạn session dài và adapter INT32→INT64.
- Phân biệt73 câu một speaker cho WER với3 audio smoke cho matched RTF; không đồng nhất pipeline Icefall với native Sherpa.
- Sửa cách diễn giải float fallback: kernel float vẫn có thể nhận Q/DQ; không suy ra bỏ lượng tử tại đó vô ích.
- Sửa size static thành792.192bytes, điền p95 encoder từJSON.
- Verifier kiểm đủ38 Conv weight/output INT8 +3 FP32; tái tính profile từ raw trace và graph, kiểm FP32 8luồng output exact.
- Ghi Claude chưa thực hiện inference/tool do chưa đăng nhập, không gán patch Codex cho Claude. M-LSD source phục hồi đúng SHA256; converter chỉ chạy trên bản sao.

Kết luận Astra: đủ cơ sở khóa snapshot nghiên cứu sau các sửa và verifier pass; không cần chạy thêm benchmark. Giữ Zipformer Conv2D, không chọn lookup; AEQ chưa giữ chất lượng, FP32 8luồng có cải thiện trên CPU đã đo. Task Claude vẫn chưa hoàn thành.

Đối chiếu sau sửa: [VERIFICATION_final.json](VERIFICATION_final.json) đã pass. Ruff đã pass; [test guard commit](tools/schedule_guard_verification.json) kiểm bằng repo tạm riêng. Đây là review snapshot nghiên cứu, không phải chứng nhận các model INT8 đủ điều kiện triển khai hay toàn bộ task tuần đã xong.
