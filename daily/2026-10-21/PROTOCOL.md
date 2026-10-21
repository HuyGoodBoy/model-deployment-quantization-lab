# Kế hoạch thực nghiệm ngày 21/10/2026

Người dùng đã duyệt kế hoạch ngày21 và yêu cầu thực hiện, giữ nguyên báo cáo20/10. Báo cáo mới nhóm theo project, Evidence đặt đầu mỗi nhóm; chat ngắn, không kể AI nào review. REPORT.md đầy đủ phương pháp, đối chứng, bảng số liệu và giới hạn. Không tự sửa lịch sử commit20/10 hoặc tự lập lại lịch commit17:00 của ngày cũ.

## M-LSD

- Giữ nguyên checkpoint, frontend, decoder và32 ảnh Wireframe calibration ngày20;16 ảnh York development để chọn. Không fit trên54 ảnh hồi quy đã dùng. Hash kiểm đầu/cuối các nguồn ngày cũ.
- So activation float/AEQ static theo cùng output identity, dtype/shape/layout được kiểm. Dùng Interpreter preserve_all_tensors, không delegate cho diagnostic; lượng tử input theo scale thực, dequant output để so. Debug execution không dùng cho benchmark. Ghi riêng center/displacement ở output và dấu hiệu clipping/range.
- Hai giả thuyết đã duyệt: A) giữ toàn bộ head ngoài backbone float, gồm biên resize/add/output; B) chia Conv cuối thành center1/displacement4 trước lượng tử hóa để hai nhánh có range riêng. Model thật có16filter, center làfilter7,displacement là8:12; bỏ các filter không được ActiveMlsd dùng. Kiểm parity FP32 của split trước PTQ; không chỉ tách tensor sau quantization. Hai output LiteRT split làNHWC, adapter chuyểnNCHW rồi ghép sau dequant; tính cả adapter khi benchmark.
- So FP32, static ngày20 và các candidate trên cùng runtime/thread/input. Gate development: AP không thấp hơn float quá0.5pp, latency thấp hơn float cùng phép đo. Gate mới dùng float làm control chất lượng, không dùng static AP thấp làm chuẩn.
- Chạy benchmark tuần tự5vòng xoay/đảo;16warmups+16timings/vòng, batch1. CPU4threads là so precision; kiểm thêm8threads cho candidate đủ chất lượng và control nếu cần. Không so số ngày hôm qua với hôm nay để quy tốc độ.
- Khóa cấu hình trước hồi quy54; đó là regression, không gọi fresh holdout. Nếu cả hai không đạt, giữ float, báo kết quả âm và nguyên nhân. Không thêm train/calibration mới cùng lúc để tránh đổi nhiều yếu tố.

## ZipFormer

- Giữ bản Conv2D ngày20 làm LiteRT baseline. Người dùng xác nhận không có Claude và yêu cầu dùng Codex thay thế; áp dụng từ ngày21, không sửa attribution lịch sử ngày20. Không tiếp tục yêu cầu login Claude.
- Profile encoder để chọn tối đa2 biến đổi cụ thể, dùng Codex phân tích/viết patch; lưu công cụ/lệnh thực dùng, provenance và test. Báo rõ thay công cụ theo yêu cầu người dùng, không gọi đây là một lượt Claude.
- Parity tensor/state, token/WER và latency/RTF cùng pipeline. Không trộn Icefall context[0,0] với Sherpa[-1,0]. Tập73câu một speaker vẫn là subset; audio thêm người đọc chỉ bổ sung khi có đủ thời gian/provenance.
- Bổ sung được khai báo sau khi rewriteFC đã cố định: tải LibriSpeech test-clean chính thức OpenSLR, MD5 `32fa31d27d2e1cad72775fee3f4849a9`. Seed20261021 chọn20speaker,5câu/speaker,loại speaker/ID của73câu cũ;khóa100câu trước inference. Không dùng tập bổ sung để sửa/chọn lại patch; không gọi đây là full test-clean WER hay dữ liệu chưa từng có trong pretraining.

## Kiểm chứng và báo cáo

Mỗi JSON lưu model/input/sourcehash, cấu hình và mẫu timing. Verifier tái tính AP/timing/parity từ output, kiểm native dtype/precision coverage và khóa chọn trước hồi quy. Chỉ benchmark sau khi hết workload khác. Dùng Markdown và bảng; ảnh chỉ khi giúp giải thích kết quả. Không gọi gate integrity pass là model đã đạt chất lượng triển khai.
