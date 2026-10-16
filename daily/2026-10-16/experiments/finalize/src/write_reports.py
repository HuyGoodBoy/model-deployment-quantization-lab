"""Write concise day/week reports directly from verified experiment evidence."""

from common import RESULTS, ROOT, read


def gpu_sections():
    path = RESULTS / "gpu/verification.json"
    if not path.exists():
        return "", ""
    gpu = read(path)
    assert gpu["status"] == "passed"
    rows = gpu["rows"]
    ml_cpu = rows["mlsd_fp32_cpu"]
    ml_half = rows["mlsd_fp16_cuda"]
    tx_cpu = rows["text_fp32_cpu"]
    tx_half = rows["text_fp16_cuda"]
    tx_float_gpu = rows["text_fp32_cuda"]
    tx_integer_cpu = rows["text_qat_static_cpu"]
    tx_integer_cuda = rows["text_qat_static_cuda"]
    day = (
        f"\nEm bổ sung kiểm thử trên RTX 3050 Laptop 4 GB. Cùng ORT 1.20.2, M-LSD "
        f"FP16 GPU đạt {ml_half['median_ms']:.2f} ms/ảnh, sAP10 {ml_half['quality_percent']:.2f}%; "
        f"DistilBERT FP32 GPU đạt {tx_float_gpu['median_ms']:.2f} ms/câu, accuracy "
        f"{tx_float_gpu['quality_percent']:.2f}%; FP16 chưa nhanh hơn. Profiler cho thấy QAT integer vẫn có phép tính ở CPU, "
        "còn convolution của M-LSD QDQ chạy float trên GPU; chưa có GPU INT8 hoàn chỉnh.\n"
    )
    table = []
    for task, variant, label in [
        ("mlsd", "fp32", "M-LSD FP32"),
        ("mlsd", "fp16", "M-LSD FP16"),
        ("mlsd", "qat_41_integer", "M-LSD QAT 41 ConvInteger"),
        ("mlsd", "qat_qdq", "M-LSD QAT QDQ"),
        ("text", "fp32", "DistilBERT FP32"),
        ("text", "fp16", "DistilBERT FP16"),
        ("text", "qat_static", "DistilBERT QAT static"),
    ]:
        cpu, cuda = rows[f"{task}_{variant}_cpu"], rows[f"{task}_{variant}_cuda"]
        assert cpu["status"] == cuda["status"] == "ok"
        table.append(
            f"| {label} | {cpu['median_ms']:.2f} / {cuda['median_ms']:.2f} | "
            f"{cpu['quality_percent']:.2f}% / {cuda['quality_percent']:.2f}% |"
        )
    section = """## Kiểm thử thêm trên GPU máy hiện tại

RTX 3050 Laptop 4 GB, driver 572.61. Lượt này dùng cùng ONNX Runtime GPU 1.20.2 cho cả CPU và CUDA, tách khỏi bảng CPU ORT 1.30 ở trên. Giữ nguyên 54 ảnh và 772 câu, không chọn lại model. Chất lượng là sAP10 với M-LSD, accuracy với DistilBERT.

| Model | Median CPU / CUDA (ms) | Chất lượng CPU / CUDA |
|---|---:|---:|
""" + "\n".join(table)
    section += (
        f"\n\nM-LSD FP16 GPU đạt "
        f"{ml_half['median_ms']:.2f} ms so với CPU FP32 {ml_cpu['median_ms']:.2f} ms; "
        f"DistilBERT FP32 GPU đạt {tx_float_gpu['median_ms']:.2f} ms so với CPU "
        f"{tx_cpu['median_ms']:.2f} ms; FP16 {tx_half['median_ms']:.2f} ms chưa cải thiện tốc độ. "
        "FP16 là giảm precision float, không phải QAT INT8.\n\n"
        "Profiler xác nhận 41 ConvInteger của M-LSD và 38 MatMulInteger của DistilBERT "
        "vẫn ở CPU khi bật CUDA; phần float chạy GPU. M-LSD QDQ trên ORT 1.20.2 "
        "chạy 41 Conv float cả CPU và CUDA, khác bản CPU ORT 1.30 đã fuse 22 Conv integer. "
        f"DistilBERT QAT bị chia CPU/GPU kèm truyền dữ liệu, median CUDA "
        f"{tx_integer_cuda['median_ms']:.2f} ms còn chậm hơn CPU {tx_integer_cpu['median_ms']:.2f} ms. "
        "Vì vậy tốc độ GPU ở đây chưa chứng minh tăng tốc integer.\n\n"
        "Mỗi cấu hình chạy trong process riêng, 16 warm-up và 80 lượt đo, batch 1, 4 thread; "
        "session.run đồng bộ gồm truyền CPU–GPU, không gồm preprocess/decode. "
        "TF32 tắt. cuDNN báo không có execution plan ở một số lần chọn kernel FP16 M-LSD "
        "nhưng inference và profiler hoàn tất với output hữu hạn. "
        "Đây là đối chiếu runtime trên dữ liệu đã đánh giá, không phải test set mới. "
        "[Bằng chứng GPU và kiểm chứng](experiments/finalize/results/gpu/verification.json).\n\n"
    )
    return day, section


def main():
    ml = read(RESULTS / "mlsd_final.json")
    tx = read(RESULTS / "text_final.json")
    mb = read(RESULTS / "mlsd_benchmark.json")["results"]
    tb = read(RESULTS / "text_benchmark.json")["results"]
    verification = read(RESULTS / "verification.json")
    assert verification["status"] == "passed"
    ml_selection = read(RESULTS / "mlsd_selection.json")
    tx_selection = read(RESULTS / "text_selection.json")
    examples = read(RESULTS / "visual_examples.json")["examples"]
    failure = min(
        examples, key=lambda row: row["models"]["qat_selected_qdq"]["sAP10_subset_percent"]
    )

    def ap(name):
        return ml["models"][name]["metrics"]["line_quality"]["sAP10_subset_percent"]

    def accuracy(name):
        return tx["models"][name]["accuracy_percent"]

    ml_fp, ml_new = (
        mb["original_fp32"]["median_ms"],
        mb["selected_qdq_22_integer_19_float"]["median_ms"],
    )
    tx_fp, tx_new = tb["teacher_fp32"]["median_ms"], tb["qat_static_38_integer"]["median_ms"]
    ml_parity = ml_selection["candidates"][ml_selection["selected"]]["parity"]["relative_l2"] * 100
    tx_parity = tx_selection["candidates"][tx_selection["selected"]]["parity"]["relative_l2"] * 100
    comparison = ml["paired_comparisons"]["qat_original_integer_to_qat_calibrated_integer"]
    ci = comparison["paired_image_bootstrap_95_percentile_ci_pp"]
    directory = ROOT / "daily/2026-10-16"
    day = f"""# Báo cáo ngày 16/10/2026

Người thực hiện: Bùi Gia Huy.

Hôm nay em hoàn thiện export và đánh giá M-LSD trên 54 ảnh YorkUrban, DistilBERT trên 772 câu SST-2 mới đối với các lượt thực nghiệm đã ghi nhận. Dữ liệu được khóa và kiểm tra không trùng trước khi đánh giá.

M-LSD sau calibration đạt sAP10 {ap("qat_calibrated_integer"):.2f}%, tăng {comparison["delta_sAP10_pp"]:.2f} điểm so với bản integer cũ; FP32 đạt {ap("original_fp32"):.2f}%. Bản QDQ đạt {ap("qat_selected_qdq"):.2f}%, median {ml_new:.2f} ms/ảnh so với FP32 {ml_fp:.2f} ms. Bản này chạy 22 Conv integer và 19 Conv float, vẫn chưa nhanh hơn FP32.

DistilBERT đã export static W8A8 cho 38 Linear. Bản integer đúng {tx["models"]["qat_static_integer"]["correct"]}/{tx["samples"]} câu ({accuracy("qat_static_integer"):.2f}%), FP32 đúng {tx["models"]["teacher_fp32"]["correct"]}/{tx["samples"]} ({accuracy("teacher_fp32"):.2f}%). Median inference giảm từ {tx_fp:.2f} xuống {tx_new:.2f} ms/câu; embedding, attention MatMul và normalization còn float.

Đã kiểm lại metric, profiler và timing. Tuy nhiên, relative L2 khi export trên development còn {ml_parity:.2f}% với M-LSD QDQ và {tx_parity:.2f}% với DistilBERT, đều vượt gate 2%. Chưa hoàn thành full INT8 và chưa đo Android.

[Tổng kết tuần](WEEKLY-REPORT.md) · [Kết quả M-LSD](experiments/finalize/results/mlsd_final.json) · [DistilBERT](experiments/finalize/results/text_final.json) · [Kiểm chứng](experiments/finalize/results/verification.json) · [Cách chạy](RUNNING.md)
"""
    weekly = f"""# Báo cáo tuần 12–16/10/2026

Người thực hiện: Bùi Gia Huy.

Trong tuần em tiếp tục hậu xử lý M-LSD, thử PTQ, chạy QAT cho M-LSD và DistilBERT đến 128 bước/model, rồi export và đánh giá trên CPU. M-LSD dùng 32 ảnh Wireframe train với pseudo-label teacher; DistilBERT dùng 256 câu SST-2, CE kết hợp distillation. Đây là thực nghiệm nhỏ, chưa phải training đầy đủ.

## Kết quả cuối tuần

M-LSD được đánh giá trên 54 ảnh mới đối với các manifest thực nghiệm đã ghi nhận, dùng cùng decoder và evaluator subset. DistilBERT dùng 772 câu validation mới, cùng tokenizer và độ dài 64 token. Các khoảng tin cậy và dữ liệu thô được lưu cùng kết quả.

| M-LSD | sAP10 trên 54 ảnh | Median / p95 ms | Precision thực thi |
|---|---:|---:|---|
| FP32 | {ap("original_fp32"):.2f}% | {ml_fp:.2f} / {mb["original_fp32"]["p95_ms"]:.2f} | Float |
| PTQ tách head | {ap("ptq_split_full_int8"):.2f}% | Chưa đo cùng lượt này | Full INT8 TFLite |
| QAT integer cũ | {ap("qat_original_integer"):.2f}% | Chưa đo cùng lượt này | 41 ConvInteger, boundary float |
| QAT + calibration | {ap("qat_calibrated_integer"):.2f}% | {mb["calibrated_41_integer"]["median_ms"]:.2f} / {mb["calibrated_41_integer"]["p95_ms"]:.2f} | 41 ConvInteger, boundary float |
| QAT + QDQ | {ap("qat_selected_qdq"):.2f}% | {ml_new:.2f} / {mb["selected_qdq_22_integer_19_float"]["p95_ms"]:.2f} | 22 Conv integer, 19 Conv float |

| DistilBERT | Đúng / 772 câu | Accuracy | Median / p95 ms |
|---|---:|---:|---:|
| FP32 teacher | {tx["models"]["teacher_fp32"]["correct"]} | {accuracy("teacher_fp32"):.2f}% | {tx_fp:.2f} / {tb["teacher_fp32"]["p95_ms"]:.2f} |
| Weight QAT, tắt fake quant | {tx["models"]["qat_weights_fp32"]["correct"]} | {accuracy("qat_weights_fp32"):.2f}% | {tb["qat_weights_fp32"]["median_ms"]:.2f} / {tb["qat_weights_fp32"]["p95_ms"]:.2f} |
| QAT fake quant | {tx["models"]["qat_fake_quant"]["correct"]} | {accuracy("qat_fake_quant"):.2f}% | Chưa benchmark PyTorch |
| QAT static integer | {tx["models"]["qat_static_integer"]["correct"]} | {accuracy("qat_static_integer"):.2f}% | {tx_new:.2f} / {tb["qat_static_38_integer"]["p95_ms"]:.2f} |

Timing là inference CPU, batch 1, 4 thread, một ORT session mỗi lượt, 80 mẫu/model; không gồm load, preprocess, decode hoặc Android. Model M-LSD FP32/QDQ có kích thước {mb["original_fp32"]["model"]["bytes"] / 1048576:.2f}/{mb["selected_qdq_22_integer_19_float"]["model"]["bytes"] / 1048576:.2f} MiB; DistilBERT FP32/integer là {tb["teacher_fp32"]["model"]["bytes"] / 1048576:.2f}/{tb["qat_static_38_integer"]["model"]["bytes"] / 1048576:.2f} MiB.

![Chất lượng trên dữ liệu mới](figures/final-quality.png)

## Nhận xét

Cải thiện rõ nhất của M-LSD đến từ calibration output. Cùng phạm vi 41 ConvInteger, sAP10 tăng {comparison["delta_sAP10_pp"]:.2f} điểm, CI 95% [{ci[0]:.2f}; {ci[1]:.2f}]. Train thêm hoặc đổi loss chưa chứng minh hiệu quả rõ, nên em không quy điểm tăng này cho việc train QAT lâu hơn. QDQ giảm latency so với bản 41 ConvInteger nhưng vẫn chậm hơn FP32 và có 19 Conv trở lại float.

DistilBERT đã xác nhận 38 Linear chạy integer bằng profiler. Bản export giữ static scale QAT; embedding, attention MatMul/Softmax và normalization vẫn float. So với fake quant trên tập mới, chênh accuracy là {tx["accuracy_delta_integer_minus_fake_pp"]:.2f} điểm, CI paired bootstrap 95% [{tx["paired_bootstrap_delta_95_ci_pp"][0]:.2f}; {tx["paired_bootstrap_delta_95_ci_pp"][1]:.2f}]. Chưa có cơ sở kết luận QAT tốt hơn PTQ từ các đối chứng hiện có.

Trong bốn ảnh minh họa cố định, {failure["image_file"]} là ca khó ở cả baseline: sAP10 trên riêng ảnh này chỉ {failure["models"]["original_fp32"]["sAP10_subset_percent"]:.2f}% với FP32 và {failure["models"]["qat_selected_qdq"]["sAP10_subset_percent"]:.2f}% với QDQ. Cần xem thêm dữ liệu và decoder, không quy toàn bộ lỗi cho lượng tử hóa.

## Phần chưa đạt

Gate tương đương tensor trên development vẫn thất bại: relative L2 M-LSD QDQ {ml_parity:.2f}%, DistilBERT {tx_parity:.2f}%. Kiểm chứng bằng chứng đạt không đồng nghĩa model đạt mọi gate. Chưa triển khai full INT8, chưa train M-LSD bằng supervised matching loss và chưa đo Android. Đánh giá là các subset có kiểm soát, không phải benchmark paper hay bằng chứng dữ liệu chưa xuất hiện trong pretrained weights.

Hướng tiếp theo là xử lý độ nhạy số học khi export và thử supervised QAT có đối chứng, sau đó kiểm tra trên thiết bị đích.

[Ảnh so sánh cố định](figures/line-comparison.png) · [M-LSD và CI](experiments/finalize/results/mlsd_final.json) · [DistilBERT và CI](experiments/finalize/results/text_final.json) · [Timing M-LSD](experiments/finalize/results/mlsd_benchmark.json) · [Timing text](experiments/finalize/results/text_benchmark.json) · [Kiểm chứng](experiments/finalize/results/verification.json) · [Tái lập](RUNNING.md)
"""
    gpu_day, gpu_week = gpu_sections()
    if gpu_day:
        day = day.replace("\n[Tổng kết tuần]", gpu_day + "\n[Tổng kết tuần]")
        weekly = weekly.replace("## Phần chưa đạt", gpu_week + "## Phần chưa đạt")
        weekly = weekly.replace(
            "rồi export và đánh giá trên CPU.",
            "rồi export và đánh giá trên CPU, bổ sung đối chiếu CUDA trên GPU máy hiện tại.",
        )
    (directory / "REPORT.md").write_text(day, encoding="utf-8", newline="\n")
    (directory / "WEEKLY-REPORT.md").write_text(weekly, encoding="utf-8", newline="\n")
    print("Wrote concise daily and weekly reports from verified results", flush=True)


if __name__ == "__main__":
    main()
