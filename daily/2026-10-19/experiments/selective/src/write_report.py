"""Generate concise mentor report plus detailed reproducible evidence."""

import csv
import json
from pathlib import Path


def main():
    task = Path(__file__).resolve().parents[1]
    day = task.parents[1]
    results = task / "results"

    def read(name):
        return json.loads((results / (name + ".json")).read_text(encoding="utf-8"))

    diagnosis, exports = read("diagnosis"), read("exports")
    dev, benchmark = read("development"), read("benchmark")
    selection, regression = read("selection"), read("regression")
    control, verification = read("fusion_control"), read("verification")
    assert verification["status"] == "passed"
    chosen = selection["selected"]
    timing = benchmark["models"]
    sizes = {
        name: (day.parents[1] / row["model"]["path"]).stat().st_size for name, row in timing.items()
    }
    quality = {n: row["sAP10"]["sAP10_subset_percent"] for n, row in regression["models"].items()}
    b = timing["selective_b"]
    baseline = timing["qdq_all"]
    speed_percent = (1 - b["median_ms"] / baseline["median_ms"]) * 100
    delta = quality["selective_b"] - quality["qdq_all"]
    ci = regression["paired_comparisons"]["selective_b_vs_qdq"]["paired_image_bootstrap_95_ci_pp"]
    round_wins = sum(a < b for a, b in zip(b["round_medians_ms"], baseline["round_medians_ms"]))
    parity_percent = exports["candidates"]["selective_b"]["export_parity"]["relative_l2"] * 100

    def vn(value, digits=2):
        return f"{value:.{digits}f}".replace(".", ",")

    report = f"""Anh ơi, em Huy báo cáo ngày 19/10 ạ:

- Đã nghiên cứu cách ONNX Runtime thực thi QDQ. Profiler xác nhận bản cũ đã fuse 22 Conv, còn 19 Conv float. Trong lượt đối chứng riêng, tắt pass QDQ làm median tăng {vn(control["conditions"]["enabled"]["median_ms"])} → {vn(control["conditions"]["disabled"]["median_ms"])} ms, nên chưa thể quy nguyên nhân chậm chỉ cho việc chưa fuse.
- Thử lượng tử hóa chọn lọc M-LSD: giữ một hoặc hai block head cuối ở float. Vùng thử được chọn bằng profiler sau warm-up và sai số Conv trên cùng đầu vào; không train hay calibration lại.
- Bản giữ hai block được chọn trên 16 ảnh development. Median giảm {vn(baseline["median_ms"])} → {vn(b["median_ms"])} ms, tương đương {vn(speed_percent, 1)}%, thấp hơn ở cả {round_wins} vòng đo. sAP10 trên 54 ảnh hồi quy tăng {vn(quality["qdq_all"])}% → {vn(quality["selective_b"])}%.
- Bản mới chạy 19 Conv integer và 22 Conv float, giảm 4 node DQ và 2 node Q. Dung lượng {vn(sizes["selective_b"] / 1e6)} MB so với {vn(sizes["qat_weights_fp32"] / 1e6)} MB của bản QAT chạy float, nhưng tốc độ và sAP10 vẫn thấp hơn bản float.
- Đã kiểm chứng lại metric, timing và output. Sai lệch export còn {vn(parity_percent)}%, chưa đạt ngưỡng 2%; 54 ảnh đã dùng trước đây nên chỉ là kiểm tra hồi quy. Tiếp theo em sẽ xác định vị trí bắt đầu lệch giữa PyTorch và ONNX; chưa đo Android.

Code và báo cáo: [model-deployment-quantization-lab](https://github.com/HuyGoodBoy/model-deployment-quantization-lab/tree/main/daily/2026-10-19).

[Nghiên cứu chi tiết](RESEARCH.md) · [Kết quả](experiments/selective/results/regression.json) · [Kiểm chứng](experiments/selective/results/verification.json) · [Cách chạy](RUNNING.md)
"""
    (day / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    table = []
    for name in timing:
        row = timing[name]
        ap = quality.get(name)
        ap_text = f"{ap:.4f}" if ap is not None else "—"
        dev_ap = dev["models"][name]["sAP10"]["sAP10_subset_percent"]
        profile = diagnosis["profiles"].get(
            name, exports["candidates"].get(name, {}).get("profile")
        )
        ops = profile["unique_operators"] if profile else {}
        integer = ops.get("QLinearConv", 0) + ops.get("ConvInteger", 0)
        floating = ops.get("Conv", 0) + ops.get("FusedConv", 0)
        coverage = f"{integer}/{floating}" if profile else "0/41 (graph float)"
        table.append(
            f"| {name} | {dev_ap:.4f} | {ap_text} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {coverage} | {sizes[name]} |"
        )
    profiled = diagnosis["profiles"]["qdq_all"]["operator_mean_ms"]
    profiled_total = diagnosis["profiles"]["qdq_all"]["mean_model_run_ms"]
    boundaries = sum(
        profiled.get(op, 0) for op in ["QuantizeLinear", "DequantizeLinear", "Transpose"]
    )
    enabled, disabled = control["conditions"]["enabled"], control["conditions"]["disabled"]
    fusion_percent = (1 - enabled["median_ms"] / disabled["median_ms"]) * 100
    research = f"""# M-LSD: chủ động chọn vùng float, ngày 19/10

Câu hỏi: chủ động bỏ lượng tử hóa ở một cụm lớp có cải thiện chất lượng–latency so với QDQ ngày 16 không? Đây là thử nghiệm mới trong dự án, không tuyên bố phương pháp mới về học thuật. Thiết kế nằm ở [PROTOCOL.md](PROTOCOL.md).

## Điều đã xác nhận về fusion

QDQ baseline chạy 22 QLinearConv và 19 Conv float trên CPU ORT 1.30.0. Custom export ngày 15 chạy 41 ConvInteger, là artifact khác; không dùng câu “chưa fuse QDQ” để giải thích chung cả hai. Tắt toàn optimizer trên cùng QDQ cho 41 Conv float, nhưng phép so đó còn thay constant folding, layout và các fusion khác.

Đối chứng bổ sung giữ ALL và chỉ vô hiệu `QDQSelectorActionTransformer`: số QLinearConv giảm 22 → 0. Median của pass bật là **{enabled["median_ms"]:.3f} ms**, tắt là **{disabled["median_ms"]:.3f} ms**, p95 lần lượt {enabled["p95_ms"]:.3f}/{disabled["p95_ms"]:.3f} ms. Chênh median theo (1 − bật/tắt) là {fusion_percent:.2f}%. Đây là hiệu ứng của pass này trong cấu hình đo, không phải chi phí thuần của một kernel Conv và không bảo đảm fuse nhiều luôn nhanh. [Raw timing, trace và metric](experiments/selective/results/fusion_control.json).

Graph audit ghi nhận 19 Conv float gồm 3 Conv có Clip→Pad trước quantizer kế tiếp, 9 projection Conv có Add/residual hoặc nhiều consumer, và 7 Conv head có Relu→Resize/Concat/Add hoặc output Slice. Tất cả group=1. Đây là các pattern quan sát được, chưa chứng minh mỗi node không fuse do duy nhất một điều kiện. Nguồn ORT mô tả mẫu thay Conv bằng QLinearConv cần các DQ đầu vào và Q đầu ra; graph thực tế và kernel trace mới quyết định coverage.

## Chọn vùng và giữ đối chứng

Trên bốn ảnh development 0,5,10,15, từng Conv fake quant được so với Conv float cùng weight và **cùng input**, tránh quy sai số truyền từ trước thành độ nhạy riêng của lớp. Gom theo block, chọn tích thời gian Conv warm-profile × relative L2 cục bộ lớn nhất. Block được chọn: `{diagnosis["anchor"]}`. Heuristic này chỉ chọn vùng thử; không chứng minh đó là nguyên nhân duy nhất của sai số toàn mạng.

- A: bỏ lượng tử input/weight tại block16, 3 Conv; giữ output quantizer.
- B: mở rộng sang block15 và block16, tổng 5 Conv; giữ output quantizer.
- Phần còn lại giữ nguyên checkpoint QAT, weight, bias, scale/zero-point và decoder. Không train, không fit calibration mới.
- Teacher FP32 phân biệt với `qat_weights_fp32`: control sau dùng weight QAT và tắt toàn bộ fake quant, giúp phân biệt tác động weight với precision.

Kernel float không đồng nghĩa không bị lượng tử hóa: Conv float của baseline có thể nhận input/weight đã Q/DQ. A/B chủ động bỏ lượng tử ở vùng đã chọn. Cả A và B chạy 19 Conv integer +22 float nhưng không cùng precision semantics. A vẫn 40 DQ/19 Q như baseline; B còn 36 DQ/17 Q. Chỉ B giảm số boundary Q/DQ theo phép đếm này; số node không phải số byte chuyển hay tổng thời gian.

## Chất lượng và tốc độ

| Model | Dev sAP10 (%) | Hồi quy sAP10 (%) | Median ms | p95 ms | Conv integer/float | Bytes |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Benchmark cùng runtime, batch1, 4 intra/1 inter threads, sequential, spinning tắt. Chỉ một session tồn tại mỗi lượt, 5 vòng xoay/đảo thứ tự, warm-up16 +timed16 mỗi model/vòng, 80 timing/model; profiler tắt. Không gồm load/preprocess/decoder. Median từng vòng được lưu; B thấp hơn QDQ ở {round_wins}/5 vòng. Các vòng là đo lặp trên cùng máy, không phải 80 thí nghiệm độc lập. Chênh tốc độ quan sát chưa bảo đảm tổng quát sang máy khác.

Lựa chọn đã khóa: **{chosen}**, theo development và latency trước khi inference hồi quy. Giữ giới hạn giảm AP không quá 0,5 điểm và chỉ chọn A/B nếu median thấp hơn QDQ. Đây là lựa chọn nghiên cứu, không phải xác nhận qua release gate.

B tăng {delta:.4f} điểm sAP10 trên 54 ảnh hồi quy, bootstrap ảnh ghép cặp 1.000 lần seed20261019 cho CI95% [{ci[0]:.4f}; {ci[1]:.4f}] điểm. Tập này đã dùng ngày16: không phải holdout mới, CI chỉ mô tả tập đã đo. Cả A/B được báo đủ, không chọn lại sau khi xem hồi quy.

## Sai lệch export và profiler

Relative L2 ONNX so PyTorch **cùng policy**: A {exports["candidates"]["selective_a"]["export_parity"]["relative_l2"] * 100:.4f}%, B {parity_percent:.4f}%; cả hai vượt gate 2%. Chênh sAP10 export cũng được lưu riêng. Sai lệch do đổi precision so QAT gốc là phép đo khác, không dùng thay gate export. QAT-weights FP32 export relative L2 {exports["candidates"]["qat_weights_fp32"]["export_parity"]["relative_l2"] * 100:.6f}%.

Warm profile baseline: mean model_run {profiled_total:.3f} ms; Conv integer {profiled.get("QLinearConv", 0):.3f}, Conv float {profiled.get("FusedConv", 0) + profiled.get("Conv", 0):.3f}, Q/DQ/Transpose {boundaries:.3f} ms. Đây là thời gian **có overhead profiler**, không lấy thay benchmark hoặc kết luận mọi chênh latency đều do chuyển đổi. Trace chỉ dùng 8 model_run cuối sau 16 warm-up, loại warm-up/startup. Không dùng trace một lượt cold ngày16 để phân bổ bottleneck.

## Kiểm chứng và giới hạn

Verifier tái tính metric, bootstrap, median/p95 và lựa chọn; kiểm hash nguồn cũ không đổi, weight/bias/qparams ONNX, kernel/provider/dtype và graph đã tối ưu. Chạy lại 16 ảnh cho ba model mới và 54 ảnh cho model đã chọn, yêu cầu output khớp tuyệt đối; baseline FP32/QDQ trên hồi quy phải khớp output ngày16. `status=passed` xác nhận bằng chứng nhất quán; gate model vẫn thất bại.

Timing hôm nay khác các ngày trước; không trộn các số để quy khác biệt cho fusion. Không dùng kết quả CPU ORT1.30 để kết luận GPU ORT1.20.2 hay Android. Các JSON giữ hash script khi chạy; `source_archive.json` giữ bản source trước dọn lint/format để kiểm provenance, source hiện tại dùng tái lập.

Kết luận: B là phương án nghiên cứu tốt hơn QDQ baseline trên cấu hình đã đo: median giảm {speed_percent:.2f}%, AP hồi quy tăng {delta:.4f} điểm. Đổi precision vùng head và giảm Q/DQ diễn ra cùng nhau nên chưa cô lập đóng góp riêng từng yếu tố. Đối chứng weight QAT chạy float đạt {quality["qat_weights_fp32"]:.4f}% và {timing["qat_weights_fp32"]["median_ms"]:.3f} ms, tốt hơn B cả chất lượng lẫn latency; B chỉ cải thiện so với QDQ cũ, không phải cấu hình tốt nhất toàn bộ. Model B nhỏ hơn bản float ({sizes["selective_b"]} so với {sizes["qat_weights_fp32"]} bytes), nhưng chưa đo RAM runtime. Model vẫn chậm hơn FP32 và chưa đạt gate export; bước sau cần xử lý parity rồi kiểm tra dữ liệu độc lập hoặc thiết bị đích. Link source ORT main chỉ minh họa quy tắc được đọc ngày19/10; coverage và hiệu ứng pass của phiên bản1.30.0 được xác nhận bằng graph/trace thực tế đã lưu.

Nguồn: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html), [Conv QDQ rule](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc).
"""
    (day / "RESEARCH.md").write_text(research, encoding="utf-8", newline="\n")
    with (results / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "development_sAP10_percent",
                "regression_sAP10_percent",
                "median_ms",
                "p95_ms",
                "model_bytes",
                "model_sha256",
            ]
        )
        for name, row in timing.items():
            writer.writerow(
                [
                    name,
                    dev["models"][name]["sAP10"]["sAP10_subset_percent"],
                    quality.get(name),
                    row["median_ms"],
                    row["p95_ms"],
                    sizes[name],
                    row["model"]["sha256"],
                ]
            )
    print("WROTE", day / "REPORT.md")


if __name__ == "__main__":
    main()
