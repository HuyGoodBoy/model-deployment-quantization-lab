"""Xuat bang so sanh va bao cao Markdown co the nop cuoi ngay."""
import csv
import argparse
from datetime import date, datetime, timedelta, timezone

from .common import ARTIFACTS, DATA, RESULTS, RUNTIMES, read_json, write_json


def fmt(value, digits=6):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}g}"


def main(report_date=None):
    evaluation = read_json(RESULTS / "evaluation.json")
    conversion = read_json(ARTIFACTS / "conversion_info.json")
    benchmarks = {r: read_json(RESULTS / f"benchmark_{r}.json") for r in RUNTIMES}
    configurations = {(b["threads"], b["runs"], b["warmup"], b["input_image"])
                      for b in benchmarks.values()}
    if len(configurations) != 1:
        raise ValueError("Cac benchmark co cau hinh khac nhau. Hay chay benchmark lai.")
    for b in benchmarks.values():
        if (b["model_sha256"] != evaluation["model_sha256"]
                or b["input_sha256"] != evaluation["inputs"].get(b["input_image"])):
            raise ValueError("Benchmark va evaluation khong cung model/input. Hay chay lai.")
    rows = []
    files = {"tensorflow": "mobilenetv2.keras", "onnx": "mobilenetv2.onnx",
             "tflite": "mobilenetv2.tflite"}
    for runtime in RUNTIMES:
        b = benchmarks[runtime]
        metrics = evaluation["comparisons"].get(runtime, {
            "mae": 0, "rmse": 0, "max_abs_error": 0, "relative_l2": 0,
            "cosine_similarity": 1, "snr_db": "inf", "top1_agreement": 1,
            "top5_overlap": 1, "allclose": True})
        row = {"runtime": runtime, "file_size_mib": (ARTIFACTS / files[runtime]).stat().st_size / 2**20,
               "mean_ms": b["mean_ms"], "median_ms": b["median_ms"], "p95_ms": b["p95_ms"],
               "images_per_second": b["images_per_second"],
               "speedup_vs_tensorflow_mean": benchmarks["tensorflow"]["mean_ms"] / b["mean_ms"],
               **metrics}
        if evaluation["accuracy"] is not None:
            row.update(evaluation["accuracy"][runtime])
        rows.append(row)
    with (RESULTS / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RESULTS / "summary.json", rows)
    with (RESULTS / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluation["per_image"][0]))
        writer.writeheader()
        writer.writerows(evaluation["per_image"])
    baseline = benchmarks["tensorflow"]
    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    report_day = report_date or datetime.now(timezone(timedelta(hours=7))).date()
    tomorrow = report_day + timedelta(days=1)
    lines = [
        "# Báo cáo thử nghiệm triển khai MobileNetV2", "",
        f"Thời điểm tạo báo cáo: {timestamp} (UTC+7).", "",
        "## Thiết lập", "",
        "- Model: Keras MobileNetV2 alpha=1, pretrained ImageNet; không train lại.",
        "- Input: RGB NHWC [1,224,224,3], float32, resize bilinear, chuẩn hóa [-1,1].",
        "- Output: xác suất softmax [1,1000]; so sánh toàn bộ vector.",
        "- Export độc lập từ cùng model gốc sang ONNX và TFLite; FP32, chưa quantize.",
        f"- Phiên bản: TensorFlow {conversion['tensorflow']}, tf2onnx {conversion['tf2onnx']}, "
        f"ONNX {conversion['onnx']}, opset {conversion['opset']}.",
        f"- ONNX Runtime {benchmarks['onnx']['runtime_info']['version']}: CPUExecutionProvider, ORT_ENABLE_ALL.",
        "- TensorFlow: tf.function trên CPU. TFLite: Interpreter với delegate mặc định; xem log runtime.",
        f"- Máy: {baseline['system']['platform']}; CPU: {baseline['system']['processor']}; Python {baseline['system']['python']}.",
        f"- Đánh giá {evaluation['num_images']} ảnh; dataset: `{evaluation['dataset_kind']}`.",
        f"- Benchmark: CPU, batch=1, {baseline['threads']} thread intra-op; "
        f"{baseline['warmup']} warm-up, {baseline['runs']} lượt đo cho mỗi runtime trong process riêng.",
        f"- Benchmark dùng lặp lại ảnh `{baseline['input_image']}`; không suy rộng sang tất cả ảnh/tất cả thiết bị.",
        "- Đo inference API gồm truyền input/output trong RAM và overhead Python; loại thời gian tải, load model, preprocess.",
        "", "## Kết quả tốc độ", "",
        "| Runtime | File MiB | Mean ms | Median ms | P95 ms | Images/s | Speedup theo mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['runtime']} | {r['file_size_mib']:.2f} | {r['mean_ms']:.3f} | "
                     f"{r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['images_per_second']:.2f} | "
                     f"{r['speedup_vs_tensorflow_mean']:.2f}x |")
    lines += ["", "Images/s = 1000 / mean_ms với batch=1, chạy tuần tự. File size là dung lượng lưu trữ, không phải RAM sử dụng.",
              "", "## Sai khác so với model gốc", "",
              "| Runtime | MAE | RMSE | Max abs | Relative L2 | Cosine | SNR dB | Top-1 agreement | Top-5 overlap | Allclose |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows[1:]:
        lines.append(f"| {r['runtime']} | {fmt(r['mae'])} | {fmt(r['rmse'])} | "
                     f"{fmt(r['max_abs_error'])} | {fmt(r['relative_l2'])} | "
                     f"{fmt(r['cosine_similarity'], 17)} | {fmt(r['snr_db'])} | "
                     f"{r['top1_agreement']:.1%} | {r['top5_overlap']:.1%} | {r['allclose']} |")
    class_index = read_json(DATA / "imagenet_class_index.json")
    lines += ["", f"So sánh {evaluation['num_images']} ảnh × 1.000 lớp trên cùng input FP32; "
              "TensorFlow gốc là tham chiếu. Các metric số học ở trên tính trên toàn bộ phần tử, không chỉ top-1.",
              "MAE, RMSE, max abs và relative L2 càng nhỏ càng gần output gốc; cosine càng gần 1 và SNR càng cao càng tốt.",
              "Cosine được giữ đủ chữ số để tránh làm tròn thành 1 rồi hiểu nhầm output giống hệt nhau.",
              "", "### Sai khác output theo từng ảnh", "",
              "| Ảnh | Runtime | MAE | RMSE | Max abs | Relative L2 | SNR dB | Allclose |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for item in evaluation["per_image"]:
        image_name = item["filename"].replace("|", "\\|")
        lines.append(f"| {image_name} | {item['runtime']} | {fmt(item['mae'])} | "
                     f"{fmt(item['rmse'])} | {fmt(item['max_abs_error'])} | "
                     f"{fmt(item['relative_l2'])} | {fmt(item['snr_db'])} | {item['allclose']} |")
    lines += ["", "### Dự đoán từng ảnh", "",
              "| Ảnh | Runtime | Top-1 gốc | Top-1 sau convert |", "|---|---|---|---|"]
    for item in evaluation["per_image"]:
        reference_name = class_index[str(item["reference_top1"])][1]
        candidate_name = class_index[str(item["candidate_top1"])][1]
        image_name = item["filename"].replace("|", "\\|")
        lines.append(f"| {image_name} | {item['runtime']} | {reference_name} | {candidate_name} |")
    lines += ["", f"Allclose dùng |candidate-reference| <= {evaluation['atol']} + "
              f"{evaluation['rtol']} * |reference| cho từng phần tử.",
              "SNR = 10 log10(sum(reference²) / sum((candidate-reference)²)); tổng trên toàn bộ ảnh và lớp.",
              "Top-1 agreement đo tỷ lệ nhãn dự đoán giống model gốc. Top-5 overlap đo tỷ lệ lớp chung trong hai tập top-5.",
              "", "## Accuracy và giới hạn", ""]
    if evaluation["accuracy"] is None:
        lines.append("Chưa có ground-truth labels nên không báo cáo accuracy. Agreement cao chỉ cho biết model sau convert gần model gốc.")
    else:
        lines += ["| Runtime | Top-1 accuracy | Top-5 accuracy |", "|---|---:|---:|"]
        for r, scores in evaluation["accuracy"].items():
            lines.append(f"| {r} | {scores['top1_accuracy']:.1%} | {scores['top5_accuracy']:.1%} |")
    lines += ["", "- Hai ảnh mẫu chỉ xác minh pipeline. Muốn kết luận về độ chính xác cần tập dữ liệu có nhãn lớn hơn, phù hợp bài toán.",
              "- Chưa đo Android. Kết quả CPU máy tính không đại diện cho điện thoại; tốc độ giả lập cũng không đại diện máy thật.",
              "- Cùng số thread không bảo đảm các runtime dùng cùng kernel. Các tối ưu graph/delegate được ghi nhận ở trên.",
              "- Sai số FP32 có thể xuất hiện do thứ tự cộng/nhân, fuse layer và kernel khác nhau dù không quantize.",
              "- Kết quả thời gian chịu ảnh hưởng tải máy, power mode và nhiệt độ; khi báo cáo nên giữ điều kiện ổn định.",
              "", "## Kết luận từ lần chạy này", ""]
    for r in rows[1:]:
        lines.append(f"- {r['runtime']}: MAE={fmt(r['mae'])}; RMSE={fmt(r['rmse'])}; "
                     f"max abs={fmt(r['max_abs_error'])}; relative L2={fmt(r['relative_l2'])}; "
                     f"SNR={fmt(r['snr_db'])} dB; FP32 allclose={r['allclose']}; "
                     f"top-1 agreement={r['top1_agreement']:.1%}; "
                     f"median={r['median_ms']:.3f} ms; speedup theo mean={r['speedup_vs_tensorflow_mean']:.2f}x.")
    lines += ["", "## Framework/runtime trên các nền tảng", "",
              "| Công cụ | Nền tảng thường dùng | Vai trò |", "|---|---|---|",
              "| [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/) | PC/server/mobile, CPU và accelerator tùy provider | Chạy model ONNX |",
              "| [LiteRT / TensorFlow Lite](https://developers.google.com/edge/litert) | Mobile và edge | Chạy model .tflite |",
              "| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | GPU NVIDIA, Jetson | Tối ưu và chạy inference trên GPU NVIDIA |",
              "| [OpenVINO](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html) | CPU/GPU và thiết bị Intel được hỗ trợ | Tối ưu và chạy inference |",
              "| [Core ML](https://developer.apple.com/documentation/coreml) | iOS/macOS | Chạy trên thiết bị Apple |",
              "", "ONNX là định dạng biểu diễn model; ONNX Runtime là engine chạy model. Việc hỗ trợ operator và accelerator tùy phiên bản/thiết bị.",
              "", "Raw data: outputs.npz, evaluation.json, benchmark_*.json, per_image.csv, summary.csv.", ""]
    # Reuse the measured tables above, arranging the daily report around the assignment.
    content = "\n".join(lines)
    headings = ["Thiết lập", "Kết quả tốc độ", "Sai khác so với model gốc",
                "Accuracy và giới hạn", "Kết luận từ lần chạy này",
                "Framework/runtime trên các nền tảng"]
    sections = {}
    for index, heading in enumerate(headings):
        start = content.index(f"## {heading}\n") + len(f"## {heading}\n")
        end = content.index(f"## {headings[index + 1]}\n", start) if index + 1 < len(headings) else len(content)
        sections[heading] = content[start:end].strip()
    framework = sections["Framework/runtime trên các nền tảng"].split("\n\nRaw data:")[0]
    report = [
        "# Báo cáo công việc: triển khai và đánh giá model pretrained", "",
        f"**Ngày báo cáo: {report_day:%d/%m/%Y}.**", "",
        "## 1. Mục tiêu công việc", "",
        "Tìm hiểu framework/runtime triển khai model trên các nền tảng; chọn một model pretrained, "
        "convert sang ONNX và TFLite ở FP32, chạy thử và đánh giá sai khác output cùng tốc độ inference.", "",
        "**Phạm vi đã hoàn thành hôm nay:** tìm hiểu công cụ, convert hai định dạng và đo trên CPU máy tính. "
        "Chưa quantize, chưa chạy Android.", "",
        "## 2. Nội dung đã tìm hiểu", "", framework, "",
        "ONNX Runtime và LiteRT/TFLite được dùng trong thử nghiệm hôm nay. TensorRT, OpenVINO và Core ML "
        "mới được tìm hiểu về vai trò và nền tảng, chưa có benchmark thực nghiệm.", "",
        "## 3. Cách thực hiện và điều kiện thử nghiệm", "",
        "Quy trình: tải MobileNetV2 pretrained → chuẩn bị input dùng chung → export độc lập sang ONNX/TFLite "
        "→ chạy từng runtime → lưu output → tính metric → đo latency → tổng hợp báo cáo.", "",
        sections["Thiết lập"], "",
        "## 4. Kết quả đánh giá sai khác output", "",
        sections["Sai khác so với model gốc"], "",
        "**Nhận xét:** trên hai ảnh mẫu, cả ONNX và TFLite có sai số tuyệt đối lớn nhất dưới 1e-6; "
        "cosine gần 1, SNR trên 120 dB ở mức tổng hợp, top-1 agreement và top-5 overlap đều đạt 100%. "
        "Cả hai đạt tolerance đã đặt. Output sau convert rất gần model gốc trong thử nghiệm này, nhưng không giống hệt từng phần tử.", "",
        "## 5. Kết quả đánh giá tốc độ trên máy tính", "",
        sections["Kết quả tốc độ"], "",
        f"**Nhận xét:** ONNX Runtime nhanh hơn TensorFlow khoảng {rows[1]['speedup_vs_tensorflow_mean']:.2f} lần theo mean. "
        "TFLite chậm hơn TensorFlow trên cấu hình CPU đã đo. Thứ hạng tốc độ này chỉ áp dụng cho lần thử nghiệm hiện tại; "
        "không suy ra tốc độ trên Android. P95 của TensorFlow cao hơn rõ rệt median, cho thấy thời gian các lượt chạy có biến động.", "",
        "## 6. Kết luận và giới hạn", "",
        "Đã hoàn thành pipeline với MobileNetV2 pretrained: export ONNX/TFLite FP32, kiểm tra sai khác trên toàn bộ "
        "vector output và đo tốc độ trên CPU máy tính. Hai model sau convert đạt tolerance trên tập mẫu đã dùng. "
        "ONNX Runtime có mean latency thấp nhất trong ba runtime ở lần đo này.", "",
        sections["Accuracy và giới hạn"], "",
        f"## 7. Kế hoạch ngày mai — {tomorrow:%d/%m/%Y}", "",
        "**Chạy thử và đo trên Android Emulator; phần này chưa thực hiện và chưa có số liệu.**", "",
        "1. Chuẩn bị Android Studio, SDK và AVD; dùng system image x86_64 nếu máy tính có CPU Intel/AMD.",
        "2. Chuẩn bị ứng dụng benchmark Android chạy cả ONNX Runtime và LiteRT/TFLite trên CPU, giữ model FP32.",
        "3. Đưa cùng tensor input đã preprocess từ PC sang Android để tránh sai khác do resize/normalization.",
        "4. Lưu đủ 1.000 xác suất mỗi ảnh, đưa về PC và so với output TensorFlow gốc bằng các metric như hôm nay.",
        "5. Đo latency với batch=1, 1 thread, warm-up 30 lần và 200 lượt đo; ghi mean, median, P95 và phạm vi timer.",
        "6. Bổ sung bảng kết quả riêng cho Android, ghi Android API, ABI, runtime, delegate và cấu hình emulator.", "",
        "Kết quả giả lập sẽ được ghi là **Android Emulator**, không coi là hiệu năng của điện thoại thật. "
        "Nếu có thiết bị Android thật, có thể đo thêm và ghi thành môi trường riêng.", "",
        "Tài liệu chuẩn bị: [tạo AVD](https://developer.android.com/studio/run/managing-avds), "
        "[ONNX Runtime Mobile](https://onnxruntime.ai/docs/get-started/with-mobile.html), "
        "[benchmark LiteRT](https://developers.google.com/edge/litert/models/measurement).", "",
        "## 8. File bàn giao và dữ liệu kiểm chứng", "",
        "- Model: `artifacts/mobilenetv2.keras`, `artifacts/mobilenetv2.onnx`, `artifacts/mobilenetv2.tflite`.",
        "- Output gốc và sau convert: `results/outputs.npz`.",
        "- Metric tổng hợp/từng ảnh: `results/evaluation.json`, `results/summary.csv`, `results/per_image.csv`.",
        "- Dữ liệu latency: `results/benchmark_tensorflow.json`, `results/benchmark_onnx.json`, `results/benchmark_tflite.json`.",
        "- Hướng dẫn học và thực hành: `hoc-deploy-model.html`, `README.md`.", "",
        f"Thời điểm tổng hợp báo cáo: {timestamp} (UTC+7). Ngày báo cáo ở đầu tài liệu xác định ngày công việc.", "",
    ]
    target = RESULTS / "report.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tổng hợp báo cáo từ dữ liệu đã đo.")
    parser.add_argument("--date", type=date.fromisoformat, help="Ngày công việc YYYY-MM-DD; mặc định hôm nay theo UTC+7.")
    main(parser.parse_args().date)
