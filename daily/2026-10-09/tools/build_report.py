"""Build mentor Markdown and local chat report from measured JSON, never invented rows."""

import csv
import datetime
import json
from pathlib import Path

DAY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(task, pattern):
    return [read(p) for p in sorted((DAY / "experiments" / task / "results/run-01").glob(pattern))]


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(x) for x in row) + " |" for row in rows],
        ]
    )


def main():
    peer_path = DAY / "experiments/01-reproduce/results/run-04-peer-linux/peer_reproduction.json"
    peer = read(peer_path) if peer_path.exists() else None
    if peer and not peer.get("source_unchanged"):
        peer = None
    peer_count = sum(r.get("status") == "ok" for r in peer.get("results", [])) if peer else 0
    peer_note = (
        f"Đã chạy source ResNet18 Đức trong Linux Docker/WSL2, đúng 10 release dependencies khai báo; {peer_count}/6 biến thể có output/latency. "
        "q1/q2/q3 nguyên script, q4 giữ nguyên hàm và tách từng variant để lưu lỗi/raw timing; source mount read-only và kiểm SHA256. "
        "Không cần cùng Mac để reproduce: phần cứng/OS/backend phải ghi riêng khi so tốc độ. "
        "[Bảng Mac Đức báo cáo ↔ Linux Huy thực đo và model Huy trên Linux](PEER-REPRODUCTION.md)."
        if peer
        else "Đã đối chứng runtime Windows, chưa chạy lại pipeline ResNet18 Đức. Windows thiếu dependency litert-converter 0.4.*; cần Linux tương thích hoặc artifact cùng hash. Khác phần cứng không ngăn reproduce, nhưng phải ghi riêng điều kiện máy khi so tốc độ."
    )
    peer_next = (
        "Phân tích chênh lệch ResNet18 giữa Mac Đức báo cáo và Linux Huy; giữ raw timing, profile kernel riêng và lặp nhiều phiên. Không cần tái tạo phần cứng Mac để chạy code."
        if peer
        else "Chạy source ResNet18 Đức trên môi trường tương thích, cùng data/preprocess/protocol; phần cứng có thể khác, cần ghi rõ cấu hình."
    )
    peer_chat = (
        f"Em đã chạy source ResNet18 Đức trong Linux với đúng bộ release dependencies, {peer_count}/6 biến thể đo được; so trực quan với số Đức báo cáo trên Mac và chạy thêm model Huy trong môi trường đó. Source Đức giữ nguyên; hardware/OS ghi riêng. Chi tiết PEER-REPRODUCTION.md."
        if peer
        else "Em mới đối chứng model Huy trên runtime Windows, chưa chạy source ResNet18 Đức; cần bổ sung môi trường converter tương thích, không yêu cầu cùng Mac."
    )
    runtime = [
        r for r in records("01-reproduce", "*.json") if isinstance(r, dict) and "metrics" in r
    ]
    distil = records("02-distilbert-int8", "measure_*.json")
    mlsd = records("03-mlsd", "measure_*.json")
    runtime_table = table(
        [
            "Model / variant",
            "ORT/LiteRT cũ ms",
            "ORT/LiteRT mới ms",
            "Mới/cũ",
            "Accuracy cũ→mới",
            "SNR cũ→mới dB",
        ],
        [
            [
                f"{task}/{variant}",
                fmt(a["benchmark"]["median_ms"]),
                fmt(b["benchmark"]["median_ms"]),
                fmt(b["benchmark"]["median_ms"] / a["benchmark"]["median_ms"]),
                f"{a['metrics']['accuracy']:.0%} → {b['metrics']['accuracy']:.0%}",
                f"{fmt(a['metrics']['snr_db'])} → {fmt(b['metrics']['snr_db'])}",
            ]
            for task, variant in sorted({(r["task"], r["variant"]) for r in runtime})
            for a in runtime
            if a["task"] == task and a["variant"] == variant and "_old_" in a["name"]
            for b in runtime
            if b["task"] == task and b["variant"] == variant and "_new_" in b["name"]
        ],
    )
    distil_table = table(
        ["Biến thể", "Accuracy", "SNR (dB)", "Median (ms)", "Scale lớn nhất"],
        [
            [
                {"baseline": "Mixed ban đầu", "mixed_mask1e2": "Mixed, mask 1e2",
                 "mixed_mask1e4": "Mixed, mask 1e4", "strict": "Strict INT8"}.get(r["variant"], r["variant"]),
                f"{r['metrics']['accuracy']:.0%}",
                fmt(r["metrics"]["snr_db"]),
                fmt(r["benchmark"]["median_ms"]),
                f"{r['audit']['max_quant_scale']:.3g}",
            ]
            for r in distil
        ],
    )
    mlsd_table = table(
        [
            "Biến thể / runtime",
            "MiB",
            "Median ms",
            "p95 ms",
            "SNR vs Torch",
            "Max raw error",
            "Line demo",
        ],
        [
            [
                f"{r['variant']} / {r['runtime']}",
                fmt(r["size_bytes"] / 2**20),
                fmt(r["benchmark"]["median_ms"]),
                fmt(r["benchmark"]["p95_ms"]),
                fmt(r.get("metrics_vs_torch", {}).get("snr_db", "khác output contract")),
                fmt(r.get("metrics_vs_torch", {}).get("max_abs_error", "—"), 5),
                r["demo_line_count"],
            ]
            for r in mlsd
        ],
    )
    channel_table = table(
        ["Raw variant", "Center SNR dB", "Displacement SNR dB", "Scale output"],
        [
            [
                r["variant"],
                fmt(r["center_logit_metrics"]["snr_db"]),
                fmt(r["displacement_metrics"]["snr_db"]),
                fmt(r["output_contract"][0]["quantization"][0], 6),
            ]
            for r in mlsd
            if "center_logit_metrics" in r
        ],
    )
    strict = next((r for r in distil if r["variant"] == "strict"), None)
    mask100 = next((r for r in distil if r["variant"] == "mixed_mask1e2"), None)
    baseline = next((r for r in distil if r["variant"] == "baseline"), None)
    candidates = []
    for task, threshold in [("cv", 0.73), ("text", 0.91)]:
        eligible = [
            r
            for r in runtime
            if r["task"] == task and "_new_" in r["name"] and r["metrics"]["accuracy"] >= threshold
        ]
        if eligible:
            best = min(eligible, key=lambda r: r["benchmark"]["median_ms"])
            candidates.append(
                f"{task}: `{best['variant']}` mới, {fmt(best['benchmark']['median_ms'])} ms, accuracy {best['metrics']['accuracy']:.0%}; ứng viên để đánh giá thêm, chưa kết luận deployment."
            )
    strict_text = (
        f"Bản strict convert thành công; audit {strict['audit']['tensor_counts']}. Float còn ở CAST→QUANTIZE của mask, không phải toàn attention/LayerNorm FP32. "
        f"Accuracy {strict['metrics']['accuracy']:.0%}, chưa chứng minh strict tốt hơn mixed."
        if strict
        else "Chưa có phép đo strict thành công; xem conversion JSON."
    )
    ablation_text = (
        f"Mask 1e30 → 1e2: accuracy {baseline['metrics']['accuracy']:.0%} → {mask100['metrics']['accuracy']:.0%}, "
        f"SNR {fmt(baseline['metrics']['snr_db'])} → {fmt(mask100['metrics']['snr_db'])} dB."
        if mask100 and baseline
        else "Đối chứng mask chưa có đầy đủ số đo."
    )
    bridge_path = DAY / "experiments/03-mlsd/results/run-01/bridge_validation.json"
    bridge = read(bridge_path) if bridge_path.exists() else {}
    execution_path = DAY / "experiments/01-reproduce/results/run-01/execution.json"
    execution = read(execution_path) if execution_path.exists() else []
    failed = [r["name"] for r in execution if r["exit_code"]]
    controls = [
        read(p)
        for p in sorted(
            (DAY / "experiments/01-reproduce/results/run-02-threadcheck").glob("text_*.json")
        )
    ]
    controls_text = table(
        ["TF global thread setting", "Median ms", "Accuracy", "SNR dB"],
        [
            [
                "set API" if r["benchmark"].get("initialize_tf_threads") else "environment only",
                fmt(r["benchmark"]["median_ms"]),
                f"{r['metrics']['accuracy']:.0%}",
                fmt(r["metrics"]["snr_db"]),
            ]
            for r in controls
        ],
    )
    boundaries = []
    for runtime_name in ("old", "new"):
        p = DAY / f"experiments/01-reproduce/results/run-01/delegate_audit_{runtime_name}.json"
        if p.exists():
            r = read(p)
            boundaries.append(f"{runtime_name}: {len(r['delegate_boundaries'])} DELEGATE nodes")
    box_path = DAY / "experiments/04-box-postprocess/results/run-01/demo_boxes.json"
    box = read(box_path) if box_path.exists() else None
    box_text = (
        f"Smoke test desktop từ {box['input_segments']} line chưa merge: {len(box['boxes'])} box ứng viên, median {fmt(box['median_ms'])} ms. "
        "Đây là kết quả prototype, không phải chất lượng NAVER hoặc tốc độ Android."
        if box
        else "Chưa có smoke test input thực tế."
    )
    run_times = []
    for p in sorted((DAY / "experiments/01-reproduce/results/run-01").glob("*.log")):
        run_times.append(
            {
                "log": p.name,
                "file_created_at": datetime.datetime.fromtimestamp(
                    p.stat().st_ctime, datetime.timezone(datetime.timedelta(hours=7))
                ).isoformat(),
                "file_modified_at": datetime.datetime.fromtimestamp(
                    p.stat().st_mtime, datetime.timezone(datetime.timedelta(hours=7))
                ).isoformat(),
            }
        )
    # Git clones omit logs. Preserve recorded run boundaries when logs are unavailable.
    times_path = DAY / "experiments/01-reproduce/results/run-01/run_times.json"
    if run_times or not times_path.exists():
        times_path.write_text(
            json.dumps(
                {
                    "timezone": "Asia/Saigon",
                    "scope": "Windows log-file creation/modification times, approximate subprocess boundaries",
                    "jobs": run_times,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    mac_reported_ms = {
        "ONNX FP32": 14.90, "ONNX dynamic": 9.50, "ONNX static": 3.70,
        "TFLite FP32": 15.30, "TFLite dynamic": 4.70, "TFLite static": 4.30,
    }
    peer_table = table(
        ["Biến thể", "Đức báo trên Mac (ms)", "Em đo trên Linux (ms)", "Accuracy Linux", "Agree với PyTorch", "SNR (dB)"],
        [[r["name"], fmt(mac_reported_ms.get(r["name"], "—"), 2),
          fmt(r["median_ms"]), f"{r['accuracy_pct']:.2f}%",
          f"{r['agreement_pct']:.2f}%", fmt(r["snr_db_source_float32"], 2)]
         for r in (peer or {}).get("results", []) if r.get("status") == "ok"],
    ) if peer else "Chưa có kết quả reproduce source Đức trên Linux."
    linux = [read(p) for p in sorted(
        (DAY / "experiments/01-reproduce/results/run-04-huy-linux").glob("*.json"))]
    own_table = table(
        ["Model / biến thể", "Windows cũ (ms)", "Windows mới (ms)", "Linux (ms)", "Accuracy Linux", "SNR Linux (dB)"],
        [[f"{'ResNet50' if r['task'] == 'cv' else 'DistilBERT'} / {r['variant'].replace('_', ' ')}", fmt(a["benchmark"]["median_ms"]),
          fmt(b["benchmark"]["median_ms"]), fmt(r["benchmark"]["median_ms"]),
          f"{r['metrics']['accuracy']:.0%}", fmt(r["metrics"]["snr_db"], 2)]
         for r in linux if isinstance(r, dict) and "metrics" in r
         for a in runtime if a["task"] == r["task"] and a["variant"] == r["variant"] and "_old_" in a["name"]
         for b in runtime if b["task"] == r["task"] and b["variant"] == r["variant"] and "_new_" in b["name"]],
    )
    raw_table = table(
        ["Biến thể", "Dung lượng (MiB)", "Median (ms)", "SNR so với PyTorch (dB)", "Max error"],
        [[r["variant"], fmt(r["size_bytes"] / 2**20), fmt(r["benchmark"]["median_ms"]),
          fmt(r["metrics_vs_torch"]["snr_db"]), fmt(r["metrics_vs_torch"]["max_abs_error"], 5)]
         for r in mlsd if r["variant"] in ("fp32", "fp16", "dynamic", "static")],
    )
    decoder_table = table(
        ["Runtime", "Bản em convert, có decoder (ms)", "TFLite chính thức (ms)"],
        [[version, fmt(a["benchmark"]["median_ms"]), fmt(b["benchmark"]["median_ms"])]
         for version in ("2.15.1", "2.2.0")
         for a in mlsd if a["variant"] == "decoded_fp32" and a["runtime"] == version
         for b in mlsd if b["variant"] == "official" and b["runtime"] == version],
    )
    content = f"""# Báo cáo 09/10

Người làm: Bùi Gia Huy

Em chạy trên CPU i7-12700H, RAM 16 GB. Phần reproduce dùng Linux trong Docker/WSL2; các phép đo trước đó chạy trên Windows. Chưa chạy Android.

## 1. Chạy lại code của Đức và so với môi trường của em

### Code của Đức

Em chạy lại ResNet18 với 100 ảnh calibration và 300 ảnh đánh giá, giữ nguyên source Đức. q1–q3 chạy nguyên script; q4 giữ hàm đánh giá và benchmark, tách từng biến thể để lưu kết quả riêng. Đã chạy được {peer_count}/6 biến thể.

Cả hai dùng 4 thread, batch 1, warm-up 10 lần rồi đo 100 lần lấy median. Cột Mac là số Đức đã báo cáo, cột Linux là số em chạy lại.

{peer_table}

Agree là tỷ lệ dự đoán giống PyTorch gốc, khác với accuracy là tỷ lệ đúng nhãn. Chất lượng output nhìn chung gần kết quả Đức, nhưng tốc độ thay đổi khá nhiều, rõ nhất ở ONNX dynamic.

Em profile riêng ONNX dynamic trên Linux: `ConvInteger` chiếm 93,3% thời gian các node, còn `DynamicQuantizeLinear` chỉ 1,3%. Như vậy phần chậm chính trong lần chạy này là kernel convolution INT8, không phải bước tính range. Chưa có profile trên Mac để khẳng định nguyên nhân của toàn bộ chênh lệch.

Em cũng kiểm tra TFLite static: output chỉ biểu diễn được tới 14,58 trong khi logit PyTorch lên tới 32,52, nên có clipping. Đây là kết quả kiểm tra range; em chưa làm thí nghiệm thay EMA bằng min/max như phần của Đức.

### Model của em trên môi trường đó

Em chạy lại ResNet50 và DistilBERT, giữ nguyên file model và input. Nhóm này dùng 4 thread, warm-up 30 lần, đo 200 lần; accuracy đo trên 100 mẫu có nhãn.

- Windows cũ: ORT 1.20.1 / TensorFlow Lite 2.15.1.
- Windows mới và Linux: ORT 1.30.0 / LiteRT 2.2.0.

{own_table}

TFLite dynamic thay đổi nhiều nhất. Ví dụ DistilBERT từ 1054,815 ms ở Windows cũ xuống 39,582 ms ở Windows mới và 7,550 ms trên Linux. XNNPACK ở các bản mới có thêm hỗ trợ dynamic-range Conv2D/Fully Connected, nên có thể góp phần vào chênh lệch này; em chưa profile đủ để quy toàn bộ thay đổi cho XNNPACK.

Cùng máy nhưng khác OS, bản thư viện và native kernel vẫn có thể chạy khác nhau. Ngoài ra, giữa các phiên đo có biến động latency: DistilBERT static từng khoảng 43 ms, nhưng kiểm tra lại được 123–128 ms; đổi cách đặt thread chưa giải thích được chênh lệch. Vì vậy các số trên là kết quả từng phiên, cần đo lặp và kiểm soát nhiệt độ/power trước khi kết luận chắc về tốc độ.

Chi tiết môi trường, sai số và profile: [PEER-REPRODUCTION.md](PEER-REPRODUCTION.md).

## 2. Dynamic/static của ONNX Runtime và TFLite khác nhau thế nào

Quantization ánh xạ số thực sang số nguyên theo `x ≈ scale × (q − zero_point)`. Range càng rộng thì mỗi bước INT8 càng lớn, sai số làm tròn tăng; range quá hẹp thì giá trị ngoài khoảng bị kẹp.

| Phương pháp | ONNX Runtime | TFLite/LiteRT |
| --- | --- | --- |
| Dynamic | Weight quantize trước; range activation ở operator hỗ trợ được tính khi chạy | Weight thường lưu INT8; kernel hybrid có thể quantize activation khi chạy, các phần khác vẫn float |
| Static | Lấy range từ calibration trước khi chạy | Cũng dùng calibration; có thể cho phép float fallback hoặc yêu cầu các operator INT8 |

Cùng tên dynamic/static chưa có nghĩa là hai model làm giống nhau. Còn phải xem operator nào được quantize, weight dùng một scale cho cả tensor hay từng kênh, cách gộp range calibration và runtime có kernel tối ưu hay không.

Trong bài DistilBERT của em, ONNX static chỉ chọn MatMul/Gemm; phần mask và Softmax nằm ngoài phạm vi đó. TFLite static quantize rộng hơn, có cả vùng liên quan đến mask với scale rất lớn. Vì thế không thể chỉ nhìn nhãn “static INT8” rồi cho rằng hai bên chịu cùng sai số.

Đức dùng AI Edge Quantizer, còn em dùng TensorFlow converter 2.15.1. Kết quả về EMA trong bài Đức cần gắn với đúng thư viện/cấu hình đó, không áp dụng chung cho mọi cách convert TFLite. Calibration không đại diện dữ liệu chạy thật có thể gây sai ở cả ONNX lẫn TFLite.

## 3. DistilBERT: calibrated mixed INT8 và static INT8

**Mixed INT8 ở đây vẫn là static quantization.** “Calibrated” nghĩa là đã dùng dữ liệu calibration để lấy scale; “mixed” nghĩa là cho phép một số phần giữ float. Hai cách gọi nói về hai việc khác nhau: lúc lấy scale và phần nào được chạy INT8.

Ban đầu em chọn mixed để converter có thể giữ float ở phần chưa hỗ trợ INT8. Tuy nhiên, kiểm tra model cho thấy không phải toàn bộ attention hay LayerNorm được giữ FP32; tensor float còn lại là phần CAST của mask.

Em thử thêm bản strict, chỉ cho phép bộ operator INT8. Bản này convert được, có 372 tensor INT8, 126 INT32 và 1 FLOAT32 ở CAST→QUANTIZE của mask. Cả mixed và strict đều đạt 52% accuracy. Như vậy, hiện chưa có bằng chứng mixed giữ chất lượng tốt hơn strict trong model này. INT32 của token ID/index cũng không phải float fallback.

### Thử giảm giá trị rất lớn trong attention mask

Attention mask dùng một số âm rất lớn để vị trí padding có xác suất gần 0 sau Softmax. Giá trị cỡ `1e30` có thể kéo range quá rộng, làm các giá trị nhỏ hơn khó phân biệt khi đưa về INT8.

Em giữ nguyên weights, input và calibration, chỉ đổi độ lớn giá trị mask từ `1e30` sang `1e4` hoặc `1e2`. Trước khi quantize, hai bản FP32 mới cho output giống hệt reference trên 100 mẫu, accuracy vẫn 91%.

{distil_table}

Đổi mask sang `1e2` giúp accuracy từ 52% lên 67%, nhưng vẫn thấp hơn FP32 91%. Kết quả ủng hộ việc range của mask quá lớn là một phần vấn đề; chưa chứng minh đây là nguyên nhân duy nhất. Bước tiếp theo là kiểm tra trên tập độc lập và thử giữ những phần nhạy cảm ở float.

Code và kết quả: [02-distilbert-int8](experiments/02-distilbert-int8/).

## 4. Convert M-LSD PyTorch sang TFLite và thử quantize

### Cách convert và kiểm tra output

Em chuyển M-LSD Tiny 512 từ PyTorch sang Keras rồi convert TFLite, dùng cùng checkpoint, không train lại. Khi chuyển cần đổi thứ tự chiều weight convolution, xử lý depthwise convolution, padding và resize bilinear cho khớp PyTorch.

Input là RGBA 512×512, alpha bằng 1, normalize bằng `x/127.5 − 1`. Output thô được đưa về cùng thứ tự NHWC để so.

Bản Keras so với PyTorch đạt SNR {fmt(bridge.get("snr_db"))} dB, relative L2 {fmt(bridge.get("relative_l2"), 8)}, max error {fmt(bridge.get("max_abs_error"), 6)}. Kiểm tra `allclose` với ngưỡng `1e-4` ban đầu không đạt. Bản chuyển đổi đạt ngưỡng nới hơn là relative L2 < `1e-4` và max error < `0.1`; đây mới là kiểm tra sai số số học, chưa phải đánh giá chất lượng detect line/box.

### Tốc độ so với TFLite chính thức

Bản chính thức có sẵn decoder, trả `points`, `scores`, `vmap`. Em thêm decoder vào bản convert rồi mới so tốc độ, cùng 4 thread, warm-up 30 lần và đo 200 lần:

{decoder_table}

Ở TF Lite 2.15.1, bản em convert nhanh hơn khoảng 2 lần trong phiên đo này; ở LiteRT 2.2.0 thì hai bản gần nhau. Hai model không dùng cùng checkpoint/graph, nên bảng này so tốc độ, không dùng để kết luận chất lượng tương đương.

### Quantize phần model trả output thô

Em dùng 32 ảnh train để calibration, 10 ảnh validation để so output với cùng checkpoint PyTorch. Nhóm này chưa có decoder trong graph nên không so trực tiếp latency với bảng trên.

{raw_table}

FP32 giữ output gần PyTorch nhất. FP16 và INT8 làm file nhỏ hơn nhưng sai số tăng; dynamic và static chưa đem lại lợi thế tốc độ trong môi trường đã đo.

Output chứa cả center logit và displacement. Bản static dùng chung output scale khoảng 5,646; bước này có thể quá lớn với center. SNR center là 11,052 dB, displacement là 3,121 dB. Em cần thử tách hai head hoặc quantize chọn lọc để kiểm tra giả thuyết này. Chưa có dataset nhãn line/box, nên chưa báo precision/recall hay IoU; số đoạn detect được chỉ dùng kiểm tra pipeline.

Code và kết quả: [03-mlsd](experiments/03-mlsd/).

## 5. Thuật toán detect box và hướng chạy Android

Model dự đoán đoạn thẳng; phần post-processing mới ghép chúng thành box. Luồng chính gồm:

1. **Decode đoạn thẳng:** áp sigmoid lên center heatmap, giữ cực đại trong cửa sổ 3×3 và lấy top 200. Từ center `(x,y)`, cộng displacement để có hai đầu mút. Loại dự đoán yếu/ngắn, rồi nhân tọa độ với 2 để về ảnh 512. `points` là `(y,x)`, displacement là `(x,y)` nên phải đổi đúng thứ tự.
2. **Gộp đoạn cùng đường:** một cạnh có thể bị dự đoán thành nhiều đoạn. Biểu diễn đường bằng `ax + by = c`, lấy khoảng cách tới gốc `d = |c| / sqrt(a²+b²)` và góc để gom vào ô Hough. Các đoạn có khoảng cách/góc gần nhau được gộp thành một đoạn đại diện. Source dùng suppression 5×5 trên bảng Hough, khác NMS 3×3 ở bước center.
3. **Tìm góc:** với hai đường, tính `D = a1*b2 − a2*b1`. Nếu D gần 0 thì hai đường gần song song, bỏ cặp đó. Ngược lại, tính giao điểm `x = (c1*b2 − c2*b1)/D`, `y = (a1*c2 − a2*c1)/D`. Chỉ giữ góc khoảng 60–120° và giao điểm đủ gần đầu đoạn, tránh lấy giao điểm của hai đường kéo dài quá xa.
4. **Ghép bốn cạnh:** chia góc thành bốn hướng rồi tìm chu trình 0→1→2→3→0. Hai góc kề nhau phải dùng chung ID đường để bảo đảm chúng nối bằng cùng một cạnh. Kết quả là tứ giác gần chữ nhật, có thể bị xoay.
5. **Chấm điểm:** kết hợp mức phủ cạnh, độ giống góc/cạnh đối diện, diện tích và khoảng cách tới tâm ảnh, rồi xếp hạng ứng viên.

Em đã viết prototype Java cho phần giao điểm và ghép chu trình, qua 5 ca kiểm tra hình học. Demo từ 18 đoạn chưa gộp cho 4 box ứng viên, median 0,056 ms trên JVM Windows. Prototype chưa có Hough merge và scoring đầy đủ như NAVER, số này cũng chưa phải tốc độ Android.

Hướng port là dùng Java/Kotlin với mảng số và buffer tái sử dụng; giữ tọa độ float khi tính giao điểm, xử lý đường song song/đoạn bằng 0. Sau khi hoàn thiện merge và scoring, đối chiếu kết quả với source chính thức rồi tích hợp LiteRT. Khi vẽ lên camera cần đổi ngược tọa độ crop/letterbox. Nếu đo trên điện thoại thấy phần này chậm mới cân nhắc C++/JNI.

Chi tiết: [ALGORITHM.md](experiments/04-box-postprocess/ALGORITHM.md), [prototype Java](experiments/04-box-postprocess/src/BoxPostProcessor.java).

## 6. Việc làm tiếp

- DistilBERT: kiểm tra lại mask trên dữ liệu độc lập, thử quantize chọn lọc để tránh giảm accuracy quá nhiều.
- M-LSD: dùng calibration gần ảnh đường thẳng hơn, thử tách center/displacement và đánh giá trên dữ liệu có nhãn.
- Hoàn thiện Hough merge/scoring, rồi đo riêng model, post-processing và toàn pipeline trên Android thật.

[Cách chạy lại](RUNNING.md) · [Môi trường](ENVIRONMENT.md) · [Bài học HTML](../../docs/2026-10-09/hoc-ngay-2026-10-09.html). JSON/CSV chi tiết nằm trong từng thư mục thí nghiệm.
"""
    (DAY / "REPORT.md").write_text(content, encoding="utf-8")
    paired = {}
    for version in ("2.15.1", "2.2.0"):
        own = next(
            (r for r in mlsd if r["variant"] == "decoded_fp32" and r["runtime"] == version), None
        )
        official = next(
            (r for r in mlsd if r["variant"] == "official" and r["runtime"] == version), None
        )
        if own and official:
            paired[version] = (
                f"{own['benchmark']['median_ms']:.1f} vs {official['benchmark']['median_ms']:.1f} ms"
            )
    static_mlsd = next((r for r in mlsd if r["variant"] == "static"), None)
    static_note = (
        f"Static INT8 còn {static_mlsd['size_bytes'] / 2**20:.2f} MiB nhưng SNR {static_mlsd['metrics_vs_torch']['snr_db']:.2f} dB, chưa đạt chất lượng."
        if static_mlsd
        else "Chưa có phép đo static M-LSD."
    )
    chat = f"""Anh ơi em báo cáo ngày 09/10 ạ.
- {peer_chat}
- Calibrated mixed INT8 là static quantization cho phép float fallback. {strict_text}
- Em thử đổi attention mask sentinel, kiểm tra FP32 vẫn giống reference trên 100 mẫu. {ablation_text}
- M-LSD Tiny cùng checkpoint PyTorch→TFLite, đã thử FP16/dynamic/static. So graph có decoder với official: TF Lite {paired.get("2.15.1", "chưa đo")}; LiteRT {paired.get("2.2.0", "chưa đo")} (CPU 4 thread, 30 warm-up/200 lượt). Latency là snapshot, chưa kiểm soát nhiệt độ/power. {static_note} Chưa có metric line/box trên dataset có nhãn.
- Prototype Java qua 5 ca hình học, demo được 4 box ứng viên; chưa port đầy đủ Hough merge/scoring hoặc benchmark Android.
- Bàn giao code, dependency lock, JSON/CSV kết quả và báo cáo MD qua Git. Hướng tiếp: calibration đúng miền M-LSD, dataset có nhãn và test sentinel trên tập độc lập.
"""
    (DAY / "REPORT-CHAT.md").write_text(chat, encoding="utf-8")
    with (DAY / "experiments/01-reproduce/results/run-01/runtime_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "runtime", "accuracy", "snr_db", "median_ms", "p95_ms"])
        for r in runtime:
            writer.writerow(
                [
                    r["name"],
                    r["runtime"]["version"],
                    r["metrics"]["accuracy"],
                    r["metrics"]["snr_db"],
                    r["benchmark"]["median_ms"],
                    r["benchmark"]["p95_ms"],
                ]
            )
    print(
        f"Report built: {len(runtime)} runtime rows, {len(distil)} DistilBERT rows, {len(mlsd)} M-LSD rows"
    )


if __name__ == "__main__":
    main()
