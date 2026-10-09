"""Render measured peer reproduction and hardware comparison into Markdown."""

import json
from pathlib import Path

DAY = Path(__file__).resolve().parents[1]
TASK = DAY / "experiments/01-reproduce"
ROOT = DAY.parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def source_rows():
    text = (ROOT / ".cache/duke-readonly/4-quantization/BAO_CAO.md").read_text(encoding="utf-8-sig")
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == 6 and cells[0].startswith(("ONNX ", "TFLite ")):
            rows[cells[0]] = {
                "size_mb": float(cells[1]),
                "accuracy": float(cells[2]),
                "agreement": float(cells[3]),
                "snr": float(cells[4]),
                "ms": float(cells[5]),
            }
    if len(rows) != 6:
        raise ValueError("Expected six rows in the pinned peer report.")
    return rows


def main():
    peer = read(TASK / "results/run-04-peer-linux/peer_reproduction.json")
    if not peer.get("source_unchanged") or len(peer["results"]) != 6:
        raise ValueError("Complete the measured run and source verification before publishing.")
    published = source_rows()
    measured = {r["name"]: r for r in peer["results"]}
    ok = [r for r in measured.values() if r["status"] == "ok"]
    good_stages = all(
        (
            any((s["name"] == n and s["returncode"] == 0 for s in peer["stages"]))
            for n in ("q1_prepare", "q2_quant_onnx", "q3_quant_tflite")
        )
    )
    if not good_stages:
        raise ValueError("Conversion stages must have succeeded before building the peer report")
    audit_path = TASK / "results/run-04-peer-linux/output_and_profile_audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    audit_md = ""
    if audit:
        static = measured["TFLite static"]
        ties = audit["ties"]["TFLite static"]
        dynamic_ops = audit["profiles"]["dynamic"]["ops"]
        static_ops = audit["profiles"]["static"]["ops"]
        integer_share = (
            dynamic_ops.get("ConvInteger", {}).get("fraction_of_node_event_time", 0) * 100
        )
        qdq_share = (
            sum(
                (
                    static_ops.get(k, {}).get("fraction_of_node_event_time", 0)
                    for k in ["QuantizeLinear", "DequantizeLinear"]
                )
            )
            * 100
        )
        output_max = static["output_representable_range"][1]
        ref_max = static["reference_logits_range"][1]
        audit_md = f"### Kiểm chứng clipping và profile riêng\n\n- TFLite static output INT8 có scale {static['io']['out']['scale']:.9f}, zero-point {static['io']['out']['zero_point']}; max biểu diễn **{output_max:.6f}**, trong khi reference max **{ref_max:.6f}**. Có {static['reference_values_outside_output_range']} giá trị reference ngoài range output; không đồng nhất con số này với số lỗi phân loại.\n- Có **{ties['total']}/300** mẫu hòa điểm top-1, trong đó argmax vẫn đúng **{ties['correct_argmax']}** mẫu và sai **{ties['wrong_argmax']}** mẫu. Không gọi cả 62 mẫu hòa là 62 lỗi; ONNX static cũng có {audit['ties']['ONNX static']['total']} mẫu hòa, nên không nói ONNX miễn nhiễm.\n- Profile riêng 5 lượt sau 3 warm-up: ONNX dynamic, `ConvInteger` chiếm **{integer_share:.1f}%** tổng thời lượng node-kernel events. `DynamicQuantizeLinear` chỉ {dynamic_ops.get('DynamicQuantizeLinear', {}).get('fraction_of_node_event_time', 0) * 100:.1f}%; ở phép thử này, chi phí kernel convolution integer là phần chính, không chỉ overhead tính range.\n- ONNX static profile còn {static_ops.get('Conv', {}).get('events', 0) // 5} `Conv` và {static_ops.get('QLinearConv', {}).get('events', 0) // 5} `QLinearConv` mỗi lượt; Quantize/Dequantize chiếm **{qdq_share:.1f}%** node-event time. Cùng tên static không bảo đảm cùng mức fusion/kernel coverage giữa backend.\n- Profile là phép đo riêng có overhead, không thay số median trong bảng. Không có profile máy Đức nên chưa xác định nguyên nhân riêng của toàn bộ chênh lệch Mac/Linux. [Audit output và profile](experiments/01-reproduce/results/run-04-peer-linux/output_and_profile_audit.json).\n"
    summary = f"Đã chạy source ResNet18 của Đức trong Linux; {len(ok)}/6 biến thể đo được."
    table = [
        "| Model | Đức báo cáo Mac ms | Huy đo Linux ms | Accuracy Đức → Huy % | Agreement Đức → Huy % | SNR Đức → Huy dB |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, p in published.items():
        r = measured[name]
        if r["status"] == "ok":
            table.append(
                f"| {name} | {p['ms']:.2f} | {r['median_ms']:.3f} | {p['accuracy']:.1f} → {r['accuracy_pct']:.2f} | {p['agreement']:.1f} → {r['agreement_pct']:.2f} | {p['snr']:.1f} → {r['snr_db_source_float32']:.2f} |"
            )
        else:
            table.append(f"| {name} | {p['ms']:.2f} | {r['status']} | — | — | — |")
    failures = (
        "\n".join(
            (
                f"- **{r['name']}**: `{r.get('error', r.get('reason', ''))}`"
                for r in measured.values()
                if r["status"] != "ok"
            )
        )
        or "- Không có variant lỗi."
    )
    extra = []
    for p in sorted((TASK / "results/run-04-huy-linux").glob("*_new_t4_matched.json")):
        r = read(p)
        old = read(TASK / "results/run-01" / f"{r['task']}_{r['variant']}_old_t4_matched.json")
        new = read(TASK / "results/run-01" / f"{r['task']}_{r['variant']}_new_t4_matched.json")
        if r["model_sha256"] != new["model_sha256"] or r["input_sha256"] != new["input_sha256"]:
            raise ValueError("Huy model/input hash differs across environments.")
        name = r["task"] + "/" + r["variant"]
        times = [
            old["benchmark"]["median_ms"],
            new["benchmark"]["median_ms"],
            r["benchmark"]["median_ms"],
        ]
        extra.append(
            f"| {name} | {times[0]:.3f} | {times[1]:.3f} | {times[2]:.3f} | {r['metrics']['accuracy']:.0%} | {r['metrics']['snr_db']:.3f} |"
        )
    md = f"# Reproduce code Đức và đối chiếu phần cứng — 09/10/2026\n\n{summary}\n\n**Không cần cùng phần cứng để reproduce.** Cần giữ pipeline, model/weights, preprocessing, selection dữ liệu, dependency chính và giao thức đo; ghi riêng cấu hình phần cứng/OS. So latency qua hai máy là quan sát toàn hệ thống, không tự quy cho riêng framework hoặc CPU.\n\n## 1. Code và môi trường thực chạy\n\n- Source Đức ghim revision `{peer['peer_revision']}`; SHA256 bảy file trong `peer-sources.lock.json`. Source giữ nguyên và được mount read-only.\n- q1/q2/q3 chạy nguyên script trong process riêng. q4 dùng nguyên make_runner/evaluate/benchmark, tách từng variant để lỗi một model không chặn model khác; thu raw `times` sau timer qua `np.median`. Adaptation nằm trong JSON.\n- ResNet18 pretrained ImageNet; 100 calibration train / 300 evaluation val, đúng sorted first 10/30 ảnh mỗi class theo code Đức. Dùng archive Imagenette đã có; cache weights chính thức.\n- PyTorch reference accuracy: **{peer['reference_accuracy_pct']:.4f}%**, đối chiếu Đức báo **64.7%**. Không có artifact/hash raw của Đức để khẳng định tensor bit-identical giữa hai máy.\n- Đức: Mac M1 Pro theo báo cáo. Huy: i7-12700H, Windows host, Linux x86_64 trong Docker/WSL2. Virtualization, OS, ISA, native kernel, backend build và dependencies gián tiếp cũng có thể ảnh hưởng tốc độ.\n- Protocol của Đức: CPU 4 thread, batch 1, warm-up 10, đo 100 lượt median; API đồng bộ gồm copy/q-deq, không gồm load/preprocess.\n- Main versions thực cài: `{json.dumps(peer['packages'], ensure_ascii=False)}`. Torch/torchvision là CPU build; chỉ ghim release versions khai báo, không có full lockfile môi trường Mac của Đức.\n- Máy Linux nhìn thấy: `{json.dumps(peer['hardware'], ensure_ascii=False)}`.\n- Source unchanged: `{peer['source_unchanged']}`. Evidence: [peer_reproduction.json](experiments/01-reproduce/results/run-04-peer-linux/peer_reproduction.json). Kết quả Mac là **số Đức báo cáo**, không phải Huy đo trên Mac.\n- Collector đã sửa tên file raw FP32 trùng nhau. Output ONNX FP32 được phục hồi riêng và kiểm lại toàn bộ metric; các số quality/latency chính không đổi. [Kiểm chứng export output](experiments/01-reproduce/results/run-04-peer-linux/output_export_verification.json). [Verifier offline 18 cấu hình / 3.000 samples](experiments/01-reproduce/results/run-04-peer-linux/verification.json).\n\n## 2. Cùng ResNet18: Đức báo cáo và Huy reproduce\n\n{chr(10).join(table)}\n\nSNR dùng công thức float32 gốc trong q4 để đối chiếu số Đức; JSON có thêm SNR float64, MAE/max error, hash và 100 raw latency samples. Không gọi tỉ số giữa hai máy là speedup của riêng framework. Chưa kiểm soát nhiệt/power và lặp nhiều session.\n\n{audit_md}\n\n## 3. Variant lỗi hoặc chưa hoàn tất\n\n{failures}\n\nTrạng thái pipeline: `{peer['status']}`. Chỉ báo đủ sáu biến thể khi cả sáu có output/latency hợp lệ. Kết quả lỗi cũng là bằng chứng về tính portable của pipeline; không điền số latency giả.\n\n## 4. Model Huy trên bộ môi trường đã reproduce\n\nĐối chứng giữ nguyên artifact/input Huy, cùng protocol 4 thread / 30 warm-up / 200 lượt; so với Windows run-01. Không trộn protocol này với bảng ResNet18 10/100.\n\n| Model Huy | Windows cũ ms | Windows mới ms | Linux peer env ms | Accuracy Linux | SNR Linux dB |\n| --- | --- | --- | --- | --- | --- |\n{(chr(10).join(extra) if extra else '| Chưa có phép đo model Huy trong Linux | — | — | — | — | — |')}\n\nWindows mới và Linux dùng cùng release ORT/LiteRT; Windows cũ còn khác runtime version. Hash model/input đã đối chiếu giữa hai môi trường. Những chênh lệch này không chứng minh tác động riêng của hardware vì môi trường native/OS/container cũng thay đổi.\n\n## 5. Cách chạy\n\nXem [PEER-RUNNING.md](experiments/01-reproduce/PEER-RUNNING.md). Không sửa hoặc push repo Đức. q1 Windows ONNX-only trước đó nằm riêng ở `run-03-peer-windows`, có bỏ converter và không dùng thay cho kết quả Linux full-mode.\n"
    (DAY / "PEER-REPRODUCTION.md").write_text(md, encoding="utf-8")
    print(f"Peer reproduction: {len(ok)}/6 measured; Markdown report updated.")


if __name__ == "__main__":
    main()
