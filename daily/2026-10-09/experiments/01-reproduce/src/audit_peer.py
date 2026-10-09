"""Untimed output audit and separate ORT profiling; never mix with benchmarks."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

TASK = Path(__file__).resolve().parents[1]
RESULTS = TASK / "results/run-04-peer-linux"
ARTIFACTS = TASK / "artifacts/run-04-peer-linux"


def main():
    measured = json.loads((RESULTS / "peer_reproduction.json").read_text())
    labels = np.load(ARTIFACTS / "val_y.npy", allow_pickle=False)
    audit = {
        "scope": "post-benchmark audit; profiling durations are not benchmark latency",
        "ties": {},
        "profiles": {},
    }
    for record in measured["results"]:
        if record["status"] != "ok":
            continue
        output = np.load(RESULTS / record["output_file"], allow_pickle=False)
        tied = np.sum(output == output.max(axis=1, keepdims=True), axis=1) > 1
        correct = output.argmax(1) == labels
        audit["ties"][record["name"]] = {
            "total": int(tied.sum()),
            "correct_argmax": int(np.sum(tied & correct)),
            "wrong_argmax": int(np.sum(tied & ~correct)),
        }
    sample = np.load(ARTIFACTS / "val_x.npy", mmap_mode="r", allow_pickle=False)[:1]
    for variant, filename in [
        ("fp32", "resnet18_fp32.onnx"),
        ("dynamic", "resnet18_onnx_dynamic.onnx"),
        ("static", "resnet18_onnx_static.onnx"),
    ]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.enable_profiling = True
        options.profile_file_prefix = str(ARTIFACTS / ("ort_profile_" + variant))
        session = ort.InferenceSession(
            str(ARTIFACTS / filename), options, providers=["CPUExecutionProvider"]
        )
        for _ in range(8):
            session.run(None, {"input": sample})
        events = json.loads(Path(session.end_profiling()).read_text())
        runs = [e for e in events if e.get("name") == "model_run" and "dur" in e][-5:]
        per_op = defaultdict(lambda: {"duration_us": 0, "events": 0})
        for event in events:
            if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
                continue
            if not any(r["ts"] <= event["ts"] < r["ts"] + r["dur"] for r in runs):
                continue
            key = event.get("args", {}).get("op_name", "unknown")
            per_op[key]["duration_us"] += event["dur"]
            per_op[key]["events"] += 1
        total = sum(v["duration_us"] for v in per_op.values())
        audit["profiles"][variant] = {
            "runs": len(runs),
            "provider": "CPUExecutionProvider",
            "threads": 4,
            "node_event_total_us": total,
            "ops": {
                k: {**v, "fraction_of_node_event_time": v["duration_us"] / total if total else 0}
                for k, v in sorted(per_op.items(), key=lambda kv: -kv[1]["duration_us"])
            },
        }
    (RESULTS / "output_and_profile_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
