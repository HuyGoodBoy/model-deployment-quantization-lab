"""Warm ORT node profile and actual LiteRT graph inventory, not LiteRT kernel timing."""

from collections import Counter, defaultdict

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from asr21 import ART, DAY20, RESULTS, ROOT, OnnxRunner, info, now, paths, read, source, write


def main():
    target = RESULTS / "encoder_profile.json"
    if target.exists():
        raise FileExistsError(target)
    ART.mkdir(parents=True, exist_ok=True)
    smoke = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in smoke for p in row["captures"]]
    feeds = [{k: v.copy() for k, v in np.load(ROOT / p["path"]).items()} for p in captures]
    runner = OnnxRunner(source("encoder"), profile=ART / "encoder_profile")
    for n in range(8):
        runner.run(feeds[n % len(feeds)])
    for n in range(8):
        runner.run(feeds[n % len(feeds)])
    trace = runner.runtime.end_profiling()
    events = read(trace)
    windows = [e for e in events if e.get("name") == "model_run"][-8:]
    nodes = defaultdict(list)
    for event in events:
        if (
            event.get("cat") == "Node"
            and event.get("name", "").endswith("_kernel_time")
            and any(w["ts"] <= event["ts"] < w["ts"] + w["dur"] for w in windows)
        ):
            nodes[event["name"]].append(event)
    rows = [
        {
            "name": name,
            "op": values[0]["args"]["op_name"],
            "calls": len(values),
            "mean_ms": float(np.mean([e["dur"] for e in values]) / 1000),
        }
        for name, values in nodes.items()
    ]
    groups = defaultdict(float)
    for row in rows:
        groups[row["op"]] += row["mean_ms"]
    runtime = Interpreter(model_path=str(paths("baseline")["encoder"]), num_threads=4)
    runtime.allocate_tensors()
    tensors = {t["index"]: t for t in runtime.get_tensor_details()}
    operators = runtime._get_ops_details()
    constant_linear = []
    for op in operators:
        if op["op_name"] == "BATCH_MATMUL":
            rhs = tensors[int(op["inputs"][1])]
            if len(rhs["shape"]) == 2:
                constant_linear.append({"name": rhs["name"], "shape": rhs["shape"].tolist()})
    write(
        target,
        {
            "finished_at_utc": now(),
            "onnxruntime": ort.__version__,
            "source": info(source("encoder")),
            "trace": info(trace),
            "warmups": 8,
            "measured": 8,
            "profile_op_ms_per_run": dict(sorted(groups.items(), key=lambda x: -x[1])),
            "nodes": sorted(rows, key=lambda r: -r["mean_ms"]),
            "litert_model": info(paths("baseline")["encoder"]),
            "litert_operator_counts": dict(Counter(o["op_name"] for o in operators)),
            "rank2_rhs_BMM_candidates": constant_linear,
            "scope": "ORT warmed8runs,4threads; ranking only, not LiteRT per-kernel timing. LiteRT constant RHS candidates are structurally checked against FlatBuffer buffers before rewriting. Profile is not a performance benchmark.",
            "source_code": info(__file__),
        },
    )
    print(
        "ENCODER OPS",
        dict(sorted(groups.items(), key=lambda x: -x[1])[:8]),
        "rank2BMM",
        len(constant_linear),
        flush=True,
    )


if __name__ == "__main__":
    main()
