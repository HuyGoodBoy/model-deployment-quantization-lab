"""Warm per-Conv comparison of fused QDQ and same-checkpoint FP32 in ORT."""

import csv
import gc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from mlsd_support import ART, RESULTS, info, model_paths, now, pilot, read, session, write


def profile(name, path, inputs):
    optimized = ART / f"{name}_optimized.onnx"
    existing = sorted(ART.glob(name + "_profile_*.json"))
    if existing and optimized.exists():
        trace = existing[-1]
    else:
        runtime = session(path, optimized, ART / (name + "_profile"))
        for index in range(16):
            runtime.run(None, {"input": inputs[index][None]})
        for index in range(8):
            runtime.run(None, {"input": inputs[index][None]})
        trace = Path(runtime.end_profiling())
        del runtime
        gc.collect()
    events = read(trace)
    runs = [e for e in events if e.get("name") == "model_run"][-8:]
    assert len(runs) == 8
    nodes = defaultdict(list)
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        if any(run["ts"] <= event["ts"] < run["ts"] + run["dur"] for run in runs):
            nodes[event["name"]].append(event)
    rows = [
        {
            "name": node,
            "op": values[0]["args"]["op_name"],
            "provider": values[0]["args"].get("provider"),
            "input_type_shape": values[0]["args"].get("input_type_shape"),
            "output_type_shape": values[0]["args"].get("output_type_shape"),
            "calls": len(values),
            "mean_ms": float(np.mean([e["dur"] / 1000 for e in values])),
        }
        for node, values in nodes.items()
    ]
    assert all(row["calls"] == 8 for row in rows)
    source_graph = onnx.load(path)
    producers = {out: node for node in source_graph.graph.node for out in node.output}
    consumers = defaultdict(list)
    for source_node in source_graph.graph.node:
        for value in source_node.input:
            consumers[value].append(source_node)
    convolutions = {}
    for node in source_graph.graph.node:
        if node.op_type != "Conv":
            continue
        weight = node.input[1]
        if weight in producers and producers[weight].op_type == "DequantizeLinear":
            weight = producers[weight].input[0]
        layer = weight.removesuffix(".weight").replace(".layer", "")
        candidates = [node.name, *node.output]
        frontier = list(node.output)
        for _ in range(3):
            fused = [
                child
                for value in frontier
                for child in consumers[value]
                if child.op_type in ["Clip", "Relu", "Identity"]
            ]
            candidates.extend(out for child in fused for out in child.output)
            frontier = [out for child in fused for out in child.output]
        executed = [
            row
            for row in rows
            if any(row["name"].startswith(candidate) for candidate in candidates)
            and row["op"] in ["Conv", "FusedConv", "QLinearConv", "ConvInteger"]
        ]
        assert len(executed) == 1, (node.name, executed)
        convolutions[layer] = {"original_node": node.name, **executed[0]}
    return {
        "model": info(path),
        "optimized": info(optimized),
        "trace": info(trace),
        "unique_operators": dict(Counter(row["op"] for row in rows)),
        "nodes": rows,
        "convolutions": convolutions,
        "warmup": 16,
        "measured_runs": 8,
    }


def main():
    target = RESULTS / "ort_profile_comparison.json"
    if target.exists():
        raise FileExistsError(target)
    ART.mkdir(parents=True, exist_ok=True)
    data = pilot.load_datasets("mlsd")
    profiles = {
        name: profile(name, path, data["evaluation"][0].numpy())
        for name, path in model_paths().items()
    }
    a, b = profiles["qat_weights_fp32"]["convolutions"], profiles["qdq_all"]["convolutions"]
    assert set(a) == set(b) and len(a) == 41
    comparison = []
    for layer in a:
        fp32, qdq = a[layer], b[layer]
        comparison.append(
            {
                "layer": layer,
                "fp32_kernel": fp32["op"],
                "qdq_kernel": qdq["op"],
                "fp32_mean_ms": fp32["mean_ms"],
                "qdq_mean_ms": qdq["mean_ms"],
                "extra_ms": qdq["mean_ms"] - fp32["mean_ms"],
                "ratio": qdq["mean_ms"] / max(fp32["mean_ms"], 1e-30),
            }
        )
    comparison.sort(key=lambda row: -row["extra_ms"])
    selected = [
        row["layer"]
        for row in comparison
        if row["qdq_kernel"] == "QLinearConv" and row["ratio"] > 1.1 and row["extra_ms"] > 0.02
    ][:3]
    write(
        target,
        {
            "measured_at_utc": now(),
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "threads": [4, 1],
            "profiles": profiles,
            "comparison": comparison,
            "float_candidate_layers": selected,
            "selection": "Top3 positive warm Conv overhead among actual integer QLinearConv with ratio>1.1 and extra>0.02ms, development only",
            "scope": "Same QAT checkpoint and geometry; FP32 disables fake quant. Profile includes overhead and different fused activation/quantization boundaries. ORT timings only suggest LiteRT scopes; LiteRT selective result must be measured separately.",
            "source_code": info(__file__),
        },
    )
    with (RESULTS / "ort_conv_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    print("SLOW INTEGER CANDIDATES", selected, flush=True)
    print("TOP OVERHEAD", comparison[:6], flush=True)


if __name__ == "__main__":
    main()
