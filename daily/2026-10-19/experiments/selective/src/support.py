"""Frozen inputs and evidence helpers for the approved day-19 experiment."""

import gc
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
RESULTS = TASK / "results"
DAY16 = ROOT / "daily/2026-10-16/experiments/finalize"
DAY15 = ROOT / "daily/2026-10-15/experiments/qat"
for folder in [DAY16 / "src", DAY15 / "src", ROOT / "daily/2026-10-14/experiments/qat/src"]:
    sys.path.append(str(folder))
from train_continue import pilot as pilot  # noqa: PLC0414 -- public re-export


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve recorded evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    path = Path(path)
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def checked(item):
    path = ROOT / item["path"]
    if sha(path) != item["sha256"]:
        raise ValueError(f"Evidence changed: {path}")
    return path


def session(path, optimization="all", optimized=None, profile=None, disabled_optimizers=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimization == "all"
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.log_severity_level = 3
    if optimized:
        options.optimized_model_filepath = str(optimized)
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    return ort.InferenceSession(
        str(path),
        options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=disabled_optimizers,
    )


def infer(path, inputs, optimization="all"):
    runtime = session(path, optimization)
    values = [runtime.run(None, {"input": x[None]})[0] for x in inputs]
    del runtime
    gc.collect()
    outputs = np.concatenate(values).transpose(0, 2, 3, 1)
    if not np.isfinite(outputs).all():
        raise ValueError("Nonfinite inference output")
    return outputs


def save_output(name, output):
    path = ART / (name + ".npy")
    if path.exists():
        raise FileExistsError(path)
    np.save(path, output, allow_pickle=False)
    return info(path)


def baselines():
    selection = read(DAY16 / "results/mlsd_selection.json")
    deployment = read(DAY15 / "results/research/deployment.json")
    qdq = selection["candidates"]["frozen_qdq"]["model"]
    return {
        "teacher_fp32": {
            "model": deployment["artifacts"]["original_fp32_model"],
            "optimization": "all",
        },
        "custom_41_integer": {
            "model": deployment["artifacts"]["integer_model"],
            "optimization": "all",
        },
        "qdq_all": {"model": qdq, "optimization": "all"},
        "qdq_optimizer_off": {"model": qdq, "optimization": "off"},
    }


def block_name(name):
    parts = name.split(".")
    if "features" in parts:
        index = parts.index("features")
        return ".".join(parts[: index + 2])
    return ".".join(parts[:3])


def summarize_profile(trace_path, measured_runs=8):
    """Only the final measured model_run intervals; exclude warm-up and startup."""
    from collections import Counter, defaultdict

    events = read(trace_path)
    runs = [e for e in events if e.get("name") == "model_run"][-measured_runs:]
    if len(runs) != measured_runs:
        raise ValueError("Incomplete profile run intervals")
    kernels = [
        e
        for e in events
        if e.get("cat") == "Node"
        and e.get("name", "").endswith("_kernel_time")
        and any(r["ts"] <= e["ts"] < r["ts"] + r["dur"] for r in runs)
    ]
    unique = {e["name"]: e["args"] for e in kernels}
    totals, nodes = Counter(), defaultdict(lambda: {"durations_ms": [], "args": None})
    for event in kernels:
        op = event["args"]["op_name"]
        totals[op] += event["dur"] / 1000 / measured_runs
        item = nodes[event["name"]]
        item["durations_ms"].append(event["dur"] / 1000)
        item["args"] = event["args"]
    return {
        "measured_runs": measured_runs,
        "unique_operators": dict(Counter(a["op_name"] for a in unique.values())),
        "providers": dict(Counter(a.get("provider", "unknown") for a in unique.values())),
        "operator_mean_ms": dict(totals),
        "mean_model_run_ms": float(np.mean([e["dur"] / 1000 for e in runs])),
        "nodes": [
            {
                "name": name,
                "op": item["args"]["op_name"],
                "provider": item["args"].get("provider"),
                "input_type_shape": item["args"].get("input_type_shape"),
                "mean_ms": float(np.mean(item["durations_ms"])),
                "calls": len(item["durations_ms"]),
            }
            for name, item in nodes.items()
        ],
        "trace": info(trace_path),
        "scope": "Warm profile, final 8 model_run intervals only; profiler overhead included; not benchmark latency",
    }


def profile_model(name, item, inputs):
    optimized = ART / f"{name}_optimized.onnx"
    if optimized.exists():
        raise FileExistsError(optimized)
    runtime = session(
        checked(item["model"]),
        item["optimization"],
        optimized,
        ART / f"{name}_profile",
        item.get("disabled_optimizers"),
    )
    for index in range(16):
        runtime.run(None, {"input": inputs[index % len(inputs)][None]})
    for index in range(8):
        runtime.run(None, {"input": inputs[index][None]})
    trace = Path(runtime.end_profiling())
    del runtime
    gc.collect()
    summary = summarize_profile(trace)
    summary["optimized_model"] = info(optimized)
    return summary
