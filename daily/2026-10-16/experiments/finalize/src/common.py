"""Paths and evidence helpers for the final week evaluation."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
RESULTS = TASK / "results"
DAY15 = ROOT / "daily/2026-10-15/experiments/qat"
sys.path.insert(0, str(DAY15 / "src"))
sys.path.insert(0, str(ROOT / "daily/2026-10-14/experiments/qat/src"))
sys.path.append(str(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/src"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
        raise ValueError(f"Evidence hash mismatch: {path}")
    return path


def session(path, optimized=None, profile=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.log_severity_level = 3
    if optimized:
        options.optimized_model_filepath = str(optimized)
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def profile_summary(path):
    from collections import Counter

    events = [
        e
        for e in read(path)
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]
    nodes = {e["name"]: e["args"] for e in events}
    durations = Counter()
    for event in events:
        durations[event["args"]["op_name"]] += event["dur"] / 1000
    return {
        "unique_executed_operators": dict(Counter(e["op_name"] for e in nodes.values())),
        "providers": dict(Counter(e.get("provider", "unknown") for e in nodes.values())),
        "operator_ms_profiled": dict(durations),
        "profile": info(path),
    }


def wilson(correct, count):
    z = 1.959963984540054
    p = correct / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [float((center - radius) * 100), float((center + radius) * 100)]
