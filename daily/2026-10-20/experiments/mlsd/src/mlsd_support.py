"""Day-20 M-LSD evidence paths; older models/data are read-only."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import onnxruntime as ort

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
RESULTS = TASK / "results"
DAY19 = ROOT / "daily/2026-10-19/experiments/selective"
for folder in [
    ROOT / "daily/2026-10-15/experiments/qat/src",
    ROOT / "daily/2026-10-14/experiments/qat/src",
    ROOT / "daily/2026-10-13/experiments/mlsd-ptq/src",
]:
    sys.path.append(str(folder))


def __getattr__(name):
    # Conversion-only environment does not need PyTorch/data-loading imports.
    if name == "pilot":
        from train_continue import pilot

        return pilot
    raise AttributeError(name)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(path):
    path = Path(path)
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def model_paths():
    records = read(DAY19 / "results/exports.json")["models"]
    result = {}
    for name in ["qat_weights_fp32", "qdq_all"]:
        item = records[name]["model"]
        path = ROOT / item["path"]
        assert sha(path) == item["sha256"]
        result[name] = path
    return result


def session(path, optimized=None, profile=None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.log_severity_level = 3
    if optimized:
        opts.optimized_model_filepath = str(optimized)
    if profile:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
