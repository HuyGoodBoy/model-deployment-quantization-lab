"""Shared runtime and evidence helpers for the requested streaming transducer."""

import hashlib
import json
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
RESULTS = TASK / "results"
SOURCE = ART / "source"


def now():
    return datetime.now(UTC).isoformat()


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


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def source(component):
    return SOURCE / f"{component}-epoch-99-avg-1.onnx"


def signature(value):
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": [
            d.dim_value if d.HasField("dim_value") else d.dim_param for d in tensor.shape.dim
        ],
    }


@lru_cache(maxsize=16)
def input_specs(path):
    graph = onnx.load(path)
    result = []
    for node in graph.graph.input:
        shape = [
            d.dim_value if d.HasField("dim_value") else 1 for d in node.type.tensor_type.shape.dim
        ]
        dtype = onnx.helper.tensor_dtype_to_np_dtype(node.type.tensor_type.elem_type)
        result.append((node.name, shape, dtype))
    return result


def zeros_for_model(path):
    return {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in input_specs(path)}


def numeric(reference, actual):
    a, b = np.asarray(reference), np.asarray(actual)
    if a.shape != b.shape or not np.isfinite(b).all():
        return {
            "shape_equal": False,
            "reference_shape": list(a.shape),
            "actual_shape": list(b.shape),
            "finite": bool(np.isfinite(b).all()),
        }
    delta = b.astype(np.float64) - a.astype(np.float64)
    denom = np.linalg.norm(a.astype(np.float64))
    return {
        "shape_equal": True,
        "finite": True,
        "reference_dtype": str(a.dtype),
        "actual_dtype": str(b.dtype),
        "dtype_equal": a.dtype == b.dtype,
        "exact": bool(np.array_equal(a, b)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0,
        "relative_l2": float(np.linalg.norm(delta) / max(denom, 1e-30)),
        "allclose_rtol1e_4_atol1e_4": bool(np.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


class OnnxRunner:
    def __init__(self, path, profile=None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if profile:
            opts.enable_profiling = True
            opts.profile_file_prefix = str(profile)
        self.runtime = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [x.name for x in self.runtime.get_inputs()]
        self.output_names = [x.name for x in self.runtime.get_outputs()]
        self.times = []

    def run(self, inputs):
        started = time.perf_counter_ns()
        outputs = self.runtime.run(self.output_names, inputs)
        self.times.append((time.perf_counter_ns() - started) / 1e6)
        return dict(zip(self.output_names, outputs))


class LiteRunner:
    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.runtime = Interpreter(model_path=str(path), num_threads=4)
        self.runtime.allocate_tensors()
        signatures = self.runtime.get_signature_list()
        if len(signatures) != 1:
            raise ValueError(f"Expected one signature, got {signatures}")
        name = next(iter(signatures))
        self.function = self.runtime.get_signature_runner(name)
        self.input_names = signatures[name]["inputs"]
        self.output_names = signatures[name]["outputs"]
        self.input_details = self.function.get_input_details()
        self.counter_widenings = 0
        self.times = []

    def run(self, inputs):
        started = time.perf_counter_ns()
        adapted = {}
        for name, value in inputs.items():
            expected = self.input_details[name]["dtype"]
            if value.dtype != expected:
                if (
                    name.startswith("cached_len_")
                    and value.dtype == np.int32
                    and expected == np.int64
                ):
                    value = value.astype(np.int64)
                    self.counter_widenings += 1
                else:
                    raise ValueError(
                        f"Unexpected dtype change for {name}: {value.dtype} -> {expected}"
                    )
            adapted[name] = value
        result = self.function(**adapted)
        self.times.append((time.perf_counter_ns() - started) / 1e6)
        return result
