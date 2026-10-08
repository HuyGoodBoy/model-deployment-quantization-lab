"""Cau hinh CPU, I/O va cac cong thuc danh gia sai khac."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
INPUT_SHAPE = (1, 224, 224, 3)
RUNTIMES = ("tensorflow", "onnx", "tflite")


def configure_cpu(threads: int = 1) -> None:
    # Goi truoc import TensorFlow va ONNX Runtime.
    if threads < 1:
        raise ValueError("threads phai >= 1")
    (ROOT / ".cache" / "keras").mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["KERAS_HOME"] = str(ROOT / ".cache" / "keras")


def tensorflow(threads: int = 1):
    configure_cpu(threads)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_input(x: np.ndarray) -> None:
    if x.shape != INPUT_SHAPE or x.dtype != np.float32:
        raise ValueError(f"Input phai float32 {INPUT_SHAPE}, nhan {x.dtype} {x.shape}")
    if not np.isfinite(x).all() or x.min() < -1.001 or x.max() > 1.001:
        raise ValueError("Input khong finite hoac ngoai [-1, 1]")


def numerical_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """So sanh tensor. Tinh bang float64 de giam sai so cua phep do."""
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("Hai output phai cung shape va khong rong")
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(candidate, dtype=np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Output co NaN hoac infinity")
    error = b - a
    signal_power = float(np.dot(a, a))
    noise_power = float(np.dot(error, error))
    if noise_power == 0:
        snr = math.inf if signal_power > 0 else None
    elif signal_power == 0:
        snr = -math.inf
    else:
        snr = 10 * math.log10(signal_power / noise_power)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine = (float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1, 1))
              if norm_a > 0 and norm_b > 0 else None)
    relative_l2 = (float(np.linalg.norm(error) / norm_a) if norm_a > 0
                   else 0.0 if noise_power == 0 else math.inf)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "snr_db": snr,
    }


def create_runner(runtime: str, threads: int = 1):
    """Moi runner nhan NHWC float32 va tra numpy probabilities [1,1000]."""
    configure_cpu(threads)
    if runtime == "tensorflow":
        tf = tensorflow(threads)
        model = tf.keras.models.load_model(ARTIFACTS / "mobilenetv2.keras",
                                          compile=False)

        @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)],
                     autograph=False)
        def infer(x):
            return model(x, training=False)

        def run(x):
            # .numpy() dam bao da hoan tat phep tinh truoc khi dung timer.
            return infer(tf.convert_to_tensor(x)).numpy()

        return run, {"version": tf.__version__, "mode": "tf.function, CPU"}
    if runtime == "onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(ARTIFACTS / "mobilenetv2.onnx"),
                                       sess_options=options,
                                       providers=["CPUExecutionProvider"])
        spec = session.get_inputs()[0]
        if tuple(spec.shape) != INPUT_SHAPE or spec.type != "tensor(float)":
            raise ValueError(f"ONNX input bat ngo: {spec.shape}, {spec.type}")
        return (lambda x: session.run(None, {spec.name: x})[0]), {
            "version": ort.__version__, "providers": session.get_providers(),
            "graph_optimization": "ORT_ENABLE_ALL",
        }
    if runtime == "tflite":
        # Neu TF da duoc import trong evaluate, khong cau hinh thread lan hai.
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(
            model_path=str(ARTIFACTS / "mobilenetv2.tflite"), num_threads=threads)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        if tuple(inp["shape"]) != INPUT_SHAPE or inp["dtype"] != np.float32:
            raise ValueError("TFLite input khong phai NHWC FP32 nhu thiet ke")
        if out["dtype"] != np.float32:
            raise ValueError("TFLite output khong phai FP32")

        def run(x):
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            return interpreter.get_tensor(out["index"])

        return run, {"version": tf.__version__, "mode": "tf.lite.Interpreter",
                     "delegates": "default runtime delegates (see stderr, e.g. XNNPACK)"}
    raise ValueError(f"Runtime khong hop le: {runtime}")
