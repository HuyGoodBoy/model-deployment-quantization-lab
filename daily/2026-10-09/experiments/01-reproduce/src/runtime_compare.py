"""Run Huy's unchanged models with old/new ORT/LiteRT and matched protocols."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from qlab.common import classification_metrics, numerical_metrics, sha256, system_info, write_json


def main(args):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS=str(args.threads),
        TF_NUM_INTRAOP_THREADS=str(args.threads),
        TF_NUM_INTEROP_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    baseline = REPO / "quantization"
    manifest = json.loads((baseline / "data" / args.task / "manifest.json").read_text())
    input_file = baseline / "data" / args.task / "evaluation.npz"
    if sha256(input_file) != manifest["tensor_hashes"]["evaluation"]:
        raise ValueError("Input checksum mismatch")
    with np.load(input_file, allow_pickle=False) as archive:
        labels = archive["labels"].copy()
        inputs = {k: archive[k].copy() for k in archive.files if k != "labels"}
    path = (
        baseline
        / "artifacts"
        / args.task
        / (args.variant + (".onnx" if args.variant.startswith("onnx") else ".tflite"))
    )
    if args.variant.startswith("onnx"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        if args.protocol == "matched":
            options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        run = lambda x: session.run(None, x)[0]
        runtime = {
            "package": "onnxruntime",
            "version": ort.__version__,
            "providers": session.get_providers(),
        }
    else:
        if args.runtime == "new":
            from ai_edge_litert.interpreter import Interpreter

            version = importlib.metadata.version("ai-edge-litert")
        else:
            import tensorflow as tf

            if args.initialize_tf_threads:
                tf.config.threading.set_intra_op_parallelism_threads(args.threads)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            Interpreter = tf.lite.Interpreter
            version = tf.__version__
        interpreter = Interpreter(model_path=str(path), num_threads=args.threads)
        interpreter.allocate_tensors()
        details = interpreter.get_input_details()
        out = interpreter.get_output_details()[0]
        mapping = {k: next(d for d in details if k in d["name"]) for k in inputs}

        def run(x):
            for k, d in mapping.items():
                value = x[k]
                if value.dtype != d["dtype"]:
                    scale, zero = d["quantization"]
                    limits = np.iinfo(d["dtype"])
                    value = np.clip(
                        np.rint(value.astype(np.float64) / scale) + zero, limits.min, limits.max
                    ).astype(d["dtype"])
                interpreter.set_tensor(d["index"], value)
            interpreter.invoke()
            value = interpreter.get_tensor(out["index"])
            if np.issubdtype(value.dtype, np.integer):
                scale, zero = out["quantization"]
                value = (value.astype(np.float32) - zero) * scale
            return value

        runtime = {
            "package": "ai-edge-litert" if args.runtime == "new" else "tensorflow",
            "version": version,
            "delegate": "default Interpreter delegates; inspect log",
        }

    def item(i):
        return {k: v[i : i + 1] for k, v in inputs.items()}

    output = np.concatenate([run(item(i)) for i in range(len(labels))])
    with np.load(baseline / "results" / args.task / "outputs.npz", allow_pickle=False) as archive:
        original = archive["tensorflow_fp32"].copy()
    first = item(0)
    for _ in range(args.warmup):
        run(first)
    latency = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        run(first)
        latency.append((time.perf_counter_ns() - start) / 1e6)
    name = f"{args.task}_{args.variant}_{args.runtime}_t{args.threads}_{args.protocol}"
    if args.initialize_tf_threads:
        name += "_tfinit"
    folder = TASK / "results" / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    output_file = folder / f"output_{name}.npy"
    np.save(output_file, output, allow_pickle=False)
    record = {
        "status": "ok",
        "name": name,
        "task": args.task,
        "variant": args.variant,
        "runtime": runtime,
        "model_sha256": sha256(path),
        "input_sha256": sha256(input_file),
        "output_sha256": sha256(output_file),
        "metrics": {
            **numerical_metrics(original, output),
            **classification_metrics(output, labels, args.task),
            "agreement": float(np.mean(output.argmax(1) == original.argmax(1))),
        },
        "benchmark": {
            "threads": args.threads,
            "protocol": args.protocol,
            "warmup": args.warmup,
            "runs": args.runs,
            "initialize_tf_threads": args.initialize_tf_threads,
            "batch": 1,
            "median_ms": float(np.median(latency)),
            "p95_ms": float(np.percentile(latency, 95)),
            "samples_ms": latency,
            "scope": "synchronous API, copies and q/deq; excludes load/preprocess",
        },
        "system": system_info(),
    }
    write_json(folder / f"{name}.json", record)
    print(
        f"{name}: {record['benchmark']['median_ms']:.3f} ms, accuracy {record['metrics']['accuracy']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cv", "text"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "onnx_fp32",
            "onnx_dynamic",
            "onnx_static",
            "tflite_fp32",
            "tflite_dynamic",
            "tflite_static",
        ),
        required=True,
    )
    parser.add_argument("--runtime", choices=("old", "new"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--protocol", choices=("matched", "peer"), default="matched")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--run-id", default="run-01")
    parser.add_argument(
        "--initialize-tf-threads",
        action="store_true",
        help="Explicit TF global threading control; old TFLite only",
    )
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.runs) < 1:
        parser.error("Positive settings required")
    if args.initialize_tf_threads and (args.runtime != "old" or args.variant.startswith("onnx")):
        parser.error("Explicit TF threading control applies only to old TFLite")
    main(args)
