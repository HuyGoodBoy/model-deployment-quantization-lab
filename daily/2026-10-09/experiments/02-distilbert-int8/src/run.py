"""Controlled DistilBERT conversion; never overwrite the 08 October baseline."""

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import numpy as np

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from qlab.common import (
    classification_metrics,
    load_inputs,
    numerical_metrics,
    samples,
    sha256,
    system_info,
    write_json,
)


def reference(mask_bound=None):
    if mask_bound is not None:
        from qlab.common import tensorflow

        tensorflow(1)
        from transformers.models.distilbert import modeling_tf_distilbert as module

        original = module.TFMultiHeadSelfAttention.call
        source = textwrap.dedent(inspect.getsource(original))
        if source.count("1e30") != 1:
            raise RuntimeError("Unexpected Transformers masking implementation")
        namespace = dict(original.__globals__)
        exec(source.replace("1e30", repr(float(mask_bound))), namespace)
        module.TFMultiHeadSelfAttention.call = namespace["call"]
    from qlab.runtime import reference_function

    return reference_function("text")


def convert(args):
    result = TASK / "results" / args.run_id
    result.mkdir(parents=True, exist_ok=True)
    model_file = TASK / "artifacts" / f"{args.variant}.tflite"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "variant": args.variant,
        "strict_ops": args.variant.startswith("strict"),
        "mask_bound": 1e2
        if "mask1e2" in args.variant
        else 1e4
        if "mask1e4" in args.variant
        else None,
        "scope": "static calibration; INT32 token IDs/mask are preserved",
    }
    try:
        tf, model, infer = reference(record["mask_bound"])
        if record["mask_bound"] is not None:
            inputs, labels, manifest = load_inputs("text", "evaluation")
            output = np.concatenate([infer(**item).numpy() for item in samples(inputs)])
            with np.load(REPO / "quantization/results/text/outputs.npz") as archive:
                original = archive["tensorflow_fp32"].copy()
            np.save(result / f"output_{args.variant}_reference.npy", output, allow_pickle=False)
            check = {
                **numerical_metrics(original, output),
                "allclose": bool(np.allclose(original, output, atol=1e-4, rtol=1e-4)),
                "classification": classification_metrics(output, labels, "text"),
            }
            record["fp32_mask_change_check"] = check
            if not check["allclose"]:
                raise ValueError("Mask change fails the predetermined FP32 equivalence gate")
        calibration, _, manifest = load_inputs("text", "calibration")
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [infer.get_concrete_function()], model
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: iter(samples(calibration))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if not record["strict_ops"]:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
        converter.inference_output_type = tf.int8
        started = time.perf_counter()
        model_file.write_bytes(converter.convert())
        record.update(
            status="ok",
            conversion_seconds=time.perf_counter() - started,
            model_sha256=sha256(model_file),
            size_bytes=model_file.stat().st_size,
            calibration_sha256=manifest["tensor_hashes"]["calibration"],
            tensorflow_version=tf.__version__,
            source_model_sha256=sha256(REPO / "quantization/.cache/text_model/tf_model.h5"),
        )
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(result / f"conversion_{args.variant}.json", record)


def measure(args):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_CPP_MIN_LOG_LEVEL="2",
        OMP_NUM_THREADS=str(args.threads),
        TF_NUM_INTRAOP_THREADS=str(args.threads),
        TF_NUM_INTEROP_THREADS="1",
    )
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    folder = TASK / "results" / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    path = (
        REPO / "quantization/artifacts/text/tflite_static.tflite"
        if args.variant == "baseline"
        else TASK / "artifacts" / f"{args.variant}.tflite"
    )
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=args.threads)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    inputs = interpreter.get_input_details()
    mapping = {
        name: next(d for d in inputs if name in d["name"])
        for name in ("input_ids", "attention_mask")
    }
    if any(d["dtype"] != np.int32 for d in inputs):
        raise ValueError("Token IDs/mask must remain INT32")
    out = interpreter.get_output_details()[0]
    scale, zero = out["quantization"]

    def run(item):
        for name, d in mapping.items():
            interpreter.set_tensor(d["index"], item[name])
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"])
        return (y.astype(np.float32) - zero) * scale if np.issubdtype(y.dtype, np.integer) else y

    canonical, labels, manifest = load_inputs("text", "evaluation")
    output = np.concatenate([run(item) for item in samples(canonical)])
    with np.load(REPO / "quantization/results/text/outputs.npz") as archive:
        original = archive["tensorflow_fp32"].copy()
    first = next(samples(canonical))
    for _ in range(args.warmup):
        run(first)
    latency = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        run(first)
        latency.append((time.perf_counter_ns() - start) / 1e6)
    output_file = folder / f"output_{args.variant}_t{args.threads}.npy"
    np.save(output_file, output, allow_pickle=False)
    audit = {
        dtype: sum(d["dtype"] == dtype for d in details)
        for dtype in (np.float32, np.int8, np.int32)
    }
    from tensorflow.lite.python import schema_py_generated as schema

    flat = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    graph = flat.Subgraphs(0)
    operator_names = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int)
    }
    float_edges = []
    for index in range(graph.OperatorsLength()):
        op = graph.Operators(index)
        tensor_ids = [int(v) for v in [*op.InputsAsNumpy(), *op.OutputsAsNumpy()] if v >= 0]
        float_ids = [i for i in tensor_ids if graph.Tensors(i).Type() == schema.TensorType.FLOAT32]
        if float_ids:
            float_edges.append(
                {
                    "operator": operator_names.get(
                        flat.OperatorCodes(op.OpcodeIndex()).BuiltinCode(), "unknown"
                    ),
                    "float_tensors": [graph.Tensors(i).Name().decode() for i in float_ids],
                }
            )
    record = {
        "status": "ok",
        "variant": args.variant,
        "threads": args.threads,
        "model_sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "input_sha256": manifest["tensor_hashes"]["evaluation"],
        "output_sha256": sha256(output_file),
        "runtime": tf.__version__,
        "system": system_info(),
        "metrics": {
            **numerical_metrics(original, output),
            **classification_metrics(output, labels, "text"),
            "top1_agreement": float(np.mean(original.argmax(1) == output.argmax(1))),
        },
        "audit": {
            "tensor_counts": {np.dtype(k).name: v for k, v in audit.items()},
            "operators_touching_float32": float_edges,
            "max_quant_scale": max(
                (
                    float(np.max(d["quantization_parameters"]["scales"]))
                    for d in details
                    if len(d["quantization_parameters"]["scales"])
                ),
                default=0,
            ),
            "scales_over_1e6": sum(
                bool(np.any(d["quantization_parameters"]["scales"] > 1e6)) for d in details
            ),
            "float_tensors": [
                {"name": d["name"], "shape": d["shape"].tolist()}
                for d in details
                if d["dtype"] == np.float32
            ],
            "output_scale": float(scale),
            "output_zero_point": int(zero),
        },
        "benchmark": {
            "batch": 1,
            "warmup": args.warmup,
            "runs": args.runs,
            "median_ms": float(np.median(latency)),
            "p95_ms": float(np.percentile(latency, 95)),
            "samples_ms": latency,
            "scope": "synchronous inference API, copies and dequantization; excludes load/preprocess",
        },
    }
    write_json(folder / f"measure_{args.variant}_t{args.threads}.json", record)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "accuracy": record["metrics"]["accuracy"],
                "median_ms": record["benchmark"]["median_ms"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("convert", "measure"))
    parser.add_argument(
        "--variant",
        choices=("baseline", "strict", "mixed_mask1e4", "strict_mask1e4", "mixed_mask1e2"),
        required=True,
    )
    parser.add_argument("--run-id", default="run-01")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if args.action == "convert" and args.variant == "baseline":
        parser.error("Baseline is read-only")
    if min(args.threads, args.warmup, args.runs) < 1:
        parser.error("Positive benchmark settings required")
    (convert if args.action == "convert" else measure)(args)
