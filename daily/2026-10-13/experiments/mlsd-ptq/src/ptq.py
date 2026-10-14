"""Controlled PTQ ablations: output heads, calibration and selective precision."""

import argparse
import importlib.util
import os
import time
from collections import Counter
from datetime import UTC, datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import flatbuffers
import numpy as np
import tensorflow as tf
from metrics import line_metrics, numerical
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.python.convert import mlir_quantize

from data import ROOT, TASK, load, sha, write_json

OLD = ROOT / "daily/2026-10-09/experiments/03-mlsd"
ART = TASK / "artifacts"
RESULTS = TASK / "results"


def build_models():
    spec = importlib.util.spec_from_file_location("keras_bridge", OLD / "src/port_keras.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.build(np.load(OLD / "artifacts/torch_state.npz", allow_pickle=False))
    head = next(l for l in reversed(original.layers) if isinstance(l, tf.keras.layers.Conv2D))
    weights = head.get_weights()

    def output(start, stop, name, divisor=1):
        conv = tf.keras.layers.Conv2D(stop - start, 1, use_bias=len(weights) > 1, name=name)
        value = conv(head.input)
        selected = [weights[0][..., start:stop] / divisor]
        if len(weights) > 1:
            selected.append(weights[1][start:stop] / divisor)
        conv.set_weights(selected)
        return tf.keras.layers.Lambda(
            lambda v: tf.raw_ops.ResizeBilinear(
                images=v, size=[256, 256], align_corners=True, half_pixel_centers=False
            ),
            name=name + "_resize",
        )(value)

    active = tf.keras.Model(original.input, output(7, 12, "active_head"))
    split = tf.keras.Model(
        original.input, [output(7, 8, "center_head"), output(8, 12, "displacement_head")]
    )
    scaled = tf.keras.Model(
        original.input, [output(7, 8, "scaled_center"), output(8, 12, "scaled_displacement", 128)]
    )
    return original, active, split, scaled


def convert(model, name, calibration, mode="int8", mixed_last=0, mixed_output=False):
    path = ART / (name + ".tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if mode == "fp16":
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.representative_dataset = lambda: ([image[None]] for image in calibration)
        if not mixed_last:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            converter._experimental_calibrate_only = True
    raw = converter.convert()
    denied = []
    denied_resize = []
    if mixed_last:
        model_t = schema.Model.GetRootAsModel(raw, 0)
        graph = model_t.Subgraphs(0)
        names = []
        resize_names = []
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            code = model_t.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if code in (schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.DEPTHWISE_CONV_2D):
                names.append(graph.Tensors(op.Outputs(0)).Name().decode())
            if code == schema.BuiltinOperator.RESIZE_BILINEAR:
                resize_names.append(graph.Tensors(op.Outputs(0)).Name().decode())
        denied = names[-mixed_last:]
        if mixed_output:
            denied_resize = resize_names[-1:]
        raw = mlir_quantize(
            raw,
            fully_quantize=False,
            input_data_type=tf.float32,
            output_data_type=tf.float32,
            denylisted_nodes=denied + denied_resize,
        )
    path.write_bytes(raw)
    return path, {
        "denied_conv_outputs": denied,
        "denied_resize_outputs": denied_resize,
        "converter": tf.__version__,
    }


def validate_heads(models, inputs, reference):
    for model, scale, name in models:
        values = []
        for image in inputs["evaluation"]:
            predicted = model(image[None], training=False)
            if isinstance(predicted, list):
                predicted = np.concatenate(
                    [predicted[0].numpy(), predicted[1].numpy() * scale], axis=-1
                )
            else:
                predicted = predicted.numpy()
            values.append(predicted)
        gate = numerical(reference, np.concatenate(values))
        gate.update(examples=len(values), relative_l2_limit=1e-4, passed=gate["relative_l2"] < 1e-4)
        write_json(RESULTS / ("gate_" + name + ".json"), gate)
        if not gate["passed"]:
            raise RuntimeError(f"Head transform FP32 gate failed: {name}")
        print("FP32 gate", name, "passed on", len(values), "images", flush=True)


def selective_fp16(name, count):
    """Restore last N conv weights/biases to original FP32 in a FP16-storage graph.

    Pair corresponding convolution inputs in FP32 and FP16 graphs, replace the
    dequantized weight tensor with a FP32 constant and remove its DEQUANTIZE.
    No training, no fitting to evaluation images, arithmetic remains CPU FP32.
    """
    fp = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel((OLD / "artifacts/fp32.tflite").read_bytes(), 0)
    )
    low = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel((OLD / "artifacts/fp16.tflite").read_bytes(), 0)
    )

    def code(model, op):
        return model.operatorCodes[op.opcodeIndex].builtinCode

    kinds = (schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.DEPTHWISE_CONV_2D)
    a = [op for op in fp.subgraphs[0].operators if code(fp, op) in kinds]
    b = [op for op in low.subgraphs[0].operators if code(low, op) in kinds]
    assert len(a) == len(b)
    producers = {int(index): op for op in low.subgraphs[0].operators for index in op.outputs}
    removed = set()
    restored = []
    for first, second in list(zip(a, b))[-count:]:
        assert code(fp, first) == code(low, second)
        for slot in [1, 2]:
            if len(first.inputs) <= slot or first.inputs[slot] < 0:
                continue
            source = fp.subgraphs[0].tensors[first.inputs[slot]]
            target = low.subgraphs[0].tensors[second.inputs[slot]]
            producer = producers.get(int(second.inputs[slot]))
            assert producer is not None and code(low, producer) == schema.BuiltinOperator.DEQUANTIZE
            assert source.shape.tolist() == target.shape.tolist()
            buffer = schema.BufferT()
            buffer.data = fp.buffers[source.buffer].data.copy()
            low.buffers.append(buffer)
            target.buffer = len(low.buffers) - 1
            target.type = schema.TensorType.FLOAT32
            removed.add(id(producer))
            restored.append(target.name.decode())
    low.subgraphs[0].operators = [op for op in low.subgraphs[0].operators if id(op) not in removed]
    builder = flatbuffers.Builder(0)
    offset = low.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    path = ART / (name + ".tflite")
    path.write_bytes(builder.Output())
    return path, {
        "restored_last_convolutions": count,
        "restored_constants": restored,
        "scope": "mixed weight storage FP16/FP32; CPU arithmetic is FP32; no gradient training",
    }


def runner(path, displacement_scale=1):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()

    def invoke(image):
        x = image[None]
        if np.issubdtype(inp["dtype"], np.integer):
            s, z = inp["quantization"]
            limits = np.iinfo(inp["dtype"])
            x = np.clip(np.rint(x / s) + z, limits.min, limits.max).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        values = []
        for d in outs:
            y = interpreter.get_tensor(d["index"])
            if np.issubdtype(d["dtype"], np.integer):
                s, z = d["quantization"]
                y = (y.astype(np.float32) - z) * s
            values.append(y)
        if len(values) == 1:
            return values[0][..., :5]
        center = next(v for v in values if v.shape[-1] == 1)
        displacement = next(v for v in values if v.shape[-1] == 4) * displacement_scale
        return np.concatenate([center, displacement], axis=-1)

    details = interpreter.get_tensor_details()
    ops = interpreter._get_ops_details()
    audit = {
        "tensor_dtypes": dict(Counter(np.dtype(d["dtype"]).name for d in details)),
        "operators": dict(Counter(o["op_name"] for o in ops)),
        "input": {"dtype": np.dtype(inp["dtype"]).name, "quantization": list(inp["quantization"])},
        "outputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": np.dtype(d["dtype"]).name,
                "quantization": list(d["quantization"]),
            }
            for d in outs
        ],
    }
    audit["float_conv_nodes"] = sum(
        any(
            details_by_index(interpreter)[int(i)]["dtype"] == np.float32
            for i in o["inputs"][:2]
            if i >= 0
        )
        for o in ops
        if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")
    )
    return invoke, audit


def details_by_index(interpreter):
    return {d["index"]: d for d in interpreter.get_tensor_details()}


def measure(name, path, inputs, ground_truth, reference, scale=1, warmup=10, runs=40, extra=None):
    invoke, audit = runner(path, scale)
    actual = np.concatenate([invoke(image) for image in inputs["evaluation"]])
    np.save(ART / ("output_" + name + ".npy"), actual, allow_pickle=False)
    metrics = {
        "active": numerical(reference, actual),
        "center": numerical(reference[..., :1], actual[..., :1]),
        "displacement": numerical(reference[..., 1:5], actual[..., 1:5]),
    }
    for _ in range(warmup):
        invoke(inputs["evaluation"][0])
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        invoke(inputs["evaluation"][0])
        times.append((time.perf_counter_ns() - start) / 1e6)
    record = {
        "status": "ok",
        "variant": name,
        "model_sha256": sha(path),
        "model_bytes": path.stat().st_size,
        "output_sha256": sha(ART / ("output_" + name + ".npy")),
        "metrics": metrics,
        "line_quality": line_metrics(actual, ground_truth),
        "audit": audit,
        "benchmark": {
            "threads": 4,
            "warmup": warmup,
            "runs": runs,
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "samples_ms": times,
            "scope": "TFLite inference API + input quantize + output copy/dequantize/reconstruct; no image preprocessing or line decoding",
        },
        "changes": extra or {},
    }
    write_json(RESULTS / (name + ".json"), record)
    print(
        f"{name}: active SNR {metrics['active']['snr_db']:.2f}, center {metrics['center']['snr_db']:.2f}, sAP10 {record['line_quality']['sAP10_subset_percent']:.2f}, {np.median(times):.2f} ms",
        flush=True,
    )
    return record


def main(args):
    ART.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    inputs, ground_truth, manifest = load()
    reference = np.load(ART / "reference_evaluation.npy", allow_pickle=False)
    records = []
    for name in ["fp32", "fp16", "static"]:
        records.append(
            measure(
                "original_" + name,
                OLD / "artifacts" / (name + ".tflite"),
                inputs,
                ground_truth,
                reference,
                runs=args.runs,
            )
        )
    if records[0]["metrics"]["active"]["relative_l2"] >= 1e-4:
        raise RuntimeError("Original FP32 TFLite fails fresh-data numerical gate")
    for count in [1, 6, 12]:
        name = f"fp16_keep_last{count}_fp32"
        path, extra = selective_fp16(name, count)
        records.append(
            measure(name, path, inputs, ground_truth, reference, runs=args.runs, extra=extra)
        )
    original, active, split, scaled = build_models()
    validate_heads(
        [
            (active, 1, "active5"),
            (split, 1, "split"),
            (scaled, 128, "scaled_split"),
        ],
        inputs,
        reference,
    )
    jobs = [
        ("recalibrated_int8", original, "int8", 0, 1, False),
        ("active5_int8", active, "int8", 0, 1, False),
        ("split_int8", split, "int8", 0, 1, False),
        ("scaled_split_int8", scaled, "int8", 0, 128, False),
        ("mixed_last6_fp32", active, "int8", 6, 1, False),
        ("mixed_last6_and_output_fp32", active, "int8", 6, 1, True),
    ]
    failures = []
    for name, model, mode, mixed_last, scale, mixed_output in jobs:
        try:
            path, extra = convert(
                model, name, inputs["calibration"], mode, mixed_last, mixed_output
            )
            records.append(
                measure(
                    name,
                    path,
                    inputs,
                    ground_truth,
                    reference,
                    scale=scale,
                    runs=args.runs,
                    extra=extra,
                )
            )
        except (ValueError, RuntimeError, AssertionError) as error:
            failed = {"status": "failed", "variant": name, "error": str(error)}
            write_json(RESULTS / (name + ".json"), failed)
            failures.append(failed)
            print(name, "FAILED", error, flush=True)
    write_json(
        RESULTS / "summary.json",
        {
            "task_date": "2026-10-13",
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "status": "completed experiments; bounded dataset, not full benchmark",
            "tensorflow": tf.__version__,
            "calibration_images": len(inputs["calibration"]),
            "evaluation_images": len(inputs["evaluation"]),
            "input_sha256": manifest["inputs_sha256"],
            "reference_sha256": sha(ART / "reference_evaluation.npy"),
            "notes": [
                "Task 13/10 executed as catch-up on 14/10.",
                "Selective FP16 keeps some constants FP32; not pure FP16.",
                "mixed_last6 permits FP32 ops; not full INT8.",
                "sAP10 is labelled YorkUrban subset with stated decoder protocol, not paper score.",
            ],
            "variants": [
                {
                    "variant": r["variant"],
                    "snr": r["metrics"]["active"]["snr_db"],
                    "center_snr": r["metrics"]["center"]["snr_db"],
                    "sAP10": r["line_quality"]["sAP10_subset_percent"],
                    "median_ms": r["benchmark"]["median_ms"],
                    "bytes": r["model_bytes"],
                }
                for r in records
            ],
            "failures": failures,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40)
    main(parser.parse_args())
