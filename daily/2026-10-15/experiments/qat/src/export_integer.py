"""Lower a trained M-LSD checkpoint to 8-bit ConvInteger with float boundaries."""

import argparse
import copy
import json
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto
from torch import nn
from torch.ao.quantization import disable_observer
from torch.nn import functional as F
from torch.onnx import symbolic_helper
from train_continue import PILOT, TASK, pilot, relative

sys.path.insert(0, str(PILOT / "src"))
from models import QatOperation, prepare_qat


class IntegerConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, xs, xz, ws, bias, stride, padding, dilation, groups):
        # The tracing forward is a float reference; symbolic emits integer accumulation.
        x_float = torch.fake_quantize_per_tensor_affine(x, xs, xz.to(torch.int32) - 128, -128, 127)
        w_float = (weight.to(torch.float32) - 128) * ws.reshape(-1, 1, 1, 1)
        return F.conv2d(x_float, w_float, bias, stride, padding, dilation, groups)

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "is", "is", "is", "i")
    def symbolic(g, x, weight, xs, xz, ws, bias, stride, padding, dilation, groups):
        qx = g.op("QuantizeLinear", x, xs, xz)
        wz = g.op("Constant", value_t=torch.tensor(128, dtype=torch.uint8))
        accumulator = g.op(
            "ConvInteger",
            qx,
            weight,
            xz,
            wz,
            strides_i=stride,
            pads_i=padding + padding,
            dilations_i=dilation,
            group_i=groups,
        )
        product_scale = g.op("Mul", xs, ws)
        scale = g.op(
            "Reshape",
            product_scale,
            g.op("Constant", value_t=torch.tensor([1, -1, 1, 1], dtype=torch.int64)),
        )
        output = g.op("Mul", g.op("Cast", accumulator, to_i=TensorProto.FLOAT), scale)
        shaped_bias = g.op(
            "Reshape",
            bias,
            g.op("Constant", value_t=torch.tensor([1, -1, 1, 1], dtype=torch.int64)),
        )
        return g.op("Add", output, shaped_bias)


class ExportConv(nn.Module):
    def __init__(self, operation):
        super().__init__()
        layer = operation.layer
        aq, wq = operation.activation_fake_quant, operation.weight_fake_quant
        if aq.quant_min != -128 or aq.quant_max != 127 or not torch.all(wq.zero_point == 0):
            raise ValueError("Unexpected trained quantization scheme")
        self.register_buffer("xs", aq.scale.detach().reshape(()).clone())
        self.register_buffer("xz", (aq.zero_point + 128).to(torch.uint8).reshape(()))
        self.register_buffer("ws", wq.scale.detach().clone())
        quantized = torch.clamp(
            torch.round(layer.weight.detach() / self.ws.reshape(-1, 1, 1, 1)),
            wq.quant_min,
            wq.quant_max,
        )
        self.register_buffer("weight", (quantized + 128).to(torch.uint8))
        self.register_buffer(
            "bias",
            layer.bias.detach().clone()
            if layer.bias is not None
            else torch.zeros(layer.out_channels),
        )
        self.stride, self.padding = layer.stride, layer.padding
        self.dilation, self.groups = layer.dilation, layer.groups

    def forward(self, x):
        return IntegerConvolution.apply(
            x,
            self.weight,
            self.xs,
            self.xz,
            self.ws,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def replace_convs(parent):
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, QatOperation):
            if not isinstance(child.layer, nn.Conv2d):
                raise TypeError("This lowering handles M-LSD Conv2d only")
            parent._modules[name] = ExportConv(child)
            count += 1
        else:
            count += replace_convs(child)
    return count


def graph_audit(path):
    model = onnx.load(path)
    nodes = Counter(n.op_type for n in model.graph.node)
    dtypes = Counter(TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer)
    initializers = {t.name: t for t in model.graph.initializer}
    convs = []
    for node in model.graph.node:
        if node.op_type == "ConvInteger":
            weight = initializers.get(node.input[1])
            convs.append(
                {
                    "name": node.name,
                    "weight_dtype": TensorProto.DataType.Name(weight.data_type) if weight else None,
                    "attributes": {
                        a.name: onnx.helper.get_attribute_value(a).decode("utf-8")
                        if a.type == onnx.AttributeProto.STRING
                        else onnx.helper.get_attribute_value(a)
                        for a in node.attribute
                    },
                }
            )
    return {
        "operators": dict(nodes),
        "initializer_dtypes": dict(dtypes),
        "conv_integer": convs,
        "float_conv_count": nodes.get("Conv", 0) + nodes.get("FusedConv", 0),
        "size_bytes": path.stat().st_size,
    }


def make_session(path, optimized=None, profile_prefix=None, disable_spinning=False):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if disable_spinning:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if optimized:
        options.optimized_model_filepath = str(optimized)
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def infer_all(session, inputs):
    outputs = [session.run(None, {"input": x[None]})[0] for x in inputs]
    result = np.concatenate(outputs).transpose(0, 2, 3, 1)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite deployed outputs")
    return result


def benchmark_pair(paths, examples, rounds=5, disable_spinning=False):
    # Same images, same 4 threads, alternating order; exclude loading/preprocess/postprocess.
    sessions = {
        name: make_session(path, disable_spinning=disable_spinning) for name, path in paths.items()
    }
    for session in sessions.values():
        for x in examples:
            session.run(None, {"input": x[None]})
    samples = {name: [] for name in sessions}
    order = list(sessions)
    for cycle in range(rounds):
        for index, x in enumerate(examples):
            names = order if (cycle + index) % 2 == 0 else order[::-1]
            for name in names:
                started = time.perf_counter()
                sessions[name].run(None, {"input": x[None]})
                samples[name].append((time.perf_counter() - started) * 1000)
    return {
        name: {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values,
            "runs": len(values),
        }
        for name, values in samples.items()
    }


def main(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    artifacts = args.artifact_dir or TASK / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    data = pilot.load_datasets("mlsd")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["task"] != "mlsd":
        raise ValueError("M-LSD checkpoint required")
    teacher = pilot.mlsd_teacher()
    qat, _config = prepare_qat(teacher, "mlsd")
    if saved.get("research_policy"):
        from research_mlsd import configure_range

        configure_range(qat, saved["research_policy"])
    qat.load_state_dict(saved["model"], strict=True)
    qat.apply(disable_observer)
    qat.eval()
    step = saved["step"]
    del saved
    fake_output = pilot.forward_all(qat, data["evaluation"], "mlsd")
    lowering = copy.deepcopy(qat)
    count = replace_convs(lowering)
    lowering.eval()
    torch_output = pilot.forward_all(lowering, data["evaluation"], "mlsd")
    lowering_gate = pilot.numerical(fake_output, torch_output)
    if lowering_gate["relative_l2"] > 1e-5:
        raise RuntimeError(f"Offset encoding changed float reference: {lowering_gate}")

    sample = data["evaluation"][0][:1]
    quant_path = artifacts / f"mlsd_step{step:04d}_integer.onnx"
    fp32_path = artifacts / "mlsd_original_fp32.onnx"
    for model, path in [(lowering, quant_path), (teacher, fp32_path)]:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (sample,),
                str(path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
        onnx.checker.check_model(str(path), full_check=True)
        print("EXPORTED", path.name, flush=True)

    optimized = artifacts / f"mlsd_step{step:04d}_integer_optimized.onnx"
    profiled = make_session(quant_path, optimized, artifacts / "mlsd_integer_profile")
    profiled.run(None, {"input": sample.numpy()})
    profile_path = Path(profiled.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    kernels = [e for e in events if e.get("cat") == "Node" and "op_name" in e.get("args", {})]
    executed = Counter(e["args"]["op_name"] for e in kernels)
    providers = Counter(e["args"].get("provider", "unknown") for e in kernels)
    del profiled
    audit = graph_audit(quant_path)
    optimized_audit = graph_audit(optimized)
    if audit["operators"].get("ConvInteger", 0) != count or audit["float_conv_count"]:
        raise RuntimeError("Unexpected original integer graph coverage")
    if executed.get("ConvInteger", 0) != count:
        raise RuntimeError(f"Profile did not execute all integer convs: {dict(executed)}")

    inputs = data["evaluation"][0].numpy()
    integer_output = infer_all(make_session(quant_path), inputs)
    fp32_output = infer_all(make_session(fp32_path), inputs)
    fp32_gate = pilot.numerical(data["reference"], fp32_output)
    if fp32_gate["relative_l2"] > 1e-4:
        raise RuntimeError("Exported original FP32 fails reference gate")
    parity = pilot.numerical(fake_output, integer_output)
    fake_metrics = pilot.evaluate(fake_output, data, "mlsd")
    deployed_metrics = pilot.evaluate(integer_output, data, "mlsd")
    ap_delta = (
        deployed_metrics["line_quality"]["sAP10_subset_percent"]
        - fake_metrics["line_quality"]["sAP10_subset_percent"]
    )
    # Float convolution rounding vs integer accumulation can move values over quantizer ties.
    parity_passed = parity["relative_l2"] <= 0.02 and abs(ap_delta) <= 0.5
    output_path = artifacts / f"mlsd_step{step:04d}_integer_output.npy"
    fp32_output_path = artifacts / "mlsd_original_fp32_ort_output.npy"
    np.save(output_path, integer_output, allow_pickle=False)
    np.save(fp32_output_path, fp32_output, allow_pickle=False)
    timing = benchmark_pair(
        {"original_fp32": fp32_path, "qat_integer": quant_path},
        inputs,
        disable_spinning=args.disable_spinning,
    )
    record = {
        "task_date": "2026-10-15",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "status": "exported_and_measured" if parity_passed else "exported_parity_gate_failed",
        "checkpoint": {
            "path": relative(args.checkpoint),
            "sha256": pilot.sha(args.checkpoint),
            "step": step,
        },
        "selection": "Step chosen from repeatedly monitored 16-image subset; not an independent final test",
        "format": "ONNX opset17; ConvInteger UINT8 x UINT8 -> INT32, per-channel scales, float bias/boundaries",
        "encoding": "Trained signed activations/weights shifted by +128 with zero points shifted equally; no recalibration",
        "is_full_integer_model": False,
        "float_regions": "Bias, scale application, ReLU/Clip, residual Add, Resize, Concat and float I/O; output QDQ retains trained center/displacement ranges",
        "qat_float_lowering_gate": lowering_gate,
        "original_fp32_export_gate": fp32_gate,
        "fake_quant_to_integer_parity": parity,
        "parity_gate": {
            "passed": parity_passed,
            "maximum_relative_l2": 0.02,
            "maximum_sAP10_delta_pp": 0.5,
            "actual_sAP10_delta_pp": ap_delta,
        },
        "fake_quant_metrics": fake_metrics,
        "deployed_integer_metrics": deployed_metrics,
        "original_fp32_metrics": pilot.evaluate(fp32_output, data, "mlsd"),
        "graph_audit": audit,
        "optimized_graph_audit": optimized_audit,
        "runtime_profile": {
            "path": relative(profile_path),
            "sha256": pilot.sha(profile_path),
            "executed_operators": dict(executed),
            "providers": dict(providers),
            "integer_conv_nodes_executed": executed.get("ConvInteger", 0),
        },
        "benchmark": {
            "protocol": "CPU ORT, batch1 NCHW 1x4x512x512, 4 intra/1 inter threads, sequential, ALL optimizations; 16 warmups/model, 5 rounds x 16 images, alternating model order; no profiler",
            "scope": "Inference only; excludes session loading, preprocessing, line decoding and Android",
            "idle_thread_spinning_disabled": args.disable_spinning,
            "results": timing,
            "speedup_vs_original_fp32": timing["original_fp32"]["median_ms"]
            / timing["qat_integer"]["median_ms"],
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
        },
        "artifacts": {
            name: {"path": relative(path), "sha256": pilot.sha(path)}
            for name, path in [
                ("integer_model", quant_path),
                ("original_fp32_model", fp32_path),
                ("optimized_integer_model", optimized),
                ("integer_output", output_path),
                ("original_fp32_output", fp32_output_path),
            ]
        },
        "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        "references": [
            "https://onnx.ai/onnx/operators/onnx__ConvInteger.html",
            "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html",
        ],
        "limitations": [
            "Mixed 8-bit integer/float experiment; unsigned offset storage, not an all-signed/full-INT8 model",
            "Custom ConvInteger lowering preserves trained quantization boundaries but is not a tuned QLinearConv deployment",
            "Fixed input shape; only tested on this Windows ORT CPU build; not Android/TFLite",
            "Small repeatedly monitored evaluation subset, no final held-out quality claim",
        ],
    }
    pilot.write_json(args.result_file or TASK / "results/mlsd_integer_export.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "integer_convs": count,
                "sAP10": deployed_metrics["line_quality"]["sAP10_subset_percent"],
                "parity": record["parity_gate"],
                "timing": {k: v["median_ms"] for k, v in timing.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--disable-spinning", action="store_true")
    main(parser.parse_args())
