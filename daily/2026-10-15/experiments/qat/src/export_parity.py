"""Audit frozen M-LSD quantizers and preserve a separate export rounding experiment."""

import argparse
import copy
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from export_integer import graph_audit, infer_all, make_session
from onnx import helper, numpy_helper
from research_mlsd import load_model, read
from train_continue import ROOT, TASK, pilot, relative


def frozen_values(model):
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in values:
            values[node.output[0]] = values[node.input[0]]
        elif node.op_type == "Constant":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if "value" in attrs:
                values[node.output[0]] = numpy_helper.to_array(attrs["value"])
        elif node.op_type == "Cast" and node.input[0] in values:
            dtype = helper.get_attribute_value(next(a for a in node.attribute if a.name == "to"))
            values[node.output[0]] = values[node.input[0]].astype(
                helper.tensor_dtype_to_np_dtype(dtype)
            )
    return values


def explicit_rounding(source):
    """Use FP32 reciprocal, then ties-to-even round, then add the frozen zero point.

    This follows the PyTorch CPU fake-quant kernel's order of operations. It does
    not change any scale, zero point, convolution, weight, bias or decoder.
    """
    model = copy.deepcopy(source)
    values = frozen_values(model)
    nodes, details = [], []
    reserved = {name for node in model.graph.node for name in (*node.input, *node.output)}
    reserved.update(item.name for item in model.graph.initializer)
    for index, node in enumerate(model.graph.node):
        if node.op_type != "QuantizeLinear":
            nodes.append(node)
            continue
        if len(node.input) != 3 or any(name not in values for name in node.input[1:]):
            raise ValueError("Expected frozen scale and explicit zero point")
        scale, zero = (values[name] for name in node.input[1:])
        if scale.dtype != np.float32 or zero.dtype not in (np.dtype("int8"), np.dtype("uint8")):
            raise ValueError("Only frozen FP32 / 8-bit quantizers are supported")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid quantization scale")
        prefix = f"parity_rounding_{index}"
        constants = {
            "inverse": np.reciprocal(scale),
            "zero": zero.astype(np.float32),
            "minimum": np.array(np.iinfo(zero.dtype).min, dtype=np.float32),
            "maximum": np.array(np.iinfo(zero.dtype).max, dtype=np.float32),
        }
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        if scale.size > 1:
            # M-LSD uses rank-four NCHW, and the only per-channel activation is its output.
            if axis != 1:
                raise ValueError("Expected NCHW channel axis 1")
            for key in ("inverse", "zero"):
                constants[key] = constants[key].reshape(1, -1, 1, 1)
        for suffix, array in constants.items():
            name = prefix + "_" + suffix
            if name in reserved:
                raise ValueError("Generated tensor name collision")
            model.graph.initializer.append(numpy_helper.from_array(array, name))
        for suffix in ("scaled", "rounded", "shifted", "clipped"):
            if prefix + "_" + suffix in reserved:
                raise ValueError("Generated tensor name collision")
        for op, inputs, output in [
            ("Mul", [node.input[0], prefix + "_inverse"], prefix + "_scaled"),
            ("Round", [prefix + "_scaled"], prefix + "_rounded"),
            ("Add", [prefix + "_rounded", prefix + "_zero"], prefix + "_shifted"),
            (
                "Clip",
                [prefix + "_shifted", prefix + "_minimum", prefix + "_maximum"],
                prefix + "_clipped",
            ),
        ]:
            nodes.append(helper.make_node(op, inputs, [output], name=prefix + "_" + op))
        nodes.append(
            helper.make_node(
                "Cast",
                [prefix + "_clipped"],
                list(node.output),
                name=prefix + "_Cast",
                to=helper.np_dtype_to_tensor_dtype(zero.dtype),
            )
        )
        details.append({"node": node.name, "scale": scale.tolist(), "zero": zero.tolist()})
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    return model, details


def quantizer_audit(qat, example):
    """Same-input comparisons separate quantizer arithmetic from upstream conv drift."""
    from models import QatOperation

    records, handles = [], []

    def hook(name, mod, inputs):
        x = inputs[0].detach()
        scale = mod.scale.detach()
        zero = mod.zero_point.detach()
        if scale.numel() > 1:
            scale, zero = scale.reshape(1, -1, 1, 1), zero.reshape(1, -1, 1, 1)
        expected = mod(x)
        # Compare dequantized values directly; avoids a second quantization as the oracle.
        reciprocal_codes = torch.clamp(torch.round(x * scale.reciprocal()) + zero, -128, 127)
        division_codes = torch.clamp(torch.round(x / scale) + zero, -128, 127)
        reciprocal_output = (reciprocal_codes - zero) * scale
        division_output = (division_codes - zero) * scale
        records.append(
            {
                "name": name,
                "values": x.numel(),
                "division_mismatches": int((division_output != expected).sum()),
                "reciprocal_mismatches": int((reciprocal_output != expected).sum()),
                "division_max_abs": float((division_output - expected).abs().max()),
            }
        )

    # Hook the parent operation, not the quantizer itself: mod(x) cannot recurse into this hook.
    for name, mod in qat.named_modules():
        if isinstance(mod, QatOperation):
            handles.append(
                mod.register_forward_pre_hook(
                    lambda mod, args, name=name: hook(name, mod.activation_fake_quant, args)
                )
            )
    for name in ("center_quant", "regression_quant"):
        mod = getattr(qat, name)

        # Calling forward bypasses the hook while preserving the frozen fake-quant operator.
        def output_hook(mod, args, output, name=name):
            class ForwardOnly:
                scale, zero_point = mod.scale, mod.zero_point

                def __call__(self, x):
                    return mod.forward(x)

            hook(name, ForwardOnly(), args)

        handles.append(mod.register_forward_hook(output_hook))
    try:
        with torch.no_grad():
            qat(example)
    finally:
        for handle in handles:
            handle.remove()
    return records


def main(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result_path = args.result_dir / "rounding.json"
    if result_path.exists():
        raise FileExistsError(
            "Preserve the existing rounding experiment; choose another result dir"
        )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    deployment = read(TASK / "results/research/deployment.json")
    selection = read(TASK / "results/research/selection.json")
    checkpoint = ROOT / selection["checkpoint"]["path"]
    source = ROOT / deployment["artifacts"]["integer_model"]["path"]
    for info in (
        selection["checkpoint"],
        selection["development_output"],
        deployment["artifacts"]["integer_model"],
        deployment["artifacts"]["integer_output"],
    ):
        if pilot.sha(ROOT / info["path"]) != info["sha256"]:
            raise ValueError("Selected checkpoint, baseline model or output hash changed")
    protected = {
        relative(p): pilot.sha(p)
        for p in [
            checkpoint,
            source,
            TASK / "results/research/selection.json",
            TASK / "results/research/holdout_evaluation.json",
        ]
    }
    data = pilot.load_datasets("mlsd")
    qat, *_ = load_model(checkpoint)
    audit = quantizer_audit(qat, data["evaluation"][0][:1])
    print(
        "QUANTIZERS",
        len(audit),
        "division mismatches",
        sum(r["division_mismatches"] for r in audit),
        "reciprocal mismatches",
        sum(r["reciprocal_mismatches"] for r in audit),
        flush=True,
    )
    del qat
    gc.collect()
    model, quantizers = explicit_rounding(onnx.load(source))
    path = args.artifact_dir / "mlsd_explicit_rounding.onnx"
    onnx.save(model, path)
    print("EXPORTED", len(quantizers), "frozen quantizers", flush=True)
    optimized = args.artifact_dir / "mlsd_explicit_rounding_optimized.onnx"
    session = make_session(path, optimized=optimized, disable_spinning=True)
    outputs = infer_all(session, data["evaluation"][0].numpy())
    del session
    output_path = args.artifact_dir / "mlsd_explicit_rounding_output.npy"
    np.save(output_path, outputs, allow_pickle=False)
    reference = np.load(ROOT / selection["development_output"]["path"], allow_pickle=False)
    old_output = np.load(
        ROOT / deployment["artifacts"]["integer_output"]["path"], allow_pickle=False
    )
    baseline = pilot.numerical(reference, old_output)
    parity = pilot.numerical(reference, outputs)
    metrics = pilot.evaluate(outputs, data, "mlsd")
    ap_delta = (
        metrics["line_quality"]["sAP10_subset_percent"]
        - deployment["fake_quant_metrics"]["line_quality"]["sAP10_subset_percent"]
    )
    if any(pilot.sha(ROOT / p) != digest for p, digest in protected.items()):
        raise RuntimeError("Previously locked evidence changed")
    record = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Development-only export arithmetic repair; same frozen checkpoint, quantization parameters and decoder; holdout not reused for selection",
        "baseline": baseline,
        "candidate": parity,
        "candidate_metrics": metrics,
        "parity_gate": {
            "maximum_relative_l2": 0.02,
            "maximum_sAP10_delta_pp": 0.5,
            "actual_sAP10_delta_pp": ap_delta,
            "passed": parity["relative_l2"] <= 0.02 and abs(ap_delta) <= 0.5,
        },
        "same_input_quantizer_audit": audit,
        "quantizers_rewritten": quantizers,
        "graph_audit": graph_audit(path),
        "optimized_graph_audit": graph_audit(optimized),
        "is_full_integer_model": False,
        "protected_evidence_sha256": protected,
        "artifacts": {
            name: {"path": relative(p), "sha256": pilot.sha(p)}
            for name, p in [
                ("model", path),
                ("optimized_model", optimized),
                ("output", output_path),
            ]
        },
        "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        "environment": {"torch": torch.__version__, "onnx": onnx.__version__},
        "limitations": [
            "Explicit rounding adds float elementwise operators; 41 ConvInteger nodes remain",
            "Float convolution versus integer accumulation may still amplify rounding differences; the unchanged dense-output parity gate remains authoritative",
            "No new training, independent quality test, speed or Android claim",
        ],
        "references": [
            "https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html",
            "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp",
        ],
    }
    pilot.write_json(result_path, record)
    print(
        json.dumps(
            {
                "baseline_relative_l2": baseline["relative_l2"],
                "candidate_relative_l2": parity["relative_l2"],
                "sAP10": metrics["line_quality"]["sAP10_subset_percent"],
                "gate": record["parity_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=TASK / "artifacts/export-parity")
    parser.add_argument("--result-dir", type=Path, default=TASK / "results/export-parity")
    main(parser.parse_args())
