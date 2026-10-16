"""Audit accumulated drift and test one QDQ lowering using frozen QAT parameters."""

import copy
import gc
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from common import ART, DAY15, RESULTS, checked, info, profile_summary, read, session, write
from export_integer import ExportConv, graph_audit, infer_all
from models import QatOperation
from onnx import TensorProto, helper
from research_mlsd import load_model
from torch.onnx import symbolic_helper
from train_continue import pilot


class QdqConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, xs, xz, ws, bias, stride, padding, dilation, groups):
        x = torch.fake_quantize_per_tensor_affine(x, xs, xz.to(torch.int32) - 128, -128, 127)
        w = (weight.float() - 128) * ws.reshape(-1, 1, 1, 1)
        return torch.nn.functional.conv2d(x, w, bias, stride, padding, dilation, groups)

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "is", "is", "is", "i")
    def symbolic(g, x, weight, xs, xz, ws, bias, stride, padding, dilation, groups):
        q = g.op("QuantizeLinear", x, xs, xz)
        dx = g.op("DequantizeLinear", q, xs, xz)
        wz = g.op("Constant", value_t=torch.tensor(128, dtype=torch.uint8))
        wz = g.op("Expand", wz, g.op("Shape", ws))
        dw = g.op("DequantizeLinear", weight, ws, wz, axis_i=0)
        return g.op(
            "Conv",
            dx,
            dw,
            bias,
            strides_i=stride,
            pads_i=padding + padding,
            dilations_i=dilation,
            group_i=groups,
        )


class QdqConv(ExportConv):
    def forward(self, x):
        return QdqConvolution.apply(
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


def replace(parent):
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, QatOperation):
            parent._modules[name] = QdqConv(child)
            count += 1
        else:
            count += replace(child)
    return count


def drift(model, example, baseline_path, candidate_path):
    captured, handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, QatOperation):
            handles.append(
                module.register_forward_pre_hook(
                    lambda mod, args, name=name, captured=captured: captured.__setitem__(
                        name, args[0].detach().numpy().copy()
                    )
                )
            )
    with torch.no_grad():
        model(example)
    for handle in handles:
        handle.remove()
    original = onnx.load(baseline_path)
    producers = {output: n for n in original.graph.node for output in n.output}
    selected, names = [], []
    augmented = onnx.load(candidate_path)
    for node in original.graph.node:
        if node.op_type == "ConvInteger":
            name = node.input[1].removesuffix(".weight")
            if name not in captured:
                raise ValueError(f"Unknown convolution mapping: {name}")
            tensor = producers[node.input[0]].input[0]
            selected.append((name, tensor))
            if tensor not in names:
                augmented.graph.output.append(
                    helper.make_tensor_value_info(
                        tensor, TensorProto.FLOAT, list(captured[name].shape)
                    )
                )
                names.append(tensor)
    path = ART / "mlsd_drift_instrumented.onnx"
    onnx.save(augmented, path)
    runtime = session(path)
    actual = runtime.run(None, {"input": example.numpy()})
    assert np.array_equal(
        actual[0],
        np.load(DAY15 / "artifacts/export-parity/mlsd_explicit_rounding_output.npy")[:1].transpose(
            0, 3, 1, 2
        ),
    )
    lookup = dict(zip(names, actual[1:]))
    rows = [
        {"layer": name, **pilot.numerical(captured[name], lookup[tensor])}
        for name, tensor in selected
    ]
    del runtime, actual, lookup, captured
    return rows


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if (RESULTS / "mlsd_selection.json").exists():
        raise FileExistsError("Final candidate selection already locked")
    ART.mkdir(parents=True, exist_ok=True)
    selection = read(DAY15 / "results/research/selection.json")
    old = read(DAY15 / "results/research/deployment.json")
    rounding = read(DAY15 / "results/export-parity/rounding.json")
    model, *_ = load_model(checked(selection["checkpoint"]))
    data = pilot.load_datasets("mlsd")
    reference = np.load(checked(selection["development_output"]))
    rows = drift(
        model,
        data["evaluation"][0][:1],
        checked(old["artifacts"]["integer_model"]),
        checked(rounding["artifacts"]["model"]),
    )
    write(
        RESULTS / "mlsd_drift.json",
        {
            "date": "2026-10-16",
            "images": 1,
            "mode": "Accumulated upstream activation drift; not same-input local convolution error",
            "instrumentation_preserves_output": True,
            "layers": rows,
        },
    )
    print(
        "DRIFT",
        [(r["layer"], round(r["relative_l2"], 5)) for r in rows if r["relative_l2"] > 0.02][:6],
        flush=True,
    )
    qdq = copy.deepcopy(model)
    count = replace(qdq)
    parity = pilot.numerical(reference, pilot.forward_all(qdq, data["evaluation"], "mlsd"))
    assert parity["relative_l2"] < 1e-5
    path = ART / "mlsd_frozen_qdq.onnx"
    with torch.no_grad():
        torch.onnx.export(
            qdq,
            (data["evaluation"][0][:1],),
            str(path),
            opset_version=17,
            dynamo=False,
            input_names=["input"],
            output_names=["output"],
        )
    onnx.checker.check_model(str(path), full_check=True)
    del model, qdq
    gc.collect()
    optimized = ART / "mlsd_frozen_qdq_optimized.onnx"
    runtime = session(path, optimized, ART / "mlsd_qdq_profile")
    runtime.run(None, {"input": data["evaluation"][0][:1].numpy()})
    profile = profile_summary(Path(runtime.end_profiling()))
    del runtime
    candidates = {"explicit_rounding": checked(rounding["artifacts"]["model"]), "frozen_qdq": path}
    measured = {}
    for name, candidate in candidates.items():
        runtime = session(candidate)
        outputs = infer_all(runtime, data["evaluation"][0].numpy())
        values = []
        for image in data["evaluation"][0].numpy():
            start = time.perf_counter()
            runtime.run(None, {"input": image[None]})
            values.append((time.perf_counter() - start) * 1000)
        del runtime
        output_path = ART / f"mlsd_{name}_development.npy"
        np.save(output_path, outputs, allow_pickle=False)
        metrics = pilot.evaluate(outputs, data, "mlsd")
        numeric = pilot.numerical(reference, outputs)
        delta = (
            metrics["line_quality"]["sAP10_subset_percent"]
            - old["fake_quant_metrics"]["line_quality"]["sAP10_subset_percent"]
        )
        measured[name] = {
            "model": info(candidate),
            "output": info(output_path),
            "metrics": metrics,
            "parity": numeric,
            "development_median_ms": float(np.median(values)),
            "selection_timing_ms": values,
            "parity_gate_passed": numeric["relative_l2"] <= 0.02 and abs(delta) <= 0.5,
            "sAP10_delta_pp": delta,
        }
        print(
            "CANDIDATE",
            name,
            numeric["relative_l2"],
            metrics["line_quality"]["sAP10_subset_percent"],
            np.median(values),
            flush=True,
        )
    ops = profile["unique_executed_operators"]
    integer_count = sum(
        value for name, value in ops.items() if "QLinearConv" in name or "ConvInteger" in name
    )
    eligible = [
        name
        for name, item in measured.items()
        if abs(item["sAP10_delta_pp"]) <= 0.5 and (name != "frozen_qdq" or integer_count > 0)
    ]
    passed = [name for name in eligible if measured[name]["parity_gate_passed"]]
    chosen = min(passed or eligible, key=lambda name: measured[name]["development_median_ms"])
    write(
        RESULTS / "mlsd_selection.json",
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": selection["checkpoint"],
            "selection_rule": "Development only; prefer candidates passing unchanged 2%/0.5pp gate; otherwise fastest experimental candidate within 0.5pp sAP10 with verified integer execution; failed release gate remains failed",
            "holdout_used": False,
            "selected": chosen,
            "candidates": measured,
            "qdq_float_reference_parity": parity,
            "qdq_wrapped_convolutions": count,
            "qdq_runtime": profile,
            "qdq_graph": graph_audit(path),
            "qdq_optimized_operators": dict(
                Counter(n.op_type for n in onnx.load(optimized).graph.node)
            ),
            "scope": "Frozen U8U8 QDQ lowering, no recalibration/training; actual mixed precision determined by profiler",
            "source_code": info(__file__),
        },
    )
    print("LOCKED", chosen, flush=True)


if __name__ == "__main__":
    main()
