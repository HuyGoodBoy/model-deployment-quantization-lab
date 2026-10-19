"""Recompute metrics, candidate lock, graph/kernel evidence and selected inference."""

from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from support import (
    DAY16,
    RESULTS,
    ROOT,
    checked,
    infer,
    info,
    now,
    pilot,
    read,
    summarize_profile,
    write,
)

# isort: split
from models import QatOperation
from regression import paired
from research_mlsd import load_model


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def main():
    if (RESULTS / "verification.json").exists():
        raise FileExistsError("Verification already recorded")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frozen = read(RESULTS / "frozen_sources.json")
    for item in frozen["files"]:
        checked(item)
    diagnosis = read(RESULTS / "diagnosis.json")
    exports = read(RESULTS / "exports.json")
    development = read(RESULTS / "development.json")
    benchmark = read(RESULTS / "benchmark.json")
    selection = read(RESULTS / "selection.json")
    regression = read(RESULTS / "regression.json")
    checked(exports["checkpoint"])
    checked(exports["diagnosis"])
    checked(development["exports"])
    checked(selection["development"])
    checked(selection["benchmark"])
    checked(regression["selection"])
    assert datetime.fromisoformat(selection["locked_at_utc"]) < datetime.fromisoformat(
        regression["regression_started_at_utc"]
    )
    assert not selection["regression_used"]
    assert regression["selected_before_regression"] == selection["selected"]
    data = pilot.load_datasets("mlsd")
    fake = np.load(checked(exports["fake_quant_reference"]))
    profile_counts = {}
    profiles = {
        **diagnosis["profiles"],
        **{
            name: item["profile"]
            for name, item in exports["candidates"].items()
            if "profile" in item
        },
    }
    for name, item in profiles.items():
        recomputed = summarize_profile(checked(item["trace"]))
        close(recomputed, {key: value for key, value in item.items() if key != "optimized_model"})
        graph = onnx.load(checked(item["optimized_model"]))
        counts = {
            op: sum(n.op_type == op for n in graph.graph.node) for op in item["unique_operators"]
        }
        assert counts == item["unique_operators"], (name, counts)
        assert item["providers"] == {"CPUExecutionProvider": sum(counts.values())}
        assert all(n["calls"] == 8 for n in item["nodes"])
        # QLinearConv and ConvInteger must actually receive 8-bit input and weights.
        for node in item["nodes"]:
            if node["op"] in ["QLinearConv", "ConvInteger"]:
                types = list(node["input_type_shape"] or [])
                first = types[0]
                assert any(t in first for t in ["uint8", "int8"]), first
                # ORT omits prepacked weights from profile input_type_shape.
                # Inspect the graph's schema input slot instead of indexing that list.
                matching = [n for n in graph.graph.node if n.name + "_kernel_time" == node["name"]]
                assert len(matching) == 1
                weight_slot = 3 if node["op"] == "QLinearConv" else 1
                weights = {t.name: t for t in graph.graph.initializer}
                weight = weights[matching[0].input[weight_slot]]
                assert weight.data_type in [onnx.TensorProto.UINT8, onnx.TensorProto.INT8]
        profile_counts[name] = counts
    model, _teacher, _saved, _ = load_model(checked(exports["checkpoint"]))
    modules = dict(model.named_modules())
    for name, item in exports["models"].items():
        path = checked(item["model"])
        output = np.load(checked(development["models"][name]["output"]))
        assert output.shape == fake.shape and np.isfinite(output).all()
        close(
            pilot.line_metrics(output, data["ground_truth"]), development["models"][name]["sAP10"]
        )
        close(pilot.numerical(fake, output), development["models"][name]["vs_original_fake"])
        timing = benchmark["models"][name]
        assert timing["runs"] == len(timing["samples_ms"]) == 80
        close(float(np.median(timing["samples_ms"])), timing["median_ms"])
        close(float(np.percentile(timing["samples_ms"], 95)), timing["p95_ms"])
        close(
            [float(np.median(timing["samples_ms"][i : i + 16])) for i in range(0, 80, 16)],
            timing["round_medians_ms"],
        )
        checked(timing["model"])
        if name in ["selective_a", "selective_b"]:
            graph = onnx.load(path)
            arrays = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
            producers = {out: node for node in graph.graph.node for out in node.output}

            def constant(name, arrays=arrays, producers=producers):
                if name in arrays:
                    return arrays[name]
                node = producers[name]
                if node.op_type == "Identity":
                    return constant(node.input[0])
                if node.op_type == "Constant":
                    return numpy_helper.to_array(
                        next(a.t for a in node.attribute if a.name == "value")
                    )
                if node.op_type == "Shape":
                    return np.asarray(constant(node.input[0]).shape, dtype=np.int64)
                if node.op_type == "Expand":
                    return np.broadcast_to(constant(node.input[0]), constant(node.input[1]))
                raise ValueError(f"Nonconstant qparam: {name}, {node.op_type}")

            for layer, module in modules.items():
                if not isinstance(module, QatOperation):
                    continue
                raw = constant(layer + ".weight")
                expected_bias = (
                    module.layer.bias.detach().numpy()
                    if module.layer.bias is not None
                    else np.zeros(module.layer.out_channels, np.float32)
                )
                assert np.array_equal(constant(layer + ".bias"), expected_bias)
                if layer in item["float_layers"]:
                    assert raw.dtype == np.float32
                    assert np.array_equal(raw, module.layer.weight.detach().numpy())
                else:
                    scales = module.weight_fake_quant.scale.detach().numpy().reshape(-1, 1, 1, 1)
                    quantized = (
                        np.clip(np.rint(module.layer.weight.detach().numpy() / scales), -127, 127)
                        + 128
                    )
                    assert raw.dtype == np.uint8 and np.array_equal(raw, quantized.astype(np.uint8))
                    assert np.array_equal(
                        constant(layer + ".xs"),
                        module.activation_fake_quant.scale.detach().numpy().reshape(()),
                    )
                    expected_zero = (
                        (module.activation_fake_quant.zero_point.detach().numpy() + 128)
                        .astype(np.uint8)
                        .reshape(())
                    )
                    assert np.array_equal(constant(layer + ".xz"), expected_zero)
                    assert np.array_equal(
                        constant(layer + ".ws"), module.weight_fake_quant.scale.detach().numpy()
                    )
                    weight_dq = next(
                        n
                        for n in graph.graph.node
                        if n.op_type == "DequantizeLinear" and n.input[0] == layer + ".weight"
                    )
                    assert np.all(constant(weight_dq.input[2]) == 128)
            for output_quant in ["center_quant", "regression_quant"]:
                node = next(
                    n for n in graph.graph.node if n.name == "/" + output_quant + "/QuantizeLinear"
                )
                original = modules[output_quant]
                assert np.array_equal(constant(node.input[1]), original.scale.detach().numpy())
                assert np.array_equal(
                    constant(node.input[2]), original.zero_point.detach().numpy().astype(np.int8)
                )
            for key, qp in item["retained_quantizers"].items():
                close(modules[key].scale.tolist(), qp["scale"])
                close(modules[key].zero_point.tolist(), qp["zero_point"])
            reference = np.load(checked(exports["candidates"][name]["policy_reference"]))
            parity = pilot.numerical(reference, output)
            close(parity, exports["candidates"][name]["export_parity"])
            delta = (
                pilot.line_metrics(output, data["ground_truth"])["sAP10_subset_percent"]
                - pilot.line_metrics(reference, data["ground_truth"])["sAP10_subset_percent"]
            )
            close(delta, exports["candidates"][name]["export_sAP10_delta_pp"])
            assert exports["candidates"][name]["export_gate_passed"] == (
                parity["relative_l2"] <= 0.02 and abs(delta) <= 0.5
            )
    dev, timings = development["models"], benchmark["models"]
    eligible = [
        name
        for name in ["selective_a", "selective_b"]
        if dev[name]["sAP10"]["sAP10_subset_percent"]
        >= dev["qdq_all"]["sAP10"]["sAP10_subset_percent"] - 0.5
        and timings[name]["median_ms"] < timings["qdq_all"]["median_ms"]
    ]
    chosen = min(eligible, key=lambda name: timings[name]["median_ms"]) if eligible else "qdq_all"
    assert eligible == selection["eligible"] and chosen == selection["selected"]
    manifest = read(checked(regression["manifest"]))
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    old = read(DAY16 / "results/mlsd_final.json")
    regression_outputs = {}
    for name, item in regression["models"].items():
        output = np.load(checked(item["output"]))
        regression_outputs[name] = output
        close(pilot.line_metrics(output, truth), item["sAP10"])
        if name in ["teacher_fp32", "qdq_all"]:
            previous = old["models"][
                "original_fp32" if name == "teacher_fp32" else "qat_selected_qdq"
            ]
            assert np.array_equal(output, np.load(checked(previous["output"])))
    for name in ["selective_a", "selective_b"]:
        close(
            paired(regression_outputs["qdq_all"], regression_outputs[name], truth),
            regression["paired_comparisons"][name + "_vs_qdq"],
        )
    control = read(RESULTS / "fusion_control.json")
    checked(control["protocol"])
    checked(control["same_artifact"])
    control_outputs = {}
    for name, item in control["conditions"].items():
        assert item["runs"] == len(item["samples_ms"]) == 80
        close(item["median_ms"], float(np.median(item["samples_ms"])))
        close(item["p95_ms"], float(np.percentile(item["samples_ms"], 95)))
        actual_profile = summarize_profile(checked(item["profile"]["trace"]))
        close(
            actual_profile,
            {key: value for key, value in item["profile"].items() if key != "optimized_model"},
        )
        checked(item["profile"]["optimized_model"])
        output = np.load(checked(item["development_output"]))
        close(pilot.line_metrics(output, data["ground_truth"]), item["sAP10"])
        control_outputs[name] = output
    assert (
        control["conditions"]["enabled"]["profile"]["unique_operators"].get("QLinearConv", 0) == 22
    )
    assert (
        control["conditions"]["disabled"]["profile"]["unique_operators"].get("QLinearConv", 0) == 0
    )
    close(
        pilot.numerical(control_outputs["disabled"], control_outputs["enabled"]),
        control["numeric_difference"],
    )
    archive_path = RESULTS / "source_archive.json"
    archive = read(archive_path) if archive_path.exists() else {"files": []}
    import hashlib

    for item in archive["files"]:
        assert hashlib.sha256(item["utf8_source"].encode()).hexdigest() == item["sha256"]
    archived_hashes = {item["sha256"] for item in archive["files"]}
    for record in [
        diagnosis,
        exports,
        development,
        benchmark,
        selection,
        regression,
        control,
        read(RESULTS / "fusion_control_protocol.json"),
    ]:
        source = record["source_code"]
        current = ROOT / source["path"]
        assert info(current)["sha256"] == source["sha256"] or source["sha256"] in archived_hashes
    # Re-execute all changed policies on development; selected model also on regression.
    reruns = []
    for name in ["selective_a", "selective_b", "qat_weights_fp32"]:
        item = exports["models"][name]
        actual = infer(checked(item["model"]), data["evaluation"][0].numpy())
        expected = np.load(checked(development["models"][name]["output"]))
        assert np.array_equal(actual, expected), name
        reruns.append(name + ":16 development exact")
    actual = infer(checked(exports["models"][chosen]["model"]), inputs)
    assert np.array_equal(actual, regression_outputs[chosen])
    reruns.append(chosen + ":54 regression exact")
    for item in frozen["files"]:
        checked(item)
    source = {p.name: info(p) for p in Path(__file__).parent.glob("*.py")}
    write(
        RESULTS / "verification.json",
        {
            "verified_at_utc": now(),
            "status": "passed",
            "frozen_source_files_unchanged": len(frozen["files"]),
            "candidate_policy_weights_and_retained_scales_checked": True,
            "kernel_dtypes_and_counts": profile_counts,
            "reruns": reruns,
            "metric_bootstrap_timing_recomputed": True,
            "development_selection_recomputed": chosen,
            "export_release_gates": {
                n: exports["candidates"][n]["export_gate_passed"]
                for n in ["selective_a", "selective_b"]
            },
            "scope": "Evidence consistency passed; failed model release gates remain failed; regression is not new holdout",
            "sources": source,
        },
    )
    print("VERIFIED", chosen, flush=True)


if __name__ == "__main__":
    main()
