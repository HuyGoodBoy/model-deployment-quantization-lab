"""Recompute research ablations, independent quality gains and paired uncertainty."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from evaluate_research import paired_bootstrap
from export_integer import graph_audit
from research_mlsd import RESULTS, SOURCE, read
from train_continue import ROOT, TASK, pilot, relative

spec = importlib.util.spec_from_file_location(
    "day15_evidence_verifier", Path(__file__).with_name("verify.py")
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
artifact, compare, parameter_hash = verifier.artifact, verifier.compare, verifier.parameter_hash


def verify_entry(entry, data):
    output = np.load(artifact(entry["output"]))
    assert np.isfinite(output).all()
    compare(entry["metrics"], pilot.evaluate(output, data, "mlsd"))
    return output


def main():
    torch.set_num_threads(4)
    development = pilot.load_datasets("mlsd")
    diagnostics = read(RESULTS / "diagnostics.json")
    assert artifact(diagnostics["checkpoint"]) == SOURCE
    names = []
    for filename in ["diagnostics.json", "center_calibration.json"]:
        for entry in read(RESULTS / filename)["ablations"]:
            verify_entry(entry, development)
            names.append(entry["name"])
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_weight_hash = parameter_hash(checkpoint["model"], "mlsd")
    original_center_scale = checkpoint["model"]["center_quant.scale"].tolist()
    del checkpoint
    train_hashes = {}
    selected_policy = read(RESULTS / "selected_policy.json")
    expected_order = np.random.default_rng(20261014).permutation(32)
    for variant in ["range_only", "loss_only", "range_and_loss"]:
        study = read(RESULTS / (variant + ".json"))
        assert artifact(study["source_checkpoint"]) == SOURCE
        before = verify_entry(study["before"], development)
        assert study["new_steps"] == 96 and study["train_examples"] == 32
        expected_before_name = (
            selected_policy["development_variant"] if study["range_policy"] else "scan_baseline"
        )
        entries = read(RESULTS / "center_calibration.json")["ablations"] + diagnostics["ablations"]
        expected_before = next(e for e in entries if e["name"] == expected_before_name)
        assert np.array_equal(before, np.load(artifact(expected_before["output"])))
        for entry, step in zip(study["milestones"], [64, 128]):
            verify_entry(entry, development)
            state = torch.load(
                artifact(entry["checkpoint"]), map_location="cpu", weights_only=False
            )
            assert state["step"] == step and state["task"] == "mlsd"
            assert state["research_policy"] == study["range_policy"]
            weight_hash = parameter_hash(state["model"], "mlsd")
            assert weight_hash != source_weight_hash
            train_hashes[entry["name"]] = weight_hash
            assert all(group["lr"] == 1e-5 for group in state["optimizer"]["param_groups"])
            for values in state["optimizer"]["state"].values():
                assert values["step"].item() == step
            for key, value in state["model"].items():
                if key.endswith("observer_enabled"):
                    assert not value.any()
                if key.endswith("fake_quant_enabled"):
                    assert value.all()
            del state
            assert len(entry["history"]) == step - entry["start_step"]
            for item in entry["history"]:
                assert item["train_index"] == int(expected_order[(item["step"] - 1) % 32])
                assert np.isfinite(item["loss"]) and np.isfinite(item["gradient_norm_before_clip"])
                assert np.isclose(
                    item["loss"],
                    item["soft_center_bce"] + item["positive_displacement_smooth_l1"],
                    atol=1e-7,
                )
            names.append(entry["name"])

    manifest = read(RESULTS / "holdout_manifest.json")
    selection = read(RESULTS / "selection.json")
    evaluation = read(RESULTS / "holdout_evaluation.json")
    assert evaluation["selection_sha256"] == pilot.sha(RESULTS / "selection.json")
    assert selection["holdout_manifest_sha256"] == pilot.sha(RESULTS / "holdout_manifest.json")
    assert not selection["holdout_used"]
    assert selection["selected_at_utc"] < evaluation["evaluated_at_utc"]
    assert selection["selected"]["development_sAP10"] == max(
        c["development_sAP10"] for c in selection["candidates"]
    )
    original_manifest = development["manifest"]
    forbidden = {
        r["image_sha256"] for s in original_manifest["splits"].values() for r in s["records"]
    }
    assert len(manifest["records"]) == len({r["image_file"] for r in manifest["records"]}) == 32
    assert not ({r["image_sha256"] for r in manifest["records"]} & forbidden)
    assert not (
        {r["image_file"] for r in manifest["records"]}
        & set(manifest["excluded_development_images"])
    )
    artifact(manifest["inputs"])
    ground_truth = read(artifact(manifest["ground_truth"]))
    assert len(ground_truth) == 32
    data = {
        "reference": np.load(artifact(evaluation["models"]["fp32"]["output"])),
        "ground_truth": ground_truth,
    }
    outputs = {}
    for name, entry in evaluation["models"].items():
        outputs[name] = verify_entry(entry, data)
    for a, b in [
        ("original_fake_quant_step32", "selected_fake_quant"),
        ("original_integer", "selected_integer"),
        ("ptq_split_full_int8", "selected_integer"),
    ]:
        compare(
            evaluation["paired_comparisons"][a + "_to_" + b],
            paired_bootstrap(outputs[a], outputs[b], ground_truth),
        )
    compare(
        evaluation["selected_fake_quant_to_integer_parity"],
        pilot.numerical(outputs["selected_fake_quant"], outputs["selected_integer"]),
    )
    selected = torch.load(artifact(selection["checkpoint"]), map_location="cpu", weights_only=False)
    if selection["selected"]["name"] == "range_calibration_no_new_training":
        assert parameter_hash(selected["model"], "mlsd") == source_weight_hash
    del selected

    deployment = read(RESULTS / "deployment.json")
    for info in deployment["artifacts"].values():
        artifact(info)
    compare(
        deployment["graph_audit"],
        graph_audit(ROOT / deployment["artifacts"]["integer_model"]["path"]),
    )
    compare(
        deployment["optimized_graph_audit"],
        graph_audit(ROOT / deployment["artifacts"]["optimized_integer_model"]["path"]),
    )
    assert deployment["graph_audit"]["operators"]["ConvInteger"] == 41
    assert deployment["graph_audit"]["float_conv_count"] == 0
    events = read(artifact(deployment["runtime_profile"]))
    kernels = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("op_name") == "ConvInteger"
    ]
    assert len(kernels) == 41 and all(
        k["args"]["provider"] == "CPUExecutionProvider" for k in kernels
    )
    deployed_dev = np.load(ROOT / deployment["artifacts"]["integer_output"]["path"])
    compare(
        deployment["deployed_integer_metrics"], pilot.evaluate(deployed_dev, development, "mlsd")
    )
    fake_dev = np.load(artifact(selection["development_output"]))
    compare(deployment["fake_quant_metrics"], pilot.evaluate(fake_dev, development, "mlsd"))
    compare(deployment["fake_quant_to_integer_parity"], pilot.numerical(fake_dev, deployed_dev))
    gate = deployment["parity_gate"]
    assert gate["passed"] == (
        deployment["fake_quant_to_integer_parity"]["relative_l2"] <= 0.02
        and abs(gate["actual_sAP10_delta_pp"]) <= 0.5
    )
    for filename, section in [("deployment.json", "benchmark"), ("matched_benchmark.json", None)]:
        timing = read(RESULTS / filename)
        if section:
            timing = timing[section]
        for values in timing["results"].values():
            assert values["runs"] == len(values["samples_ms"]) == 80
            assert values["median_ms"] == np.median(values["samples_ms"])
            assert values["p95_ms"] == np.percentile(values["samples_ms"], 95)
    pilot.write_json(
        RESULTS / "verification.json",
        {
            "status": "passed",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "development_ablations_recomputed": names,
            "holdout_models_recomputed": list(outputs),
            "paired_bootstrap_recomputed": True,
            "holdout_disjoint_and_selection_locked": True,
            "weights_changed_by_training": train_hashes,
            "selected_weights_sha256": source_weight_hash,
            "original_center_scale": original_center_scale,
            "integer_convs_executed": 41,
            "numerical_export_parity_gate_passed": gate["passed"],
            "scope": "Evidence, data separation, controls and quality gains verified; mixed precision and failed dense-output parity gate remain explicit",
            "source_sha256": {
                relative(p): pilot.sha(p) for p in sorted((TASK / "src").glob("*.py"))
            },
        },
    )
    print(
        "PASS: research ablations, fresh holdout, bootstrap, checkpoints and 41 integer convs",
        flush=True,
    )


if __name__ == "__main__":
    main()
