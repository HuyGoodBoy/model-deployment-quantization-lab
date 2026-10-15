"""Independently recompute day-15 evidence, resume chains, graph coverage and timings."""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import torch
from export_integer import graph_audit
from train_continue import PILOT, ROOT, TASK, pilot, relative


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            compare(a, b)
    elif isinstance(expected, float):
        assert np.isclose(expected, actual, rtol=1e-8, atol=1e-8), (expected, actual)
    else:
        assert expected == actual, (expected, actual)


def artifact(info):
    path = ROOT / info["path"]
    assert pilot.sha(path) == info["sha256"], path
    return path


def parameter_hash(state, task):
    digest = hashlib.sha256()
    for name, value in state.items():
        if name.endswith((".weight", ".bias")) and not (
            task == "text" and "distilbert.embeddings." in name
        ):
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    verified = []
    source_differences = []
    for task, targets in [("mlsd", [32, 64, 128]), ("text", [128])]:
        data = pilot.load_datasets(task)
        old = read(PILOT / "results" / (task + "_progress.json"))
        source = PILOT / "artifacts" / (task + "_qat_checkpoint.pt")
        assert pilot.sha(source) == old["checkpoint_sha256"]
        previous_output = np.load(PILOT / "artifacts" / (task + "_after.npy"))
        assert (
            pilot.sha(PILOT / "artifacts" / (task + "_after.npy")) == old["output_hashes"]["after"]
        )
        previous_hash = old["weights_after_sha256"]
        start = 8
        batch = 1 if task == "mlsd" else 2
        order = np.random.default_rng(20261014).permutation(len(data["train"][0]))
        for target in targets:
            name = f"{task}_step{target:04d}"
            record = read(TASK / "results" / (name + ".json"))
            assert record["start_step"] == start and record["total_steps"] == target
            assert record["steps_this_run"] == target - start
            assert record["resumed_from"]["step"] == start
            assert artifact(record["resumed_from"]) == source
            assert record["weights_before_sha256"] == previous_hash
            assert record["weights_before_sha256"] != record["weights_after_sha256"]
            assert not record["data"]["evaluation_used_in_training"]
            assert len(record["history"]) == target - start
            for step, item in zip(range(start, target), record["history"]):
                assert item["step"] == step + 1
                expected_indices = order[(np.arange(batch) + step * batch) % len(order)]
                assert item["train_indices"] == expected_indices.tolist()
                assert np.isfinite(item["loss"]) and np.isfinite(item["gradient_norm_before_clip"])
            for phase in ["before", "after"]:
                output = np.load(artifact(record["outputs"][phase]))
                assert np.isfinite(output).all()
                compare(record["fake_quant_" + phase], pilot.evaluate(output, data, task))
                if phase == "before":
                    assert np.array_equal(output, previous_output), "Resume discontinuity"
            checkpoint_path = artifact(record["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert checkpoint["task"] == task and checkpoint["step"] == target
            assert parameter_hash(checkpoint["model"], task) == record["weights_after_sha256"]
            for key, value in checkpoint["model"].items():
                if key.endswith("observer_enabled"):
                    assert not value.any()
                if key.endswith("fake_quant_enabled"):
                    assert value.all()
            assert all(group["lr"] == 1e-5 for group in checkpoint["optimizer"]["param_groups"])
            del checkpoint
            for path, checksum in record["source_sha256"].items():
                if pilot.sha(ROOT / path) != checksum:
                    source_differences.append(
                        {
                            "run": name,
                            "path": path,
                            "recorded_sha256": checksum,
                            "current_sha256": pilot.sha(ROOT / path),
                        }
                    )
            source, previous_output = checkpoint_path, output
            previous_hash, start = record["weights_after_sha256"], target
            verified.append(name)

    text_manifest = read(PILOT / "results/data_manifest.json")
    assert pilot.sha(PILOT / "artifacts/text_train.npz") == text_manifest["train_file_sha256"]
    assert (
        pilot.sha(PILOT / "artifacts/distilbert_weights.npz")
        == text_manifest["transposed_weights_sha256"]
    )
    baseline_manifest = read(ROOT / "quantization/data/text/manifest.json")
    forbidden = {
        hashlib.sha256(" ".join(row["sentence"].lower().split()).encode()).hexdigest()
        for rows in baseline_manifest["records"].values()
        for row in rows
    }
    train_hashes = {row["sentence_sha256"] for row in text_manifest["records"]}
    assert len(train_hashes) == 256 and not train_hashes & forbidden

    exported = read(TASK / "results/mlsd_integer_export.json")
    paths = {name: artifact(info) for name, info in exported["artifacts"].items()}
    compare(exported["graph_audit"], graph_audit(paths["integer_model"]))
    compare(exported["optimized_graph_audit"], graph_audit(paths["optimized_integer_model"]))
    assert exported["graph_audit"]["operators"]["ConvInteger"] == 41
    assert exported["graph_audit"]["float_conv_count"] == 0
    events = read(artifact(exported["runtime_profile"]))
    kernels = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("op_name") == "ConvInteger"
    ]
    assert len(kernels) == 41 and all(
        e["args"]["provider"] == "CPUExecutionProvider" for e in kernels
    )
    data = pilot.load_datasets("mlsd")
    output = np.load(paths["integer_output"])
    compare(exported["deployed_integer_metrics"], pilot.evaluate(output, data, "mlsd"))
    step = exported["checkpoint"]["step"]
    fake = np.load(TASK / "artifacts" / f"mlsd_step{step:04d}_after.npy")
    compare(exported["fake_quant_metrics"], pilot.evaluate(fake, data, "mlsd"))
    fp32_output = np.load(paths["original_fp32_output"])
    compare(exported["original_fp32_metrics"], pilot.evaluate(fp32_output, data, "mlsd"))
    compare(exported["original_fp32_export_gate"], pilot.numerical(data["reference"], fp32_output))
    assert artifact(exported["checkpoint"]) == TASK / "artifacts" / f"mlsd_step{step:04d}.pt"
    compare(exported["fake_quant_to_integer_parity"], pilot.numerical(fake, output))
    for result in exported["benchmark"]["results"].values():
        assert len(result["samples_ms"]) == result["runs"] == 80
        assert all(np.isfinite(v) and v > 0 for v in result["samples_ms"])
        assert result["median_ms"] == np.median(result["samples_ms"])
        assert result["p95_ms"] == np.percentile(result["samples_ms"], 95)
    gate = exported["parity_gate"]
    actual_delta = (
        exported["deployed_integer_metrics"]["line_quality"]["sAP10_subset_percent"]
        - exported["fake_quant_metrics"]["line_quality"]["sAP10_subset_percent"]
    )
    assert actual_delta == gate["actual_sAP10_delta_pp"]
    expected_pass = (
        exported["fake_quant_to_integer_parity"]["relative_l2"] <= gate["maximum_relative_l2"]
        and abs(gate["actual_sAP10_delta_pp"]) <= gate["maximum_sAP10_delta_pp"]
    )
    assert gate["passed"] == expected_pass
    assert exported["is_full_integer_model"] is False
    result = {
        "status": "passed",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "training_runs": verified,
        "checks": "Checkpoint hashes/weights, contiguous resume outputs, finite losses/gradients, deterministic sample indices, recomputed quality, frozen observers, data split disjointness, original and optimized graph audits, actual integer CPU kernels, timing percentiles",
        "day14_checkpoints_unchanged": True,
        "export_parity_gate_passed": gate["passed"],
        "source_differences_since_recording": source_differences,
        "source_difference_note": "Day-15 trainer was reformatted with Ruff after M-LSD ran; executed-source checksums remain recorded. Historical day-14 sources are unchanged.",
        "source_sha256": {relative(p): pilot.sha(p) for p in sorted((TASK / "src").glob("*.py"))},
        "scope": "Evidence consistency passed; does not override a failed numerical export parity gate or establish final held-out quality",
    }
    pilot.write_json(TASK / "results/verification.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "training_runs": verified,
                "export_parity_gate_passed": gate["passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
