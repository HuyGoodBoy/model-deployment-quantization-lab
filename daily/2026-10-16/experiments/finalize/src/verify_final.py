"""Recompute final metrics and verify disjoint data, selection order and precision."""

import platform
import sys
from datetime import UTC, datetime

import numpy as np
from common import ART, DAY15, RESULTS, ROOT, checked, info, read, sha, wilson, write
from evaluate_final import evaluator_cases, line_bootstrap
from train_continue import pilot


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
        # Saving a transposed tensor changes reduction order after a contiguous reload.
        assert np.isclose(expected, actual, rtol=1e-10, atol=1e-10), (expected, actual)
    else:
        assert expected == actual, (expected, actual)


def main():
    import onnx
    import onnxruntime
    import torch

    write(
        RESULTS / "environment.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "provider": "CPUExecutionProvider",
            "threads": {"intra": 4, "inter": 1},
            "spinning_disabled": True,
        },
    )
    checks = evaluator_cases()
    assert len(checks) == 9
    ml_manifest = read(RESULTS / "mlsd_manifest.json")
    ml = read(RESULTS / "mlsd_final.json")
    selection = read(RESULTS / "mlsd_selection.json")
    assert selection["locked_at_utc"] < ml["evaluated_at_utc"]
    assert not selection["holdout_used"]
    checked(ml["manifest"])
    checked(ml["selection"])
    checked(ml_manifest["inputs"])
    truth = read(checked(ml_manifest["ground_truth"]))
    old_manifest = read(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/results/data_manifest.json")
    old_holdout = read(DAY15 / "results/research/holdout_manifest.json")
    old_ids = {r["image_file"] for r in old_manifest["splits"]["evaluation"]["records"]} | {
        r["image_file"] for r in old_holdout["records"]
    }
    old_hashes = {
        r["image_sha256"] for group in old_manifest["splits"].values() for r in group["records"]
    } | {r["image_sha256"] for r in old_holdout["records"]}
    assert not old_ids & {r["image_file"] for r in ml_manifest["records"]}
    assert not old_hashes & {r["image_sha256"] for r in ml_manifest["records"]}
    assert (
        len(truth)
        == ml["samples"]
        == len({r["image_sha256"] for r in ml_manifest["records"]})
        == 54
    )
    reference = np.load(checked(ml["models"]["original_fp32"]["output"]))
    outputs = {}
    for name, entry in ml["models"].items():
        checked(entry["model"])
        output = np.load(checked(entry["output"]))
        assert len(output) == 54 and np.isfinite(output).all()
        compare(
            entry["metrics"],
            pilot.evaluate(output, {"reference": reference, "ground_truth": truth}, "mlsd"),
        )
        outputs[name] = output
    for name, expected in ml["paired_comparisons"].items():
        a, b = name.split("_to_")
        assert line_bootstrap(outputs[a], outputs[b], truth) == expected
    visual = read(RESULTS / "visual_examples.json")
    assert [row["index"] for row in visual["examples"]] == [0, 17, 35, 53]
    for row in visual["examples"]:
        index = row["index"]
        for name, expected in row["models"].items():
            compare(expected, pilot.line_metrics(outputs[name][index : index + 1], [truth[index]]))
    tx_manifest = read(RESULTS / "text_manifest.json")
    tx = read(RESULTS / "text_final.json")
    tx_selection = read(RESULTS / "text_selection.json")
    assert (
        tx_selection["selected_at_utc"] < tx["evaluated_at_utc"]
        and not tx_selection["holdout_used"]
    )
    checked(tx["manifest"])
    checked(tx["export"])
    checked(tx["selection"])
    checked(tx_manifest["source"])
    rows = tx_manifest["records"]
    hashes = {r["sentence_sha256"] for r in rows}
    old_text = read(ROOT / "quantization/data/text/manifest.json")
    train = read(ROOT / "daily/2026-10-14/experiments/qat/results/data_manifest.json")
    import hashlib

    from prepare_final import normalize

    forbidden = {
        hashlib.sha256(normalize(r["sentence"]).encode()).hexdigest()
        for group in old_text["records"].values()
        for r in group
    }
    forbidden.update(r["sentence_sha256"] for r in train["records"])
    assert not hashes & forbidden and len(hashes) == len(rows) == tx["samples"] == 772
    inputs = np.load(checked(tx_manifest["inputs"]))
    labels = inputs["labels"]
    assert labels.tolist() == [r["label"] for r in rows]
    text_outputs = {}
    for name, entry in tx["models"].items():
        output = np.load(checked(entry["output"]))
        correct = int(np.count_nonzero(output.argmax(-1) == labels))
        assert correct == entry["correct"]
        assert correct / len(labels) * 100 == entry["accuracy_percent"]
        assert wilson(correct, len(labels)) == entry["wilson_95_ci_percent"]
        text_outputs[name] = output
    a, b = text_outputs["qat_fake_quant"], text_outputs["qat_static_integer"]
    assert pilot.numerical(a, b) == tx["fake_quant_to_integer_parity"]
    assert float(np.mean(a.argmax(-1) == b.argmax(-1))) == tx["fake_quant_integer_label_agreement"]
    ml_ops = selection["qdq_runtime"]["unique_executed_operators"]
    tx_ops = tx_selection["runtime"]["unique_executed_operators"]
    assert ml_ops["QLinearConv"] == 22 and ml_ops["FusedConv"] + ml_ops["Conv"] == 19
    assert sum(n for op, n in tx_ops.items() if "MatMulInteger" in op) == 38
    for profile in [selection["qdq_runtime"], tx_selection["runtime"]]:
        events = read(checked(profile["profile"]))
        assert any(e.get("args", {}).get("provider") == "CPUExecutionProvider" for e in events)
    for task in ["mlsd", "text"]:
        timing = read(RESULTS / f"{task}_benchmark.json")
        for row in timing["results"].values():
            checked(row["model"])
            assert row["runs"] == len(row["samples_ms"]) == 80
            assert np.median(row["samples_ms"]) == row["median_ms"]
            assert np.percentile(row["samples_ms"], 95) == row["p95_ms"]
    gpu_path = RESULTS / "gpu/verification.json"
    gpu_evidence = None
    if gpu_path.exists():
        gpu = read(gpu_path)
        assert gpu["status"] == "passed" and len(gpu["rows"]) == 14
        for item in gpu["checked_results"].values():
            checked(item)
        for path, expected in gpu["source_sha256"].items():
            assert sha(ROOT / path) == expected
        gpu_evidence = info(gpu_path)
    write(
        RESULTS / "verification.json",
        {
            "status": "passed",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "hand_computed_evaluator_cases": len(checks),
            "mlsd_final_images": 54,
            "text_final_sentences": 772,
            "selection_precedes_final_evaluation": True,
            "mlsd_executed_integer_float_convs": [22, 19],
            "text_executed_integer_linear": 38,
            "mlsd_release_gate_passed": ml["selected_development_release_gate_passed"],
            "text_release_gate_passed": tx_selection["release_gate_passed"],
            "evidence_consistent_not_all_model_gates_passed": True,
            "gpu_verification": gpu_evidence,
            "source_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                for p in sorted((ART.parent / "src").glob("*.py"))
            },
            "checked_result_sha256": {
                p.name: sha(p)
                for p in sorted(RESULTS.glob("*.json"))
                if p.name != "verification.json"
            },
        },
    )
    print(
        "PASS final data separation, metrics, paired comparisons, timing and runtime precision",
        flush=True,
    )


if __name__ == "__main__":
    main()
