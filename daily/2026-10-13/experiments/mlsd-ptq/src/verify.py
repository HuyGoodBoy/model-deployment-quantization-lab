"""Recompute reported PTQ/QAT metrics from local outputs; no new inference/training."""

import hashlib
import json
import platform

import numpy as np
from metrics import line_metrics, numerical

from data import ROOT, TASK, load, sha, write_json


def metric_checks():
    gt = [{"image_size": [128, 128], "lines": [[[44, 64], [84, 64]]]}]
    raw = np.zeros((256, 256, 5), np.float32)
    raw[:, :, 0] = -20
    raw[128, 128, 0] = 10
    raw[128, 128, 1:] = [-40, 0, 40, 0]
    assert line_metrics(raw[None], gt)["sAP10_subset_percent"] == 100
    raw[128, 128, 1:] = [40, 0, -40, 0]
    assert line_metrics(raw[None], gt)["sAP10_subset_percent"] == 100
    raw[128, 128, 1:] = [-40, 20, 40, 20]
    assert line_metrics(raw[None], gt)["sAP10_subset_percent"] == 0
    assert (
        line_metrics(raw[None], [{"image_size": [128, 128], "lines": []}])["sAP10_subset_percent"]
        == 0
    )
    blank = raw.copy()
    blank[:, :, 0] = -20
    assert line_metrics(blank[None], gt)["sAP10_subset_percent"] == 0
    exact = numerical(np.ones(3), np.ones(3))
    assert exact["snr_db"] is None and exact["exact_match"]
    assert abs(numerical(np.ones(3), np.ones(3) + 0.1)["snr_db"] - 20) < 1e-10
    return 7


def compare_numbers(expected, actual):
    for key in expected:
        if key == "exact_match":
            continue
        a, b = expected[key], actual[key]
        if isinstance(a, (float, int)) and not isinstance(a, bool):
            if not np.isclose(a, b, rtol=1e-8, atol=1e-8):
                raise AssertionError((key, a, b))
        elif a != b:
            raise AssertionError((key, a, b))


def main():
    checks = metric_checks()
    _inputs, ground_truth, manifest = load()
    reference = np.load(TASK / "artifacts/reference_evaluation.npy", allow_pickle=False)
    summary = json.loads((TASK / "results/summary.json").read_text(encoding="utf-8"))
    verified = []
    for entry in summary["variants"]:
        name = entry["variant"]
        record = json.loads((TASK / "results" / (name + ".json")).read_text(encoding="utf-8"))
        model_path = (
            ROOT
            / "daily/2026-10-09/experiments/03-mlsd/artifacts"
            / (name.removeprefix("original_") + ".tflite")
            if name.startswith("original_") and name != "original_static"
            else ROOT / "daily/2026-10-09/experiments/03-mlsd/artifacts/static.tflite"
            if name == "original_static"
            else TASK / "artifacts" / (name + ".tflite")
        )
        assert sha(model_path) == record["model_sha256"]
        path = TASK / "artifacts" / ("output_" + name + ".npy")
        assert sha(path) == record["output_sha256"]
        output = np.load(path, allow_pickle=False)
        for part, selection in [
            ("active", slice(0, 5)),
            ("center", slice(0, 1)),
            ("displacement", slice(1, 5)),
        ]:
            compare_numbers(
                record["metrics"][part],
                numerical(reference[..., selection], output[..., selection]),
            )
        compare_numbers(record["line_quality"], line_metrics(output, ground_truth))
        assert (
            abs(record["benchmark"]["median_ms"] - np.median(record["benchmark"]["samples_ms"]))
            < 1e-9
        )
        assert (
            abs(
                record["benchmark"]["p95_ms"] - np.percentile(record["benchmark"]["samples_ms"], 95)
            )
            < 1e-9
        )
        if name in ["active5_int8", "split_int8", "scaled_split_int8", "recalibrated_int8"]:
            assert not record["audit"]["tensor_dtypes"].get("float32", 0)
            assert record["audit"]["input"]["dtype"] == "int8"
        verified.append(name)
    gates = []
    for name in ["active5", "split", "scaled_split"]:
        gate = json.loads(
            (TASK / "results" / ("gate_" + name + ".json")).read_text(encoding="utf-8")
        )
        assert gate["examples"] == 16 and gate["passed"] and gate["relative_l2"] < 1e-4
        gates.append(name)
    qat_task = ROOT / "daily/2026-10-14/experiments/qat"
    text_manifest_path = qat_task / "results/data_manifest.json"
    text_split_verified = False
    if text_manifest_path.exists():
        text_manifest = json.loads(text_manifest_path.read_text(encoding="utf-8"))
        assert sha(qat_task / "artifacts/text_train.npz") == text_manifest["train_file_sha256"]
        assert (
            sha(qat_task / "artifacts/distilbert_weights.npz")
            == text_manifest["transposed_weights_sha256"]
        )
        old_manifest = json.loads(
            (ROOT / "quantization/data/text/manifest.json").read_text(encoding="utf-8")
        )
        forbidden = {
            hashlib.sha256(" ".join(r["sentence"].lower().split()).encode()).hexdigest()
            for group in old_manifest["records"].values()
            for r in group
        }
        actual_train = {r["sentence_sha256"] for r in text_manifest["records"]}
        assert len(actual_train) == 256 and not (actual_train & forbidden)
        text_split_verified = True
    qat_checks = []
    for task in ["mlsd", "text"]:
        progress = qat_task / "results" / (task + "_progress.json")
        if not progress.exists():
            continue
        record = json.loads(progress.read_text(encoding="utf-8"))
        assert record["status"] == "in_progress" and record["steps_this_run"] == 8
        assert record["weights_before_sha256"] != record["weights_after_sha256"]
        assert (
            sha(qat_task / "artifacts" / (task + "_qat_checkpoint.pt"))
            == record["checkpoint_sha256"]
        )
        assert len(record["history"]) == 8
        assert all(
            np.isfinite(h["loss"]) and np.isfinite(h["gradient_norm_before_clip"])
            for h in record["history"]
        )
        for phase in ["before", "after"]:
            path = qat_task / "artifacts" / (task + "_" + phase + ".npy")
            assert sha(path) == record["output_hashes"][phase]
            output = np.load(path, allow_pickle=False)
            if task == "mlsd":
                compare_numbers(
                    record["fake_quant_" + phase]["active"], numerical(reference, output)
                )
                compare_numbers(
                    record["fake_quant_" + phase]["line_quality"],
                    line_metrics(output, ground_truth),
                )
                for part, selection in [("center", slice(0, 1)), ("displacement", slice(1, 5))]:
                    compare_numbers(
                        record["fake_quant_" + phase][part],
                        numerical(reference[..., selection], output[..., selection]),
                    )
            else:
                with np.load(
                    ROOT / "quantization/results/text/outputs.npz", allow_pickle=False
                ) as saved:
                    text_reference = saved["tensorflow_fp32"]
                    compare_numbers(
                        record["fake_quant_" + phase]["logits"], numerical(text_reference, output)
                    )
                labels = np.load(
                    ROOT / "quantization/data/text/evaluation.npz", allow_pickle=False
                )["labels"]
                assert (
                    abs(
                        record["fake_quant_" + phase]["accuracy"]
                        - np.mean(output.argmax(-1) == labels)
                    )
                    < 1e-12
                )
        qat_checks.append(task)
    result = {
        "status": "passed",
        "metric_unit_checks": checks,
        "ptq_variants_recomputed": verified,
        "qat_progress_verified": qat_checks,
        "fp32_head_gates": gates,
        "text_train_eval_calibration_split_verified": text_split_verified,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "input_sha256": manifest["inputs_sha256"],
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in [
                *sorted((TASK / "src").glob("*.py")),
                *sorted((qat_task / "src").glob("*.py")),
            ]
        },
        "scope": "recomputed output metrics, AP, raw timing percentiles, INT8 graph dtype audit, QAT finite losses/weight changes/checkpoint hashes; no new training or inference",
    }
    write_json(TASK / "results/verification.json", result)
    print(
        f"PASS: {checks} metric checks, {len(verified)} PTQ variants, QAT progress {qat_checks}",
        flush=True,
    )


if __name__ == "__main__":
    main()
