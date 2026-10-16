"""Evaluate previously locked deployments once on disjoint final data."""

import argparse
import gc
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import torch
from common import ART, DAY15, RESULTS, ROOT, checked, info, read, session, wilson, write
from export_integer import infer_all
from export_text import infer
from train_continue import PILOT, pilot


def evaluator_cases():
    import metrics

    gt = {"image_size": [256, 256], "lines": [[[20, 20], [60, 20]]]}
    line = np.array([[10, 10], [30, 10]], np.float64)
    cases = [
        ("reversed_endpoints", [(line[::-1], 0.9)], gt, 100, 1, 1),
        ("duplicate_prediction", [(line, 0.9), (line, 0.8)], gt, 100, 0.5, 1),
        ("false_positive_before_true", [(line + 30, 0.95), (line, 0.9)], gt, 50, 0.5, 1),
        (
            "non_square_resize",
            [(line, 0.9)],
            {"image_size": [512, 256], "lines": [[[40, 20], [120, 20]]]},
            100,
            1,
            1,
        ),
        ("strict_distance_equal_10", [(line + [1, 2], 0.9)], gt, 0, 0, 0),
        ("strict_distance_9", [(line + np.array([[1, 2], [0, 2]]), 0.9)], gt, 100, 1, 1),
        ("no_prediction", [], gt, 0, 0, 0),
        ("no_ground_truth", [(line, 0.9)], {"image_size": [256, 256], "lines": []}, 0, 0, 0),
    ]
    result = []
    for name, predictions, truth, ap, precision, recall in cases:
        with patch.object(metrics, "lines", return_value=predictions):
            actual = metrics.line_metrics([np.zeros((256, 256, 5))], [truth])
        assert np.isclose(actual["sAP10_subset_percent"], ap)
        assert np.isclose(actual["precision_at_0_1"], precision)
        assert np.isclose(actual["recall_at_0_1"], recall)
        result.append({"case": name, "expected_AP_percent": ap, "passed": True})
    raw = np.zeros((256, 256, 5), np.float32)
    raw[..., 0] = -20
    raw[100, 100] = [4, -30, 0, 30, 0]
    decoded = metrics.lines(raw)
    assert len(decoded) == 1 and np.array_equal(decoded[0][0], [[35, 50], [65, 50]])
    result.append({"case": "decoder_center_displacement_grid_scaling", "passed": True})
    return result


def line_bootstrap(a, b, truth):
    from evaluate_research import paired_bootstrap

    result = paired_bootstrap(a, b, truth)
    result["images"] = len(truth)
    result["note"] = (
        f"Uncertainty on this {len(truth)}-image subset only; not population-wide or cross-dataset proof"
    )
    return result


def mlsd():
    from evaluate_research import aggregate_ap, per_image_stats, tflite_outputs

    target = RESULTS / "mlsd_final.json"
    if target.exists():
        raise FileExistsError("Final M-LSD evaluation already recorded")
    cases = evaluator_cases()
    write(
        RESULTS / "evaluator_checks.json",
        {
            "status": "passed",
            "hand_computed_cases": cases,
            "scope": "Tests the repo's stated subset protocol, not equivalence to the paper evaluator",
        },
    )
    manifest = read(RESULTS / "mlsd_manifest.json")
    selection = read(RESULTS / "mlsd_selection.json")
    assert not selection["holdout_used"]
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    old = read(DAY15 / "results/mlsd_integer_export.json")
    calibrated = read(DAY15 / "results/research/deployment.json")
    rounding = read(DAY15 / "results/export-parity/rounding.json")
    paths = {
        "original_fp32": checked(calibrated["artifacts"]["original_fp32_model"]),
        "qat_original_integer": checked(old["artifacts"]["integer_model"]),
        "qat_calibrated_integer": checked(calibrated["artifacts"]["integer_model"]),
        "qat_explicit_rounding": checked(rounding["artifacts"]["model"]),
        "qat_selected_qdq": checked(selection["candidates"][selection["selected"]]["model"]),
    }
    records, outputs = {}, {}
    for name, path in paths.items():
        runtime = session(path)
        outputs[name] = infer_all(runtime, inputs)
        del runtime
        print("FINAL M-LSD", name, flush=True)
    ptq = ROOT / "daily/2026-10-13/experiments/mlsd-ptq/artifacts/split_int8.tflite"
    assert (
        read(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/results/split_int8.json")["model_sha256"]
        == info(ptq)["sha256"]
    )
    outputs["ptq_split_full_int8"] = tflite_outputs(inputs, ptq)
    paths["ptq_split_full_int8"] = ptq
    data = {"reference": outputs["original_fp32"], "ground_truth": truth}
    for name, output in outputs.items():
        path = ART / f"mlsd_final_{name}.npy"
        np.save(path, output, allow_pickle=False)
        cached = per_image_stats(output, truth)
        rng = np.random.default_rng(20261016)
        samples = [
            aggregate_ap(cached, rng.integers(0, len(truth), len(truth))) for _ in range(1000)
        ]
        records[name] = {
            "model": info(paths[name]),
            "output": info(path),
            "metrics": pilot.evaluate(output, data, "mlsd"),
            "image_bootstrap_sAP10_95_percentile_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
        }
        print(
            "sAP10",
            name,
            records[name]["metrics"]["line_quality"]["sAP10_subset_percent"],
            flush=True,
        )
    comparisons = {}
    for a, b in [
        ("qat_original_integer", "qat_calibrated_integer"),
        ("qat_calibrated_integer", "qat_selected_qdq"),
        ("original_fp32", "qat_selected_qdq"),
    ]:
        comparisons[a + "_to_" + b] = line_bootstrap(outputs[a], outputs[b], truth)
    # The numerical release gate was decided on development; do not tune against this final set.
    write(
        target,
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "manifest": info(RESULTS / "mlsd_manifest.json"),
            "selection": info(RESULTS / "mlsd_selection.json"),
            "samples": len(truth),
            "models": records,
            "paired_comparisons": comparisons,
            "selected_development_release_gate_passed": selection["candidates"][
                selection["selected"]
            ]["parity_gate_passed"],
            "scope": "54 previously unused recorded YorkUrban IDs; same fixed decoder, no inference-driven selection; not proof of pretraining disjointness or official full benchmark",
            "source_code": info(__file__),
        },
    )


def text():
    from models import load_text, prepare_qat
    from torch.ao.quantization import disable_observer

    target = RESULTS / "text_final.json"
    if target.exists():
        raise FileExistsError("Final text evaluation already recorded")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = read(RESULTS / "text_manifest.json")
    export = read(RESULTS / "text_export.json")
    selection = read(RESULTS / "text_selection.json")
    assert not selection["holdout_used"]
    data = np.load(checked(manifest["inputs"]))
    inputs = (data["input_ids"], data["attention_mask"])
    labels = data["labels"]
    records, outputs = {}, {}
    for name, entry in export["models"].items():
        chosen_model = (
            selection["candidates"][selection["selected"]]["model"]
            if name == "qat_static_integer"
            else entry["model"]
        )
        runtime = session(checked(chosen_model))
        outputs[name] = infer(runtime, inputs)
        del runtime
        gc.collect()
        print("FINAL TEXT", name, float(np.mean(outputs[name].argmax(-1) == labels)), flush=True)
    teacher = load_text(PILOT / "artifacts/distilbert_weights.npz")
    model, _ = prepare_qat(teacher, "text")
    del teacher
    saved = torch.load(
        checked(export["checkpoint"]), map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.apply(disable_observer)
    model.eval()
    fake = []
    with torch.no_grad():
        for index in range(len(labels)):
            fake.append(model(*(torch.from_numpy(x[index : index + 1]) for x in inputs)).numpy())
            if (index + 1) % 100 == 0:
                print("FAKE QUANT TEXT", index + 1, "/", len(labels), flush=True)
    outputs["qat_fake_quant"] = np.concatenate(fake)
    del model
    for name, output in outputs.items():
        path = ART / f"text_final_{name}.npy"
        np.save(path, output, allow_pickle=False)
        correct = int(np.count_nonzero(output.argmax(-1) == labels))
        records[name] = {
            "output": info(path),
            "correct": correct,
            "samples": len(labels),
            "accuracy_percent": correct / len(labels) * 100,
            "wilson_95_ci_percent": wilson(correct, len(labels)),
            "relative_to_teacher": pilot.numerical(outputs["teacher_fp32"], output),
        }
    a, b = outputs["qat_fake_quant"], outputs["qat_static_integer"]
    correct_a, correct_b = a.argmax(-1) == labels, b.argmax(-1) == labels
    rng = np.random.default_rng(20261016)
    delta = (correct_b.astype(float) - correct_a.astype(float)) * 100
    samples = [delta[rng.integers(0, len(labels), len(labels))].mean() for _ in range(1000)]
    write(
        target,
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "manifest": info(RESULTS / "text_manifest.json"),
            "export": info(RESULTS / "text_export.json"),
            "selection": info(RESULTS / "text_selection.json"),
            "samples": len(labels),
            "models": records,
            "fake_quant_to_integer_parity": pilot.numerical(a, b),
            "fake_quant_integer_label_agreement": float(np.mean(a.argmax(-1) == b.argmax(-1))),
            "accuracy_delta_integer_minus_fake_pp": float(delta.mean()),
            "paired_bootstrap_delta_95_ci_pp": np.percentile(samples, [2.5, 97.5]).tolist(),
            "paired_disagreements": {
                "fake_correct_integer_wrong": int((correct_a & ~correct_b).sum()),
                "integer_correct_fake_wrong": int((correct_b & ~correct_a).sum()),
            },
            "scope": "772 normalized unique validation sentences disjoint from recorded train/calibration/development; same fixed tokenizer and sequence length; no final-set tuning",
            "source_code": info(__file__),
        },
    )
    print("LOCKED FINAL TEXT", {n: r["accuracy_percent"] for n, r in records.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mlsd", "text"], required=True)
    args = parser.parse_args()
    (mlsd if args.task == "mlsd" else text)()
