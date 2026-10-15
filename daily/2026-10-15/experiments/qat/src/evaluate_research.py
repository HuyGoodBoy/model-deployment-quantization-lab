"""Select on development data, then evaluate locked candidates once on fresh images."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from export_integer import infer_all, make_session
from metrics import endpoint_distances, lines
from research_mlsd import ART, RESULTS, SOURCE, configure_range, load_model, read
from train_continue import ROOT, TASK, pilot, relative


def select():
    if (RESULTS / "selection.json").exists():
        raise FileExistsError("Development selection is already locked")
    if (RESULTS / "holdout_evaluation.json").exists():
        raise RuntimeError("Cannot select after holdout evaluation")
    calibration = read(RESULTS / "selected_policy.json")
    center = read(RESULTS / "center_calibration.json")
    zero_training = next(
        r for r in center["ablations"] if r["name"] == calibration["development_variant"]
    )
    candidates = [
        {
            "name": "range_calibration_no_new_training",
            "source": relative(SOURCE),
            "policy": calibration["policy"],
            "total_steps": 32,
            "development_sAP10": zero_training["metrics"]["line_quality"]["sAP10_subset_percent"],
            "development_output": zero_training["output"],
        }
    ]
    for variant in ["range_only", "loss_only", "range_and_loss"]:
        study = read(RESULTS / (variant + ".json"))
        for milestone in study["milestones"]:
            candidates.append(
                {
                    "name": milestone["name"],
                    "source": milestone["checkpoint"]["path"],
                    "policy": study["range_policy"],
                    "total_steps": milestone["history"][-1]["step"],
                    "development_sAP10": milestone["metrics"]["line_quality"][
                        "sAP10_subset_percent"
                    ],
                    "development_output": milestone["output"],
                }
            )
    best = max(candidates, key=lambda item: item["development_sAP10"])
    source = ROOT / best["source"]
    model, _teacher, saved, _config = load_model(source)
    if best["name"] == "range_calibration_no_new_training":
        configure_range(model, best["policy"])
    saved["model"] = model.state_dict()
    saved["research_policy"] = best["policy"]
    checkpoint = ART / "selected_checkpoint.pt"
    torch.save(saved, checkpoint)
    pilot.write_json(
        RESULTS / "selection.json",
        {
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "criterion": "Highest fixed-decoder sAP10 on the 16 development images, not holdout",
            "holdout_used": False,
            "holdout_manifest_sha256": pilot.sha(RESULTS / "holdout_manifest.json"),
            "candidates": candidates,
            "selected": best,
            "checkpoint": {"path": relative(checkpoint), "sha256": pilot.sha(checkpoint)},
            "development_output": best["development_output"],
        },
    )
    print("SELECTED", best["name"], best["development_sAP10"], flush=True)


def per_image_stats(outputs, ground_truth):
    result = []
    for output, row in zip(outputs, ground_truth):
        width, height = row["image_size"]
        gt = np.asarray(row["lines"], dtype=np.float64).reshape(-1, 2, 2) * [
            128 / width,
            128 / height,
        ]
        predicted = lines(output)
        scores = np.asarray([score for _, score in predicted], np.float64)
        tp = np.zeros(len(predicted), np.float64)
        if predicted and len(gt):
            distances = endpoint_distances(np.stack([line for line, _ in predicted]), gt)
            nearest = np.argmin(distances, axis=1)
            used = np.zeros(len(gt), bool)
            for index, target in enumerate(nearest):
                if distances[index, target] < 10 and not used[target]:
                    tp[index] = 1
                    used[target] = True
        result.append((scores, tp, len(gt)))
    return result


def aggregate_ap(image_stats, indices):
    scores = np.concatenate([image_stats[int(i)][0] for i in indices])
    tp = np.concatenate([image_stats[int(i)][1] for i in indices])
    gt = sum(image_stats[int(i)][2] for i in indices)
    tp = tp[np.argsort(-scores, kind="stable")]
    cumulative = np.cumsum(tp)
    recall = np.r_[0, cumulative / max(gt, 1), 1]
    precision = np.r_[0, cumulative / np.arange(1, len(tp) + 1), 0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]) * 100)


def paired_bootstrap(a, b, ground_truth):
    sa, sb = per_image_stats(a, ground_truth), per_image_stats(b, ground_truth)
    indices = np.arange(len(ground_truth))
    for output, cached in [(a, sa), (b, sb)]:
        assert (
            abs(
                aggregate_ap(cached, indices)
                - pilot.line_metrics(output, ground_truth)["sAP10_subset_percent"]
            )
            < 1e-10
        )
    rng = np.random.default_rng(20261015)
    deltas = []
    for _ in range(1000):
        ids = rng.integers(0, len(indices), len(indices))
        deltas.append(aggregate_ap(sb, ids) - aggregate_ap(sa, ids))
    return {
        "delta_sAP10_pp": aggregate_ap(sb, indices) - aggregate_ap(sa, indices),
        "paired_image_bootstrap_95_percentile_ci_pp": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "resamples": 1000,
        "seed": 20261015,
        "note": "Uncertainty on this 32-image subset only; not population-wide or cross-dataset proof",
    }


def tflite_outputs(inputs, path):
    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    outputs = []
    for image in inputs:
        raw = ((image.transpose(1, 2, 0) + 1) * 127.5)[None]
        scale, zero = inp["quantization"]
        raw = np.clip(np.rint(raw / scale) + zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], raw)
        interpreter.invoke()
        values = []
        for out in outs:
            scale, zero = out["quantization"]
            value = interpreter.get_tensor(out["index"])
            values.append((value.astype(np.float32) - zero) * scale)
        outputs.append(
            np.concatenate(
                [
                    next(value for value in values if value.shape[-1] == 1),
                    next(value for value in values if value.shape[-1] == 4),
                ],
                axis=-1,
            )
        )
    return np.concatenate(outputs)


def evaluate():
    result_path = RESULTS / "holdout_evaluation.json"
    if result_path.exists():
        raise FileExistsError("Holdout already evaluated; preserve the locked selection")
    selection = read(RESULTS / "selection.json")
    assert not selection["holdout_used"]
    assert pilot.sha(RESULTS / "holdout_manifest.json") == selection["holdout_manifest_sha256"]
    manifest = read(RESULTS / "holdout_manifest.json")
    for info in [manifest["inputs"], manifest["ground_truth"]]:
        assert pilot.sha(ROOT / info["path"]) == info["sha256"]
    inputs = np.load(ROOT / manifest["inputs"]["path"])
    ground_truth = read(ROOT / manifest["ground_truth"]["path"])
    model, teacher, _saved, _config = load_model(ROOT / selection["checkpoint"]["path"])
    data = {
        "ground_truth": ground_truth,
        "reference": pilot.forward_all(teacher, (torch.from_numpy(inputs),), "mlsd"),
    }
    records, outputs = {}, {"fp32": data["reference"]}
    outputs["selected_fake_quant"] = pilot.forward_all(model, (torch.from_numpy(inputs),), "mlsd")
    del model
    for step in [32, 128]:
        baseline, _teacher, _saved, _config = load_model(
            TASK / "artifacts" / f"mlsd_step{step:04d}.pt"
        )
        outputs[f"original_fake_quant_step{step}"] = pilot.forward_all(
            baseline, (torch.from_numpy(inputs),), "mlsd"
        )
        del baseline
    deployment = read(RESULTS / "deployment.json")
    original_deployment = read(TASK / "results/mlsd_integer_export.json")
    for name, exported in [
        ("original_integer", original_deployment),
        ("selected_integer", deployment),
    ]:
        path = ROOT / exported["artifacts"]["integer_model"]["path"]
        assert pilot.sha(path) == exported["artifacts"]["integer_model"]["sha256"]
        outputs[name] = infer_all(make_session(path, disable_spinning=True), inputs)
    ptq_path = pilot.DAY13 / "artifacts/split_int8.tflite"
    ptq_record = read(pilot.DAY13 / "results/split_int8.json")
    assert pilot.sha(ptq_path) == ptq_record["model_sha256"]
    outputs["ptq_split_full_int8"] = tflite_outputs(inputs, ptq_path)
    for name, output in outputs.items():
        path = ART / ("holdout_" + name + ".npy")
        np.save(path, output, allow_pickle=False)
        records[name] = {
            "metrics": pilot.evaluate(output, data, "mlsd"),
            "output": {"path": relative(path), "sha256": pilot.sha(path)},
        }
        print(
            "HOLDOUT",
            name,
            records[name]["metrics"]["line_quality"]["sAP10_subset_percent"],
            flush=True,
        )
    comparisons = {}
    for a, b in [
        ("original_fake_quant_step32", "selected_fake_quant"),
        ("original_integer", "selected_integer"),
        ("ptq_split_full_int8", "selected_integer"),
    ]:
        comparisons[a + "_to_" + b] = paired_bootstrap(outputs[a], outputs[b], ground_truth)
    pilot.write_json(
        result_path,
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "selection_sha256": pilot.sha(RESULTS / "selection.json"),
            "holdout_manifest_sha256": pilot.sha(RESULTS / "holdout_manifest.json"),
            "holdout_images": 32,
            "models": records,
            "paired_comparisons": comparisons,
            "selected_fake_quant_to_integer_parity": pilot.numerical(
                outputs["selected_fake_quant"], outputs["selected_integer"]
            ),
            "decoder": "Unchanged day13 NMS/top200/score>=.01/map-length>20 and sAP10 implementation; no threshold retuning",
            "precision_scope": "QAT/ONNX integer models keep non-Conv regions float; PTQ split TFLite is full INT8. Fake-quant quality is not integer deployment quality.",
            "limitations": "Fresh for this experiment, selected before evaluation; same YorkUrban source, small subset; original pretrained-data overlap not audited",
            "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.stage == "select":
        select()
    else:
        evaluate()
