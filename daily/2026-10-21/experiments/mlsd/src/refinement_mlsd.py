"""Predeclared pruned single-output precision sweep; gate before timing/regression."""

import argparse
import gc
import hashlib
import re
from collections import Counter

import numpy as np
from ai_edge_quantizer import qtyping, quantizer, recipe
from export_candidates import calibrate, graph
from runtime21 import Runner, dequantize, infer, io_details, quantize
from support21 import ART, RESULTS, checked, info, now, pilot_data, read, write


class RefinedRunner(Runner):
    def run(self, image):
        if tuple(self.outputs.get("output", {}).get("shape", [])) != (1, 256, 256, 5):
            return super().run(image)
        assert image.shape == (4, 512, 512) and image.dtype == np.float32
        result = self.signature(input=quantize(image[None], self.inputs["input"]))
        value = dequantize(result["output"], self.outputs["output"])
        return value.transpose(0, 3, 1, 2)


def save(name, values):
    path = ART / "refinement" / (name + ".npz")
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, output=values)
    return info(path)


def control():
    target = RESULTS / "refinement_prune_parity.json"
    if target.exists():
        raise FileExistsError(target)
    pilot, data = pilot_data()
    old = read(RESULTS / "development.json")
    with np.load(checked(old["models"]["float"]["output"])) as stored:
        reference = stored["output"]
    model = checked(read(RESULTS / "refinement_prune_conversion.json")["tflite"])
    runner = RefinedRunner(model)
    output = infer(runner, data["evaluation"][0].numpy())
    parity = pilot.numerical(reference, output)
    quality = pilot.line_metrics(output, data["ground_truth"])
    assert parity["relative_l2"] < 1e-4
    assert (
        abs(
            quality["sAP10_subset_percent"]
            - old["models"]["float"]["sAP10"]["sAP10_subset_percent"]
        )
        < 0.05
    )
    write(
        target,
        {
            "finished_at_utc": now(),
            "model": info(model),
            "reference": old["models"]["float"],
            "output": save("pruned_float_development", output),
            "parity": parity,
            "output_exact": bool(np.array_equal(reference, output)),
            "sAP10": quality,
            "native_IO": io_details(runner),
            "source_code": info(__file__),
        },
    )
    print("PRUNE CONTROL", parity, quality["sAP10_subset_percent"], flush=True)


def export_one(name):
    assert name in ["C1", "C2", "C3"]
    target = RESULTS / ("refinement_" + name + ".json")
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "refinement_prune_parity.json")["parity"]["relative_l2"] < 1e-4
    if name == "C3":
        assert read(RESULTS / "refinement_C1.json")["quality_passed"]
        assert not read(RESULTS / "refinement_C2.json")["quality_passed"]
    pilot, data = pilot_data()
    model = checked(read(RESULTS / "refinement_prune_conversion.json")["tflite"])
    source_graph = graph(model)
    if name == "C1":
        frozen = [r for r in source_graph if "/backbone/" not in r.get("weight", r["scope"])]
    else:
        marker = "pruned_active_conv_conv_filter" if name == "C2" else "/block16/"
        first = next(i for i, r in enumerate(source_graph) if marker in r.get("weight", r["scope"]))
        frozen = source_graph[first:]
    expected_float = {r["weight"] for r in frozen if "weight" in r}
    assert len(expected_float) == (11 if name == "C1" else 1 if name == "C2" else 3)
    q = quantizer.Quantizer(model, recipe.static_wi8_ai8())
    for row in frozen:
        assert row["scope"]
        q.update_quantization_recipe(
            "^" + re.escape(row["scope"]) + "$",
            qtyping.TFLOperationName.ALL_SUPPORTED,
            algorithm_key="no_quantize",
        )
    calibration = data["calibration"][0].numpy()
    inputs = data["evaluation"][0].numpy()
    stats = calibrate(q, calibration)
    result = q.quantize(calibration_result=stats, enable_progress_bar=False)
    path = ART / "refinement" / (name + ".tflite")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.export_model(str(path), overwrite=False)
    rows = graph(path)
    convs = [r for r in rows if "weight" in r]
    assert len(convs) == 41
    actual_float = [r for r in convs if "float32" in r["weight_dtype"]]
    assert len(actual_float) == len(expected_float)
    for row in convs:
        expected = "float32" if row["weight"] in expected_float else "int8"
        assert expected in row["weight_dtype"], row
        assert all(expected in v for v in row["output_dtypes"]), row
    runner = RefinedRunner(path)
    output = infer(runner, inputs)
    quality = pilot.line_metrics(output, data["ground_truth"])
    reference = read(RESULTS / "development.json")["models"]["float"]["sAP10"][
        "sAP10_subset_percent"
    ]
    passed = quality["sAP10_subset_percent"] >= reference - 0.5
    write(
        target,
        {
            "finished_at_utc": now(),
            "name": name,
            "model": info(path),
            "float_scopes": [r["scope"] for r in frozen],
            "graph": rows,
            "conv_weight_dtypes": dict(Counter(r["weight_dtype"] for r in convs)),
            "native_IO": io_details(runner),
            "sAP10": quality,
            "quality_passed": passed,
            "quality_gate_loss_pp": 0.5,
            "reference_sAP10_percent": reference,
            "loss_pp": reference - quality["sAP10_subset_percent"],
            "development_output": save(name + "_development", output),
            "calibration_images": len(calibration),
            "development_images": len(inputs),
            "calibration_sha256": hashlib.sha256(calibration.tobytes()).hexdigest(),
            "development_input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
            "source_model": info(model),
            "source_code": info(__file__),
            "scope": "Development-only fixed gate,not non-inferiority;new PTQ same32calibration;no regression/timing-based tuning.",
        },
    )
    print(
        "CANDIDATE", name, quality["sAP10_subset_percent"], "PASS" if passed else "FAIL", flush=True
    )
    del runner, q, result, stats
    gc.collect()


def freeze():
    target = RESULTS / "refinement_quality_selection.json"
    if target.exists():
        raise FileExistsError(target)
    c1, c2 = (read(RESULTS / ("refinement_" + n + ".json")) for n in ["C1", "C2"])
    names = ["C1", "C2"]
    if c1["quality_passed"] and not c2["quality_passed"]:
        names.append("C3")
    candidates = {n: read(RESULTS / ("refinement_" + n + ".json")) for n in names}
    eligible = [n for n, r in candidates.items() if r["quality_passed"]]
    write(
        target,
        {
            "locked_at_utc": now(),
            "quality_eligible": eligible,
            "candidates": {n: info(RESULTS / ("refinement_" + n + ".json")) for n in names},
            "C3_condition": "C1 passes and C2 fails;otherwise skipped",
            "failed_skipped_full_benchmark_regression": [n for n in names if n not in eligible],
            "rule": "Lock quality before new matched timing/regression;compare speed to FP32 as well as prior A,never reselect on regression.",
            "source_code": info(__file__),
        },
    )
    print("QUALITY LOCKED", eligible, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["control", "C1", "C2", "C3", "freeze"])
    args = parser.parse_args()
    if args.stage == "control":
        control()
    elif args.stage == "freeze":
        freeze()
    else:
        export_one(args.stage)
