"""Lock unchanged-data quality gates before timing or reused regression."""

import argparse
import gc

import numpy as np
from mlsd22 import DAY21, RESULTS, checked, info, now, pilot_data, read, references, save, write
from refinement_measure import ContiguousRunner
from runtime21 import infer, io_details


def control():
    target = RESULTS / "roundtrip_control.json"
    if target.exists():
        raise FileExistsError(target)
    pilot, data = pilot_data()
    source = read(DAY21 / "results/refinement_C1.json")
    with np.load(checked(source["development_output"])) as saved:
        reference = saved["output"]
    record = read(RESULTS / "model_C1_roundtrip.json")["model"]
    runner = ContiguousRunner(checked(record))
    output = infer(runner, data["evaluation"][0].numpy())
    exact = bool(np.array_equal(reference, output))
    metric = pilot.line_metrics(output, data["ground_truth"])
    assert exact, pilot.numerical(reference, output)
    assert metric == source["sAP10"]
    write(
        target,
        {
            "finished_at_utc": now(),
            "model": record,
            "output_exact": exact,
            "output": save("roundtrip_development", output),
            "sAP10": metric,
            "native_IO": io_details(runner),
            "source_code": info(__file__),
        },
    )
    print("ROUNDTRIP CONTROL", exact, metric["sAP10_subset_percent"], flush=True)


def quality(name):
    target = RESULTS / ("quality_" + name + ".json")
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "roundtrip_control.json")["output_exact"]
    assert read(RESULTS / "quantize_probe.json")["rounding_gate_passed"]
    pilot, data = pilot_data()
    record = read(RESULTS / ("model_" + name + ".json"))["model"]
    runner = ContiguousRunner(checked(record))
    inputs = data["evaluation"][0].numpy()
    output = infer(runner, inputs)
    metric = pilot.line_metrics(output, data["ground_truth"])
    reference = read(DAY21 / "results/development.json")["models"]["float"]["sAP10"][
        "sAP10_subset_percent"
    ]
    with np.load(
        checked(read(DAY21 / "results/refinement_C1.json")["development_output"])
    ) as saved:
        c1_output = saved["output"]
    loss = reference - metric["sAP10_subset_percent"]
    passed = bool(loss <= 0.5 and np.isfinite(output).all())
    write(
        target,
        {
            "finished_at_utc": now(),
            "name": name,
            "model": record,
            "output": save(name + "_development", output),
            "sAP10": metric,
            "loss_vs_FP32_pp": loss,
            "quality_passed": passed,
            "vs_C1": pilot.numerical(c1_output, output),
            "native_IO": io_details(runner),
            "images": len(inputs),
            "source_code": info(__file__),
        },
    )
    print(
        "QUALITY",
        name,
        metric["sAP10_subset_percent"],
        "loss",
        loss,
        "PASS" if passed else "FAIL",
        flush=True,
    )
    del runner
    gc.collect()


def freeze():
    target = RESULTS / "quality_selection.json"
    if target.exists():
        raise FileExistsError(target)
    m1 = read(RESULTS / "quality_M1.json")
    names = ["M1", "M2"] if m1["quality_passed"] else ["M1"]
    candidates = {n: read(RESULTS / ("quality_" + n + ".json")) for n in names}
    models = references()
    models["C1_roundtrip"] = read(RESULTS / "model_C1_roundtrip.json")["model"]
    eligible = [n for n, r in candidates.items() if r["quality_passed"]]
    models.update({n: candidates[n]["model"] for n in eligible})
    write(
        target,
        {
            "locked_at_utc": now(),
            "models": models,
            "quality_eligible": eligible,
            "tested": names,
            "failed_skip_full_benchmark_regression": [n for n in names if n not in eligible],
            "M2_skipped_if_M1_failed": not m1["quality_passed"],
            "rule": "FP32 development loss<=0.5pp before timing;only eligible candidate runs54image reused regression;no scale/precision tuning.",
            "source_code": info(__file__),
        },
    )
    print("QUALITY LOCKED", eligible, flush=True)


def configuration():
    """Validate the actual eight-thread configuration before speed selection."""
    target = RESULTS / "configuration_validation.json"
    if target.exists():
        raise FileExistsError(target)
    pilot, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    records = read(RESULTS / "quality_selection.json")["models"]
    baseline = read(DAY21 / "results/development.json")["models"]["float"]
    references_by_name = {
        "float": baseline["output"],
        "C1": read(DAY21 / "results/refinement_C1.json")["development_output"],
        "C1_roundtrip": read(RESULTS / "roundtrip_control.json")["output"],
        **{n: read(RESULTS / f"quality_{n}.json")["output"] for n in ["M1", "M2"] if n in records},
    }
    started, rows = now(), {}
    for name, record in records.items():
        runner = ContiguousRunner(checked(record), threads=8)
        output = infer(runner, inputs)
        with np.load(checked(references_by_name[name])) as saved:
            reference = saved["output"]
        metric = pilot.line_metrics(output, data["ground_truth"])
        loss = baseline["sAP10"]["sAP10_subset_percent"] - metric["sAP10_subset_percent"]
        exact = bool(np.array_equal(reference, output))
        rows[name] = {
            "model": record,
            "reference_output": references_by_name[name],
            "output": save(name + "_development_t8", output),
            "exact_vs_four_threads": exact,
            "sAP10": metric,
            "loss_vs_FP32_pp": loss,
            "passed": bool(exact and np.isfinite(output).all() and loss <= 0.5),
        }
        del runner
        gc.collect()
        print("CONFIGURATION", name, exact, loss, flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "threads": 8,
            "images": len(inputs),
            "models": rows,
            "passed": all(row["passed"] for row in rows.values()),
            "scope": "Executed after serial timing and before speed selection; no graph, scale or gate changes.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["control", "M1", "M2", "freeze", "configuration"])
    value = parser.parse_args().stage
    {"control": control, "freeze": freeze, "configuration": configuration}.get(
        value, lambda: quality(value)
    )()
