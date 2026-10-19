"""Evaluate locked policies on the already-used 54-image regression set."""

import gc

import numpy as np
from support import DAY16, RESULTS, checked, infer, info, now, pilot, read, save_output, write

# isort: split
from evaluate_research import aggregate_ap, per_image_stats


def paired(a, b, truth):
    sa, sb = per_image_stats(a, truth), per_image_stats(b, truth)
    ids = np.arange(len(truth))
    rng = np.random.default_rng(20261019)
    samples = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    deltas = [aggregate_ap(sb, ix) - aggregate_ap(sa, ix) for ix in samples]
    return {
        "delta_sAP10_pp": aggregate_ap(sb, ids) - aggregate_ap(sa, ids),
        "paired_image_bootstrap_95_ci_pp": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "resamples": 1000,
        "seed": 20261019,
        "images": len(truth),
        "scope": "Already-used regression subset; not independent new holdout or population proof",
    }


def main():
    if (RESULTS / "regression.json").exists():
        raise FileExistsError("Regression already evaluated")
    selection = read(RESULTS / "selection.json")
    checked(selection["development"])
    checked(selection["benchmark"])
    exports = read(RESULTS / "exports.json")
    manifest = read(DAY16 / "results/mlsd_manifest.json")
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    started_at = now()
    rows, outputs = {}, {}
    for name in ["teacher_fp32", "qdq_all", "selective_a", "selective_b", "qat_weights_fp32"]:
        item = exports["models"][name]
        output = infer(checked(item["model"]), inputs)
        outputs[name] = output
        rows[name] = {
            "output": save_output(name + "_regression", output),
            "sAP10": pilot.line_metrics(output, truth),
        }
        print("REGRESSION", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
        gc.collect()
    comparisons = {
        name + "_vs_qdq": paired(outputs["qdq_all"], outputs[name], truth)
        for name in ["selective_a", "selective_b"]
    }
    write(
        RESULTS / "regression.json",
        {
            "regression_started_at_utc": started_at,
            "evaluated_at_utc": now(),
            "images": len(truth),
            "selection": info(RESULTS / "selection.json"),
            "selected_before_regression": selection["selected"],
            "manifest": info(DAY16 / "results/mlsd_manifest.json"),
            "models": rows,
            "paired_comparisons": comparisons,
            "scope": "54 images from Oct16 reused strictly after candidate lock; regression, not fresh holdout. Both fixed candidates shown without post-regression re-selection",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
