"""Render a standalone evidence figure from the measured JSON results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK = Path(__file__).resolve().parents[1]
DAY = TASK.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results = TASK / "results"
    holdout = read(results / "research/holdout_evaluation.json")
    old = read(results / "mlsd_integer_export.json")
    new = read(results / "research/deployment.json")
    ptq = read(DAY.parent / "2026-10-13/experiments/mlsd-ptq/results/split_int8.json")
    labels = ["FP32", "PTQ full INT8", "QAT integer (old)*", "QAT integer + ranges*"]
    dev = [
        new["original_fp32_metrics"]["line_quality"]["sAP10_subset_percent"],
        ptq["line_quality"]["sAP10_subset_percent"],
        old["deployed_integer_metrics"]["line_quality"]["sAP10_subset_percent"],
        new["deployed_integer_metrics"]["line_quality"]["sAP10_subset_percent"],
    ]
    fresh = [
        holdout["models"][key]["metrics"]["line_quality"]["sAP10_subset_percent"]
        for key in ["fp32", "ptq_split_full_int8", "original_integer", "selected_integer"]
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    colors = ["#64748b", "#a5b4fc", "#d4926d", "#168577"]
    for axis, values, title in zip(
        axes, [dev, fresh], ["Development: 16 images", "Fresh evaluation: 32 images"]
    ):
        bars = axis.barh(labels, values, color=colors, height=0.62)
        axis.invert_yaxis()
        axis.set_xlim(0, 18)
        axis.set_xlabel("sAP10 (%) — unchanged decoder")
        axis.set_title(title, loc="left", fontweight="bold", pad=14)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.16)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.22,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontweight="bold",
            )
    fig.suptitle(
        "M-LSD: task-aware output calibration improves deployed quality",
        fontweight="bold",
        fontsize=15,
    )
    ci = holdout["paired_comparisons"]["original_integer_to_selected_integer"]
    low, high = ci["paired_image_bootstrap_95_percentile_ci_pp"]
    fig.supxlabel(
        f"* Mixed 8-bit Conv / float operations. Fresh gain: +{ci['delta_sAP10_pp']:.2f} pp; paired bootstrap 95% CI [{low:.2f}, {high:.2f}].\n"
        "Model selected on development only. Dense-output parity gate remains failed. Scores are subset results, not the paper benchmark.",
        fontsize=10,
    )
    folder = DAY / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "mlsd-quality.svg", bbox_inches="tight")
    fig.savefig(TASK / "artifacts/research/mlsd-quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
