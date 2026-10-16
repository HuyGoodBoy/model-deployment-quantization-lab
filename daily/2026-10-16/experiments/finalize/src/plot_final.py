"""Standalone final quality figures and predetermined scene overlays."""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, ROOT, checked, read, write
from PIL import Image

sys.path.append(str(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/src"))
from metrics import line_metrics, lines


def main():
    directory = ROOT / "daily/2026-10-16/figures"
    directory.mkdir(parents=True, exist_ok=True)
    ml, tx = read(RESULTS / "mlsd_final.json"), read(RESULTS / "text_final.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    ml_names = [
        "original_fp32",
        "ptq_split_full_int8",
        "qat_original_integer",
        "qat_calibrated_integer",
        "qat_selected_qdq",
    ]
    ml_labels = [
        "FP32",
        "PTQ full INT8",
        "QAT old integer",
        "QAT + calibration",
        "QDQ: 22 int / 19 float",
    ]
    for axis, names, labels, records, metric in [
        (axes[0], ml_names, ml_labels, ml["models"], "mlsd"),
        (
            axes[1],
            ["teacher_fp32", "qat_weights_fp32", "qat_fake_quant", "qat_static_integer"],
            ["FP32 teacher", "QAT weights FP32", "QAT fake quant", "QAT: 38 integer Linear"],
            tx["models"],
            "text",
        ),
    ]:
        values = [
            records[n]["metrics"]["line_quality"]["sAP10_subset_percent"]
            if metric == "mlsd"
            else records[n]["accuracy_percent"]
            for n in names
        ]
        cis = [
            records[n]["image_bootstrap_sAP10_95_percentile_ci"]
            if metric == "mlsd"
            else records[n]["wilson_95_ci_percent"]
            for n in names
        ]
        axis.scatter(
            values,
            range(len(names)),
            s=55,
            color=["#51677a"] + ["#348678"] * (len(names) - 1),
            zorder=3,
        )
        axis.set_yticks(range(len(names)), labels)
        for i, (value, ci) in enumerate(zip(values, cis)):
            axis.plot(ci, [i, i], color="#222222", linewidth=1.5)
            axis.text(
                value + (0.3 if metric == "mlsd" else 0.2), i + 0.25, f"{value:.2f}%", fontsize=9
            )
        axis.set_ylim(len(names) - 0.4, -0.55)
        axis.set_xlabel("sAP10 (%)" if metric == "mlsd" else "Accuracy (%)")
        axis.set_xlim(0 if metric == "mlsd" else 80, max(c[1] for c in cis) + 3)
        axis.set_title(
            "M-LSD: 54 new recorded images"
            if metric == "mlsd"
            else "DistilBERT: 772 new recorded sentences"
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.015,
        "95% intervals: image bootstrap / Wilson. Mixed precision scopes differ. No claim of pretraining disjointness.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(directory / "final-quality.svg")
    fig.savefig(directory / "final-quality.png", dpi=150)
    plt.close(fig)
    manifest = read(RESULTS / "mlsd_manifest.json")
    truth = read(checked(manifest["ground_truth"]))
    outputs = {
        name: np.load(checked(ml["models"][name]["output"]))
        for name in ["original_fp32", "qat_selected_qdq"]
    }
    indices = [
        0,
        17,
        35,
        53,
    ]  # Fixed spread by sorted image ID, chosen without per-image metric inspection.
    visual_records = []
    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    for row_index, index in enumerate(indices):
        row = truth[index]
        path = (
            ROOT / ".cache/line-data/YorkUrban" / manifest["revision"] / "test" / row["file_name"]
        )
        image = Image.open(path).convert("RGB")
        width, height = image.size
        visual_records.append(
            {
                "index": index,
                "image_file": row["image_file"],
                "models": {
                    name: line_metrics(output[index : index + 1], [row])
                    for name, output in outputs.items()
                },
            }
        )
        for column, name in enumerate(["ground_truth", "original_fp32", "qat_selected_qdq"]):
            axis = axes[row_index, column]
            axis.imshow(image)
            segments = (
                np.asarray(row["lines"]).reshape(-1, 2, 2)
                if column == 0
                else np.asarray(
                    [line for line, score in lines(outputs[name][index]) if score >= 0.1]
                ).reshape(-1, 2, 2)
                * [width / 128, height / 128]
            )
            for segment in segments:
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    linewidth=0.8,
                    color="#28e5d4" if column else "#ffcc33",
                )
            axis.axis("off")
            axis.set_xlim(0, width)
            axis.set_ylim(height, 0)
            axis.set_title(
                ["Ground truth", "FP32", "QDQ: 22 integer / 19 float"][column]
                + f" | {row['image_file']}",
                fontsize=9,
            )
    fig.suptitle("Fixed images by ID spread; prediction overlay uses score >= 0.1", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(directory / "line-comparison.png", dpi=140)
    plt.close(fig)
    write(
        RESULTS / "visual_examples.json",
        {
            "selection_rule": "Fixed sorted-ID indices [0,17,35,53], not chosen by model quality; metrics reported after final evaluation",
            "overlay_minimum_score": 0.1,
            "ap_decoder_minimum_score": 0.01,
            "examples": visual_records,
        },
    )
    print("Saved final quality chart and four fixed-scene overlays", flush=True)


if __name__ == "__main__":
    main()
