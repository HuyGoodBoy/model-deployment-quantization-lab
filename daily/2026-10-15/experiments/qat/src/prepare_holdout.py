"""Lock 32 YorkUrban images disjoint from the monitored 16-image development subset."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from research_mlsd import ART, RESULTS
from train_continue import ROOT, pilot, relative

import data as line_source


def main():
    manifest_path = RESULTS / "holdout_manifest.json"
    if manifest_path.exists():
        locked = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert pilot.sha(ROOT / locked["inputs"]["path"]) == locked["inputs"]["sha256"]
        assert pilot.sha(ROOT / locked["ground_truth"]["path"]) == locked["ground_truth"]["sha256"]
        print("Existing holdout preserved", flush=True)
        return
    original = json.loads((pilot.DAY13 / "results/data_manifest.json").read_text(encoding="utf-8"))
    split = original["splits"]["evaluation"]
    revision = split["revision"]
    folder = ROOT / ".cache/line-data/YorkUrban" / revision / "test"
    metadata = folder / "metadata.jsonl"
    assert pilot.sha(metadata) == split["metadata_sha256"]
    development_ids = {r["image_file"] for r in split["records"]}
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    eligible = sorted(
        [r for r in rows if r["image_file"] not in development_ids], key=lambda r: r["image_file"]
    )
    order = np.random.default_rng(20261015).choice(len(eligible), 32, replace=False)
    chosen = [eligible[int(i)] for i in order]
    base = f"https://huggingface.co/datasets/lh9171338/YorkUrban/resolve/{revision}/test/"

    def fetch(row):
        return line_source.fetch(base + row["file_name"], folder / row["file_name"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, chosen))
    ART.mkdir(parents=True, exist_ok=True)
    inputs = np.stack([line_source.rgba(path) for path in paths])
    inputs = ((inputs / 127.5 - 1).transpose(0, 3, 1, 2)).copy()
    inputs_path = ART / "holdout_inputs.npy"
    gt_path = ART / "holdout_ground_truth.json"
    np.save(inputs_path, inputs, allow_pickle=False)
    pilot.write_json(gt_path, chosen)
    forbidden_hashes = {
        record["image_sha256"]
        for value in original["splits"].values()
        for record in value["records"]
    }
    records = [
        {
            "image_file": row["image_file"],
            "image_sha256": pilot.sha(path),
            "url": base + row["file_name"],
        }
        for row, path in zip(chosen, paths)
    ]
    assert not ({r["image_sha256"] for r in records} & forbidden_hashes)
    pilot.write_json(
        manifest_path,
        {
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "seed": 20261015,
            "dataset": "lh9171338/YorkUrban",
            "revision": revision,
            "metadata_sha256": pilot.sha(metadata),
            "images": 32,
            "excluded_development_images": sorted(development_ids),
            "inputs": {"path": relative(inputs_path), "sha256": pilot.sha(inputs_path)},
            "ground_truth": {"path": relative(gt_path), "sha256": pilot.sha(gt_path)},
            "records": records,
            "policy": "Lock before candidate selection; no holdout inference until development-only model selection is recorded; never used for calibration/gradient/tuning",
            "scope": "A fresh subset for this experiment from the same YorkUrban mirror, not proof that original pretrained weights never saw these scenes",
            "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        },
    )
    print("Locked 32 fresh YorkUrban images, disjoint by ID and file hash", flush=True)


if __name__ == "__main__":
    main()
