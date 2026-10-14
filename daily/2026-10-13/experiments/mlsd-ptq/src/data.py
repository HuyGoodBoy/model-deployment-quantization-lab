"""Pinned, small Wireframe-train / YorkUrban-test subsets for low-bit experiments."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
CACHE = ROOT / ".cache/line-data"
SEED = 20261013


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(url, timeout=45) as source,
            path.with_suffix(path.suffix + ".partial").open("wb") as target,
        ):
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        path.with_suffix(path.suffix + ".partial").replace(path)
    return path


def rgba(path):
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), np.float32
        )
    return np.concatenate([rgb, np.ones((512, 512, 1), np.float32)], axis=-1)


def prepare(calibration_count=32, evaluation_count=16):
    folder = TASK / "data"
    folder.mkdir(parents=True, exist_ok=True)
    locked = TASK / "results/data_manifest.json"
    previous = json.loads(locked.read_text(encoding="utf-8")) if locked.exists() else None
    arrays, splits = {}, {}
    for split, dataset, source_split, count in [
        ("calibration", "Wireframe", "train", calibration_count),
        ("evaluation", "YorkUrban", "test", evaluation_count),
    ]:
        repo = "lh9171338/" + dataset
        if previous:
            revision = previous["splits"][split]["revision"]
        else:
            with urllib.request.urlopen(
                "https://huggingface.co/api/datasets/" + repo, timeout=30
            ) as response:
                revision = json.load(response)["sha"]
        base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{source_split}/"
        metadata = fetch(
            base + "metadata.jsonl", CACHE / dataset / revision / source_split / "metadata.jsonl"
        )
        rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda r: r["image_file"])
        chosen = np.random.default_rng(SEED).choice(len(rows), count, replace=False)
        rows = [rows[int(i)] for i in chosen]

        def download(row, base=base, dataset=dataset, revision=revision, source_split=source_split):
            return fetch(
                base + row["file_name"],
                CACHE / dataset / revision / source_split / row["file_name"],
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(download, rows))
        records = []
        for row, path in zip(rows, paths):
            records.append(
                {
                    "image_file": row["image_file"],
                    "image_sha256": sha(path),
                    "image_size": row["image_size"],
                    "line_count": len(row["lines"]),
                    "url": base + row["file_name"],
                }
            )
        splits[split] = {
            "dataset": repo,
            "source_split": source_split,
            "revision": revision,
            "metadata_sha256": sha(metadata),
            "records": records,
        }
        arrays[split] = np.stack([rgba(path) for path in paths])
        if split == "evaluation":
            write_json(folder / "ground_truth.json", rows)
        print(f"{split}: {dataset}/{source_split}, {len(rows)} images", flush=True)
    train_hashes = {r["image_sha256"] for r in splits["calibration"]["records"]}
    test_hashes = {r["image_sha256"] for r in splits["evaluation"]["records"]}
    if train_hashes & test_hashes:
        raise RuntimeError("Image leakage across splits")
    np.savez_compressed(folder / "inputs.npz", **arrays)
    manifest = {
        "seed": SEED,
        "preprocess": "Pillow RGB bilinear 512x512 + alpha=1; raw [0,255] RGBA",
        "splits": splits,
        "inputs_sha256": sha(folder / "inputs.npz"),
        "ground_truth_sha256": sha(folder / "ground_truth.json"),
        "note": "YorkUrban test subset is evaluation only; never calibration/training. Mirrors retain original dataset provenance.",
    }
    if previous:
        for split, value in splits.items():
            if value != previous["splits"][split]:
                raise RuntimeError("Pinned source/data changed")
    write_json(locked, manifest)
    return arrays


def load():
    manifest = json.loads((TASK / "results/data_manifest.json").read_text(encoding="utf-8"))
    if sha(TASK / "data/inputs.npz") != manifest["inputs_sha256"]:
        raise RuntimeError("Input hash mismatch")
    if sha(TASK / "data/ground_truth.json") != manifest["ground_truth_sha256"]:
        raise RuntimeError("GT hash mismatch")
    return (
        np.load(TASK / "data/inputs.npz", allow_pickle=False),
        json.loads((TASK / "data/ground_truth.json").read_text(encoding="utf-8")),
        manifest,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--evaluation-count", type=int, default=16)
    args = parser.parse_args()
    prepare(args.calibration_count, args.evaluation_count)
