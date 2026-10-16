"""Lock disjoint final data without running any model or changing older manifests."""

import argparse
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
from common import ART, DAY15, RESULTS, ROOT, checked, info, read, sha, write


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def text():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    target = RESULTS / "text_manifest.json"
    if target.exists():
        checked(read(target)["inputs"])
        print("Existing text final split preserved", flush=True)
        return
    old = read(ROOT / "quantization/data/text/manifest.json")
    train = read(ROOT / "daily/2026-10-14/experiments/qat/results/data_manifest.json")
    forbidden = {
        hashlib.sha256(normalize(r["sentence"]).encode()).hexdigest()
        for group in old["records"].values()
        for r in group
    }
    forbidden.update(r["sentence_sha256"] for r in train["records"])
    source = ROOT / "quantization/.cache/sst2/data/validation-00000-of-00001.parquet"
    rows, records, seen = [], [], set()
    for row in sorted(pq.read_table(source).to_pylist(), key=lambda r: r["idx"]):
        digest = hashlib.sha256(normalize(row["sentence"]).encode()).hexdigest()
        if digest in forbidden or digest in seen:
            continue
        seen.add(digest)
        rows.append(row)
        records.append({"idx": row["idx"], "label": row["label"], "sentence_sha256": digest})
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "quantization/.cache/text_model", local_files_only=True
    )
    encoded = tokenizer(
        [r["sentence"] for r in rows],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="np",
    )
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "text_final_inputs.npz"
    np.savez(
        path,
        input_ids=encoded["input_ids"].astype(np.int64),
        attention_mask=encoded["attention_mask"].astype(np.int64),
        labels=np.array([r["label"] for r in rows], np.int64),
    )
    write(
        target,
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "GLUE/SST-2 validation, remaining normalized unique sentences",
            "source": info(source),
            "samples": len(rows),
            "records": records,
            "inputs": info(path),
            "train_calibration_development_disjoint": True,
            "sequence_length": 64,
            "policy": "No model inference or tuning before split lock; frozen right padding/truncation; report holdout separately from 100 previously monitored sentences",
            "source_code": info(__file__),
        },
    )
    print("LOCKED text", len(rows), flush=True)


def mlsd():
    import sys

    sys.path.insert(0, str(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/src"))
    from data import fetch, rgba

    target = RESULTS / "mlsd_manifest.json"
    if target.exists():
        checked(read(target)["inputs"])
        print("Existing M-LSD final split preserved", flush=True)
        return
    old = read(ROOT / "daily/2026-10-13/experiments/mlsd-ptq/results/data_manifest.json")
    previous = read(DAY15 / "results/research/holdout_manifest.json")
    evaluation = old["splits"]["evaluation"]
    folder = ROOT / ".cache/line-data/YorkUrban" / evaluation["revision"] / "test"
    metadata = folder / "metadata.jsonl"
    assert sha(metadata) == evaluation["metadata_sha256"]
    excluded = {r["image_file"] for r in evaluation["records"]} | {
        r["image_file"] for r in previous["records"]
    }
    rows = sorted(
        [
            r
            for r in [
                __import__("json").loads(line)
                for line in metadata.read_text(encoding="utf-8").splitlines()
            ]
            if r["image_file"] not in excluded
        ],
        key=lambda r: r["image_file"],
    )
    assert len(rows) == 54
    base = f"https://huggingface.co/datasets/lh9171338/YorkUrban/resolve/{evaluation['revision']}/test/"
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(lambda row: fetch(base + row["file_name"], folder / row["file_name"]), rows)
        )
    excluded_hashes = {
        r["image_sha256"] for group in old["splits"].values() for r in group["records"]
    } | {r["image_sha256"] for r in previous["records"]}
    hashes = [sha(path) for path in paths]
    assert len(set(hashes)) == len(rows) and not set(hashes) & excluded_hashes
    ART.mkdir(parents=True, exist_ok=True)
    inputs_path, gt_path = ART / "mlsd_final_inputs.npy", ART / "mlsd_final_truth.json"
    inputs = np.stack([rgba(p) for p in paths])
    np.save(inputs_path, ((inputs / 127.5 - 1).transpose(0, 3, 1, 2)).copy(), allow_pickle=False)
    write(gt_path, rows)
    write(
        target,
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "YorkUrban remaining 54 test images",
            "revision": evaluation["revision"],
            "metadata_sha256": sha(metadata),
            "samples": len(rows),
            "records": [
                {
                    "image_file": r["image_file"],
                    "image_sha256": digest,
                    "url": base + r["file_name"],
                }
                for r, digest in zip(rows, hashes)
            ],
            "excluded_images": sorted(excluded),
            "inputs": info(inputs_path),
            "ground_truth": info(gt_path),
            "train_calibration_development_previous_holdout_disjoint": True,
            "policy": "All remaining recorded test IDs, no model-based sample selection; no claim of disjointness from original pretrained data",
            "source_code": info(__file__),
        },
    )
    print("LOCKED M-LSD", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mlsd", "text"], required=True)
    args = parser.parse_args()
    (mlsd if args.task == "mlsd" else text)()
