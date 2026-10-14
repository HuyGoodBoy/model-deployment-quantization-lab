"""Prepare a bounded QAT smoke run; reserve evaluation data for evaluation only."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from pyarrow import parquet

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface-qat"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def normalize(sentence):
    return " ".join(sentence.lower().split())


def main():
    ART.mkdir(parents=True, exist_ok=True)
    h5 = ROOT / "quantization/.cache/text_model/tf_model.h5"
    arrays = {}

    def weight(name, value):
        if not isinstance(value, h5py.Dataset):
            return
        key = name.split("tf_distil_bert_for_sequence_classification/")[1]
        key = key.replace("layer_._", "layer.").replace("/", ".")
        key = key.replace("kernel:0", "weight").replace("bias:0", "bias")
        key = key.replace("gamma:0", "weight").replace("beta:0", "bias")
        key = key.replace("embeddings:0", "weight").replace("weight:0", "weight")
        v = value[()]
        if name.endswith("kernel:0"):
            v = v.T
        arrays[key] = v

    with h5py.File(h5) as handle:
        handle.visititems(weight)
    np.savez(ART / "distilbert_weights.npz", **arrays)
    cache = ROOT / "quantization/.cache"
    previous = json.loads(
        (ROOT / "quantization/data/text/manifest.json").read_text(encoding="utf-8")
    )
    forbidden = {normalize(r["sentence"]) for group in previous["records"].values() for r in group}
    rows = parquet.read_table(cache / "sst2/data/train-00000-of-00001.parquet").to_pylist()
    eligible = []
    seen = set(forbidden)
    for row in sorted(rows, key=lambda r: r["idx"]):
        value = normalize(row["sentence"])
        if value not in seen:
            eligible.append(row)
            seen.add(value)
    rng = np.random.default_rng(20261014)
    chosen = []
    for label in [0, 1]:
        pool = [r for r in eligible if r["label"] == label]
        chosen.extend(pool[int(i)] for i in rng.choice(len(pool), 128, replace=False))
    rng.shuffle(chosen)
    tokenizer = AutoTokenizer.from_pretrained(cache / "text_model", local_files_only=True)
    tokens = tokenizer(
        [r["sentence"] for r in chosen],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="np",
    )
    np.savez(
        ART / "text_train.npz",
        input_ids=tokens["input_ids"].astype(np.int64),
        attention_mask=tokens["attention_mask"].astype(np.int64),
        labels=np.array([r["label"] for r in chosen], np.int64),
    )
    record = {
        "status": "data prepared for QAT pilot",
        "seed": 20261014,
        "dataset": "GLUE/SST-2",
        "train_examples": 256,
        "labels_per_class": 128,
        "sequence_length": 64,
        "records": [
            {
                "idx": r["idx"],
                "label": r["label"],
                "sentence_sha256": hashlib.sha256(normalize(r["sentence"]).encode()).hexdigest(),
            }
            for r in chosen
        ],
        "train_file_sha256": sha(ART / "text_train.npz"),
        "source_checkpoint_sha256": sha(h5),
        "transposed_weights_sha256": sha(ART / "distilbert_weights.npz"),
        "weight_tensor_count": len(arrays),
        "evaluation": "unchanged 100 SST-2 validation examples from day08; not used for training or observer calibration",
        "calibration": "existing 100 train examples from day08; disjoint from the new 256 train examples",
        "train_eval_or_calibration_text_overlap": False,
        "mlsd": "32 Wireframe train images from day13; teacher pseudo-labels. YorkUrban test images never used to train.",
    }
    write_json(TASK / "results/data_manifest.json", record)
    print("Prepared 104 checkpoint tensors and 256 disjoint SST-2 train examples", flush=True)


if __name__ == "__main__":
    main()
