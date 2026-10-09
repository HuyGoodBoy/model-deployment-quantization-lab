"""Offline verification of peer/Huy evidence; does not invoke any model."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
sys.path.insert(0, str(ROOT / "quantization"))
from qlab.common import classification_metrics, numerical_metrics


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def close(a, b):
    assert np.isclose(a, b, rtol=1e-8, atol=1e-10), (a, b)


def timing(samples, median, p95, count):
    assert len(samples) == count
    a = np.asarray(samples, dtype=np.float64)
    assert np.isfinite(a).all() and np.all(a > 0)
    close(median, np.median(a))
    close(p95, np.percentile(a, 95))


def main():
    folder = TASK / "results/run-04-peer-linux"
    artifacts = TASK / "artifacts/run-04-peer-linux"
    meta = read(folder / "peer_reproduction.json")
    assert meta["source_unchanged"] and meta["evaluation_passes_per_variant"] == 1
    assert len(meta["results"]) == 6 and all(r["status"] == "ok" for r in meta["results"])
    assert len({r["output_file"] for r in meta["results"]}) == 6
    for name, value in meta["source_sha256"].items():
        assert sha(ROOT / ".cache/duke-readonly/4-quantization" / name) == value
    for stage in ["q1_prepare", "q2_quant_onnx", "q3_quant_tflite"]:
        assert any(r["name"] == stage and r["returncode"] == 0 for r in meta["stages"])
    reference = np.load(artifacts / "ref_output.npy", allow_pickle=False)
    labels = np.load(artifacts / "val_y.npy", allow_pickle=False)
    assert reference.shape == (300, 1000) and labels.shape == (300,)
    close(meta["reference_accuracy_pct"], np.mean(reference.argmax(1) == labels) * 100)
    for name, value in meta["tensor_hashes"].items():
        assert sha(artifacts / name) == value
    assert len(meta["data"]["files"]) == 400
    for record in meta["data"]["files"]:
        assert sha(artifacts / record["path"]) == record["sha256"]
    expected = dict(
        line.split("==", 1)
        for line in (ROOT / "environments/peer-linux/requirements.txt").read_text().splitlines()
        if "==" in line
    )
    for package, version in expected.items():
        assert meta["packages"][package].split("+")[0] == version
    audit = read(folder / "output_and_profile_audit.json")
    for record in meta["results"]:
        path = folder / record["output_file"]
        assert sha(path) == record["output_sha256"]
        assert sha(artifacts / record["file"]) == record["model_sha256"]
        output = np.load(path, allow_pickle=False)
        assert output.shape == reference.shape and np.isfinite(output).all()
        metrics = numerical_metrics(reference, output)
        close(record["mae"], metrics["mae"])
        close(record["max_abs_error"], metrics["max_abs_error"])
        close(record["snr_db_float64"], metrics["snr_db"])
        close(
            record["snr_db_source_float32"],
            float(10 * np.log10(np.sum(reference**2) / np.sum((reference - output) ** 2))),
        )
        close(record["accuracy_pct"], np.mean(output.argmax(1) == labels) * 100)
        close(record["agreement_pct"], np.mean(output.argmax(1) == reference.argmax(1)) * 100)
        tied = np.sum(output == output.max(axis=1, keepdims=True), axis=1) > 1
        correct = output.argmax(1) == labels
        assert (
            record["top1_tie_samples"] == int(tied.sum()) == audit["ties"][record["name"]]["total"]
        )
        assert int(np.sum(tied & correct)) == audit["ties"][record["name"]]["correct_argmax"]
        assert int(np.sum(tied & ~correct)) == audit["ties"][record["name"]]["wrong_argmax"]
        timing(record["samples_ms"], record["median_ms"], record["p95_ms"], 100)
    for profile in audit["profiles"].values():
        assert profile["runs"] == 5 and profile["threads"] == 4
        close(sum(x["fraction_of_node_event_time"] for x in profile["ops"].values()), 1)
    linux = TASK / "results/run-04-huy-linux"
    files = sorted(linux.glob("*_new_t4_matched.json"))
    assert len(files) == 12
    for p in files:
        r = read(p)
        win = read(TASK / "results/run-01" / p.name)
        assert r["model_sha256"] == win["model_sha256"] and r["input_sha256"] == win["input_sha256"]
        output_file = linux / f"output_{r['name']}.npy"
        assert sha(output_file) == r["output_sha256"]
        output = np.load(output_file, allow_pickle=False)
        with np.load(
            ROOT / f"quantization/results/{r['task']}/outputs.npz", allow_pickle=False
        ) as a:
            ref = a["tensorflow_fp32"].copy()
            label = a["labels"].copy()
        for k, value in numerical_metrics(ref, output).items():
            close(r["metrics"][k], value)
        for k, value in classification_metrics(output, label, r["task"]).items():
            close(r["metrics"][k], value)
        b = r["benchmark"]
        assert b["threads"] == 4 and b["warmup"] == 30
        timing(b["samples_ms"], b["median_ms"], b["p95_ms"], 200)
    assert all(r["returncode"] == 0 for r in read(linux / "execution.json"))
    result = {
        "status": "passed",
        "peer_variants": 6,
        "huy_linux_variants": 12,
        "source_images_verified": 400,
        "latency_samples_verified": 3000,
        "distinct_peer_output_files": 6,
        "scope": "offline hashes, output metrics/classification/ties, sample statistics and model/input identity; no new inference",
    }
    (folder / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
