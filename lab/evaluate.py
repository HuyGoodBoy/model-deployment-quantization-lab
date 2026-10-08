"""Chay cung input, luu raw output, so sanh voi TensorFlow goc."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from .common import (ARTIFACTS, DATA, RESULTS, ROOT, RUNTIMES, create_runner,
                     numerical_metrics, read_json, validate_input, write_json)


def read_labels(path: Path, filenames: list[str]) -> dict:
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "class_index"}.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv can hai cot filename,class_index")
        for row in reader:
            label = int(row["class_index"])
            if not 0 <= label < 1000 or row["filename"] in labels:
                raise ValueError("Label ngoai [0,999] hoac filename trung lap")
            labels[row["filename"]] = label
    missing = set(filenames) - labels.keys()
    if missing:
        raise ValueError(f"Thieu label cho: {sorted(missing)}")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0 or args.threads < 1:
        parser.error("tolerance phai >= 0 va threads >= 1")
    conversion = read_json(ARTIFACTS / "conversion_info.json")
    current_hash = hashlib.sha256((ARTIFACTS / "mobilenetv2.keras").read_bytes()).hexdigest()
    if conversion["original_sha256"] != current_hash:
        raise ValueError("Model goc da thay doi. Hay chay convert lai.")
    for name, info in conversion["files"].items():
        if hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Artifact da thay doi: {name}. Hay convert lai.")
    manifest = read_json(DATA / "manifest.json")
    records = manifest["records"]
    for record in records:
        if hashlib.sha256((ROOT / record["input_path"]).read_bytes()).hexdigest() != record["input_sha256"]:
            raise ValueError("Tensor input da thay doi. Hay chay prepare lai.")
    labels = read_labels(args.labels, [r["filename"] for r in records]) if args.labels else None
    outputs, runtime_info = {}, {}
    for runtime in RUNTIMES:
        run, info = create_runner(runtime, args.threads)
        collected = []
        for record in records:
            x = np.load(ROOT / record["input_path"], allow_pickle=False)
            validate_input(x)
            y = run(x)
            if y.shape != (1, 1000) or y.dtype != np.float32 or not np.isfinite(y).all():
                raise ValueError(f"Output {runtime} sai shape/dtype/finite: {y.shape}, {y.dtype}")
            if np.any(y < -1e-6) or not np.allclose(y.sum(axis=1), 1, atol=1e-5):
                raise ValueError(f"Output {runtime} khong phai softmax probabilities")
            collected.append(y[0])
        outputs[runtime] = np.stack(collected)
        runtime_info[runtime] = info
        print(f"Evaluated {runtime}: {outputs[runtime].shape}", flush=True)
        # Giai phong model/session truoc khi tao runtime tiep theo.
        del run
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "outputs.npz", **outputs)
    reference = outputs["tensorflow"]
    comparisons, per_image = {}, []
    top5 = {r: np.argsort(-y, axis=1, kind="stable")[:, :5] for r, y in outputs.items()}
    for runtime in ("onnx", "tflite"):
        candidate = outputs[runtime]
        comparisons[runtime] = numerical_metrics(reference, candidate)
        comparisons[runtime].update({
            "top1_agreement": float(np.mean(reference.argmax(1) == candidate.argmax(1))),
            "top5_overlap": float(np.mean([len(set(a) & set(b)) / 5
                                           for a, b in zip(top5["tensorflow"], top5[runtime])])),
            "allclose": bool(np.allclose(candidate, reference, atol=args.atol, rtol=args.rtol)),
        })
        for i, record in enumerate(records):
            metrics = numerical_metrics(reference[i], candidate[i])
            metrics.update({"filename": record["filename"], "runtime": runtime,
                            "reference_top1": int(reference[i].argmax()),
                            "candidate_top1": int(candidate[i].argmax()),
                            "reference_top5": top5["tensorflow"][i].tolist(),
                            "candidate_top5": top5[runtime][i].tolist(),
                            "top1_agreement": bool(reference[i].argmax() == candidate[i].argmax()),
                            "allclose": bool(np.allclose(candidate[i], reference[i],
                                                        atol=args.atol, rtol=args.rtol))})
            per_image.append(metrics)
    accuracy = None
    if labels is not None:
        truth = np.asarray([labels[r["filename"]] for r in records])
        accuracy = {r: {"top1_accuracy": float(np.mean(y.argmax(1) == truth)),
                        "top5_accuracy": float(np.mean(np.any(top5[r] == truth[:, None], axis=1)))}
                    for r, y in outputs.items()}
    write_json(RESULTS / "evaluation.json", {
        "num_images": len(records), "dataset_kind": manifest["dataset_kind"],
        "model_sha256": current_hash,
        "inputs": {r["filename"]: r["input_sha256"] for r in records},
        "atol": args.atol, "rtol": args.rtol, "threads": args.threads,
        "comparisons": comparisons, "per_image": per_image, "accuracy": accuracy,
        "labels_path": str(args.labels.resolve()) if args.labels else None,
        "runtime_info": runtime_info,
    })
    print("Saved raw outputs and evaluation metrics.", flush=True)


if __name__ == "__main__":
    main()
