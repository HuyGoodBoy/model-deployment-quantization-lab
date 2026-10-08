"""Benchmark mot runtime trong process rieng; batch=1, CPU."""
import argparse
import hashlib
import platform
import time

import numpy as np

from .common import ARTIFACTS, DATA, RESULTS, ROOT, RUNTIMES, create_runner, read_json, validate_input, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=RUNTIMES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1 or args.threads < 1:
        parser.error("warmup, runs, threads phai >= 1")
    record = read_json(DATA / "manifest.json")["records"][0]
    input_hash = hashlib.sha256((ROOT / record["input_path"]).read_bytes()).hexdigest()
    if input_hash != record["input_sha256"]:
        raise ValueError("Tensor input da thay doi. Hay chay prepare lai.")
    x = np.load(ROOT / record["input_path"], allow_pickle=False)
    validate_input(x)
    run, runtime_info = create_runner(args.runtime, args.threads)
    for _ in range(args.warmup):
        run(x)
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter_ns()
        y = run(x)
        end = time.perf_counter_ns()
        samples.append((end - start) / 1e6)
    if y.shape != (1, 1000) or not np.isfinite(y).all():
        raise ValueError("Output benchmark khong hop le")
    values = np.asarray(samples)
    stats = {
        "runtime": args.runtime, "threads": args.threads, "warmup": args.warmup,
        "runs": args.runs, "batch_size": 1, "input_image": record["filename"],
        "input_sha256": input_hash,
        "model_sha256": hashlib.sha256((ARTIFACTS / "mobilenetv2.keras").read_bytes()).hexdigest(),
        "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "std_ms": float(values.std()),
        "min_ms": float(values.min()), "max_ms": float(values.max()),
        "images_per_second": float(1000 / values.mean()), "latency_samples_ms": samples,
        "timing_scope": "inference API with in-memory input/output transfers and Python overhead; no load/preprocess",
        "system": {"platform": platform.platform(), "processor": platform.processor(),
                   "architecture": platform.machine(), "python": platform.python_version()},
        "runtime_info": runtime_info,
    }
    write_json(RESULTS / f"benchmark_{args.runtime}.json", stats)
    print(f"{args.runtime}: median={stats['median_ms']:.3f} ms, p95={stats['p95_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
