"""Compare original/refined integer models and FP32 under one identical timing protocol."""

import gc
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from export_integer import make_session
from research_mlsd import RESULTS, read
from train_continue import ROOT, TASK, pilot, relative


def main():
    old = read(TASK / "results/mlsd_integer_export.json")
    new = read(RESULTS / "deployment.json")
    paths = {
        "original_fp32": ROOT / new["artifacts"]["original_fp32_model"]["path"],
        "original_integer": ROOT / old["artifacts"]["integer_model"]["path"],
        "refined_integer": ROOT / new["artifacts"]["integer_model"]["path"],
    }
    inputs = pilot.load_datasets("mlsd")["evaluation"][0].numpy()
    # Only one ORT pool exists at a time: idle pools must not spin during another model's run.
    samples = {name: [] for name in paths}
    names = list(paths)
    for cycle in range(5):
        order = list(np.roll(names, -(cycle % len(names))))
        if cycle % 2:
            order.reverse()
        for name in order:
            session = make_session(paths[name])
            for image in inputs:
                session.run(None, {"input": image[None]})
            for image in inputs:
                started = time.perf_counter()
                session.run(None, {"input": image[None]})
                samples[name].append((time.perf_counter() - started) * 1000)
            del session
            gc.collect()
        print("isolated timing round", cycle + 1, flush=True)
    timing = {
        name: {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values,
            "runs": len(values),
        }
        for name, values in samples.items()
    }
    result = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "One ORT CPU session/thread pool at a time, 4 intra/1 inter threads, sequential ALL optimizations, default spinning; five rounds, rotate/reverse model order; new session each model/round, 16 warmups then 16 timed same development images; profiler disabled",
        "scope": "Inference only; same run/environment; no preprocessing, session loading, decode, Android, training or quality selection",
        "model_sha256": {name: pilot.sha(path) for name, path in paths.items()},
        "results": timing,
        "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        "note": "Only this isolated protocol is used in the final report. Shared-pool and disabled-spinning exploratory runs are preserved separately; their absolute times are not comparable.",
    }
    pilot.write_json(RESULTS / "matched_benchmark.json", result)
    print({name: value["median_ms"] for name, value in timing.items()}, flush=True)


if __name__ == "__main__":
    main()
