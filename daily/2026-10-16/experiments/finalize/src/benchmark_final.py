"""Matched inference-only benchmark: one ORT pool at a time, 80 samples/model."""

import argparse
import gc
import time
from datetime import UTC, datetime

import numpy as np
from common import DAY15, RESULTS, checked, info, read, session, write
from train_continue import pilot


def main(task):
    target = RESULTS / f"{task}_benchmark.json"
    if target.exists():
        raise FileExistsError("Preserve existing benchmark")
    data = pilot.load_datasets("mlsd" if task == "mlsd" else "text")
    if task == "mlsd":
        base = read(DAY15 / "results/research/deployment.json")
        rounding = read(DAY15 / "results/export-parity/rounding.json")
        selected = read(RESULTS / "mlsd_selection.json")
        paths = {
            "original_fp32": checked(base["artifacts"]["original_fp32_model"]),
            "calibrated_41_integer": checked(base["artifacts"]["integer_model"]),
            "rounding_41_integer": checked(rounding["artifacts"]["model"]),
            "selected_qdq_22_integer_19_float": checked(
                selected["candidates"][selected["selected"]]["model"]
            ),
        }
        feeds = [{"input": image[None]} for image in data["evaluation"][0].numpy()]
        indices = list(range(16))
    else:
        base = read(RESULTS / "text_export.json")
        selected = read(RESULTS / "text_selection.json")
        paths = {
            "teacher_fp32": checked(base["models"]["teacher_fp32"]["model"]),
            "qat_weights_fp32": checked(base["models"]["qat_weights_fp32"]["model"]),
            "qat_static_38_integer": checked(selected["candidates"][selected["selected"]]["model"]),
        }
        indices = np.linspace(0, len(data["evaluation"][0]) - 1, 16, dtype=int).tolist()
        feeds = [
            {
                "input_ids": data["evaluation"][0][i : i + 1].numpy(),
                "attention_mask": data["evaluation"][1][i : i + 1].numpy(),
            }
            for i in indices
        ]
    timings = {name: [] for name in paths}
    names = list(paths)
    for cycle in range(5):
        order = list(np.roll(names, -cycle % len(names)))
        if cycle % 2:
            order.reverse()
        for name in order:
            runtime = session(paths[name])
            for feed in feeds:
                runtime.run(None, feed)
            for feed in feeds:
                start = time.perf_counter()
                runtime.run(None, feed)
                timings[name].append((time.perf_counter() - start) * 1000)
            del runtime
            gc.collect()
        print("BENCHMARK", task, "round", cycle + 1, "/5", flush=True)
    values = {
        name: {
            "median_ms": float(np.median(v)),
            "p95_ms": float(np.percentile(v, 95)),
            "samples_ms": v,
            "runs": len(v),
            "model": info(paths[name]),
        }
        for name, v in timings.items()
    }
    write(
        target,
        {
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "task": task,
            "protocol": "One ORT CPU session at a time; 4 intra/1 inter threads, sequential ALL optimizations, spinning disabled; 5 rounds rotate/reverse model order; fresh session each model/round; 16 warmups +16 timed development examples/round; batch1; no profiler/training/evaluation running concurrently",
            "input_indices": indices,
            "scope": "Inference only; excludes loading, tokenization/image preprocessing, decoder and Android; M-LSD QDQ has different precision coverage from 41-ConvInteger variants",
            "results": values,
            "source_code": info(__file__),
        },
    )
    print({name: v["median_ms"] for name, v in values.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mlsd", "text"], required=True)
    main(parser.parse_args().task)
