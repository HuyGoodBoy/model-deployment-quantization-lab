"""Fresh-process init/first invoke/RSS; filesystem page cache is not flushed."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def rss():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, unit = line.split()
            assert unit == "kB"
            values[key.rstrip(":")] = int(value) * 1024
    return values


def worker(variant):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    root = Path(__file__).resolve().parents[5]
    day20 = root / "daily/2026-10-20/experiments/zipformer"
    day21 = root / "daily/2026-10-21/experiments/zipformer"
    models = {
        "baseline": day20 / "artifacts/conversion/static/encoder/output/input_float32.tflite",
        "fc": day21 / "artifacts/encoder_fc.tflite",
        "fc_bias": day21 / "artifacts/refinement/encoder_fc_bias.tflite",
    }
    reference = json.loads((day20 / "results/onnx_baseline.json").read_text())
    fixture = root / reference["smoke_audio"][0]["captures"][0]["path"]
    with np.load(fixture) as saved:
        feeds = {k: v.copy() for k, v in saved.items()}
    memory = {"before_interpreter": rss()}
    start = time.perf_counter_ns()
    runtime = Interpreter(model_path=str(models[variant]), num_threads=4)
    constructed = time.perf_counter_ns()
    memory["after_constructor"] = rss()
    allocation_start = time.perf_counter_ns()
    runtime.allocate_tensors()
    allocated = time.perf_counter_ns()
    memory["after_allocation"] = rss()
    metadata_start = time.perf_counter_ns()
    sig = runtime.get_signature_runner(next(iter(runtime.get_signature_list())))
    inputs = sig.get_input_details()
    outputs = sig.get_output_details()
    assert set(feeds) == set(inputs)
    prepared = time.perf_counter_ns()
    typed_feeds = {k: feeds[k].astype(d["dtype"], copy=False) for k, d in inputs.items()}
    first_start = time.perf_counter_ns()
    result = sig(**typed_feeds)
    first_end = time.perf_counter_ns()
    assert set(result) == set(outputs)
    assert all(np.isfinite(v).all() for v in result.values())
    memory["after_first_invoke"] = rss()
    for _ in range(8):
        result = sig(**typed_feeds)
    memory["after8_warm_invokes"] = rss()
    timings = {
        "constructor_ms": (constructed - start) / 1e6,
        "allocation_ms": (allocated - allocation_start) / 1e6,
        "signature_metadata_ms": (prepared - metadata_start) / 1e6,
        "first_signature_invoke_ms": (first_end - first_start) / 1e6,
    }
    timings["init_ms"] = (
        timings["constructor_ms"] + timings["allocation_ms"] + timings["signature_metadata_ms"]
    )
    timings["init_plus_first_ms"] = timings["init_ms"] + timings["first_signature_invoke_ms"]
    print(
        json.dumps(
            {
                "variant": variant,
                "pid": os.getpid(),
                "timings": timings,
                "rss_bytes": memory,
                "peak_RSS_bytes": memory["after8_warm_invokes"]["VmHWM"],
            }
        ),
        flush=True,
    )


def main():
    import numpy as np
    from asr21 import RESULTS, info, now, read, write

    target = RESULTS / "refinement_coldstart_memory.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "refinement_bias_evaluation.json")["quality_passed"]
    started, rows, orders = now(), {v: [] for v in ["baseline", "fc", "fc_bias"]}, []
    for cycle in range(5):
        names = list(rows)
        order = names[cycle % 3 :] + names[: cycle % 3]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for variant in order:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant]
            run = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
            value = json.loads(run.stdout.strip().splitlines()[-1])
            rows[variant].append(value)
        print("COLDSTART", cycle + 1, "/5", flush=True)
    models = {}
    for name, samples in rows.items():
        keys = list(samples[0]["timings"])
        models[name] = {
            "samples": samples,
            "timing_medians_ms": {
                k: float(np.median([r["timings"][k] for r in samples])) for k in keys
            },
            "median_peak_RSS_MiB": float(
                np.median([r["peak_RSS_bytes"] for r in samples]) / 1024**2
            ),
            "max_peak_RSS_MiB": max(r["peak_RSS_bytes"] for r in samples) / 1024**2,
        }
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": models,
            "orders": orders,
            "protocol": "Linux LiteRT2.1.2 CPU4threads encoder only;5new child processes per variant,one model per process. RSS includes Python/NumPy/runtime/model/state/output,not model-only RAM;read /proc/self/status VmRSS/VmHWM. Timing excludes imports,feed loading,process launch and RSS reads;includes model constructor,allocation,signature metadata,and first signature call separately. Filesystem page cache not flushed;fresh-process start,not guaranteed cold-disk startup. Eight warm invokes after first for observed peak;not long-session/Android memory.",
            "source_code": info(__file__),
        },
    )
    print(
        "COLDSTART RESULT",
        {
            n: {"timings": r["timing_medians_ms"], "peak_MiB": r["median_peak_RSS_MiB"]}
            for n, r in models.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["baseline", "fc", "fc_bias"])
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
