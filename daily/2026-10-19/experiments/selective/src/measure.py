"""Matched sequential benchmark and development-only candidate lock."""

import gc
import time

import numpy as np
from support import (
    RESULTS,
    checked,
    infer,
    info,
    now,
    pilot,
    read,
    save_output,
    session,
    write,
)


def development():
    if (RESULTS / "development.json").exists():
        raise FileExistsError("Development results exist")
    exports = read(RESULTS / "exports.json")
    data = pilot.load_datasets("mlsd")
    fake = np.load(checked(exports["fake_quant_reference"]))
    rows = {}
    for name, item in exports["models"].items():
        if name in exports["candidates"]:
            meta = exports["candidates"][name]["development_output"]
            output = np.load(checked(meta))
        else:
            output = infer(
                checked(item["model"]), data["evaluation"][0].numpy(), item["optimization"]
            )
            meta = save_output(name + "_development", output)
        rows[name] = {
            "output": meta,
            "sAP10": pilot.line_metrics(output, data["ground_truth"]),
            "vs_teacher": pilot.numerical(data["reference"], output),
            "vs_original_fake": pilot.numerical(fake, output),
        }
        if name in exports["candidates"]:
            rows[name]["export_parity"] = exports["candidates"][name]["export_parity"]
        print("DEVELOPMENT", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
    write(
        RESULTS / "development.json",
        {
            "finished_at_utc": now(),
            "images": 16,
            "models": rows,
            "exports": info(RESULTS / "exports.json"),
            "source_code": info(__file__),
        },
    )


def benchmark():
    if (RESULTS / "benchmark.json").exists():
        raise FileExistsError("Benchmark results exist")
    exports = read(RESULTS / "exports.json")
    data = pilot.load_datasets("mlsd")
    inputs = data["evaluation"][0].numpy()
    timings = {name: [] for name in exports["models"]}
    orders = []
    names = list(timings)
    started = now()
    for cycle in range(5):
        order = names[cycle % len(names) :] + names[: cycle % len(names)]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            item = exports["models"][name]
            runtime = session(checked(item["model"]), item["optimization"])
            for image in inputs:
                runtime.run(None, {"input": image[None]})
            for image in inputs:
                start = time.perf_counter_ns()
                runtime.run(None, {"input": image[None]})
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
            del runtime
            gc.collect()
        print("BENCHMARK ROUND", cycle + 1, "/5", flush=True)
    write(
        RESULTS / "benchmark.json",
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "protocol": "CPU ORT 4 intra/1 inter threads; sequential; spinning off; batch1; one session at a time; 5 rotated/reversed rounds; 16 warmups then 16 timed development inputs per model/round; profiling off",
            "scope": "Inference only; excludes loading, image preprocessing and line decoding; optimization level as model registry; same runtime/hardware",
            "orders": orders,
            "models": {
                name: {
                    "samples_ms": values,
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "runs": len(values),
                    "round_medians_ms": [
                        float(np.median(values[i : i + 16])) for i in range(0, len(values), 16)
                    ],
                    "model": exports["models"][name]["model"],
                    "optimization": exports["models"][name]["optimization"],
                }
                for name, values in timings.items()
            },
            "source_code": info(__file__),
        },
    )


def select():
    if (RESULTS / "regression.json").exists():
        raise RuntimeError("Cannot select after regression")
    dev = read(RESULTS / "development.json")["models"]
    bench = read(RESULTS / "benchmark.json")["models"]
    baseline_ap = dev["qdq_all"]["sAP10"]["sAP10_subset_percent"]
    baseline_ms = bench["qdq_all"]["median_ms"]
    eligible = [
        n
        for n in ["selective_a", "selective_b"]
        if dev[n]["sAP10"]["sAP10_subset_percent"] >= baseline_ap - 0.5
        and bench[n]["median_ms"] < baseline_ms
    ]
    # The fastest quality-eligible candidate is necessarily nondominated in latency.
    chosen = min(eligible, key=lambda n: bench[n]["median_ms"]) if eligible else "qdq_all"
    exports = read(RESULTS / "exports.json")
    write(
        RESULTS / "selection.json",
        {
            "locked_at_utc": now(),
            "selected": chosen,
            "regression_used": False,
            "criterion": "Fastest A/B within 0.5pp development sAP10 of QDQ, and faster than QDQ; otherwise retain baseline. Export release gate is reported separately and never relaxed",
            "eligible": eligible,
            "baseline_sAP10": baseline_ap,
            "baseline_median_ms": baseline_ms,
            "selected_model": exports["models"][chosen],
            "development": info(RESULTS / "development.json"),
            "benchmark": info(RESULTS / "benchmark.json"),
            "source_code": info(__file__),
        },
    )
    print("LOCKED", chosen, flush=True)


if __name__ == "__main__":
    development()
    benchmark()
    select()
