"""Additional speed path: thread count only, identical FP32 model and decoder."""

import gc
import time

import numpy as np
from evaluate_litert import create, infer, save
from mlsd_support import RESULTS, ROOT, info, now, pilot, read, write


def main():
    target = RESULTS / "float_thread_tuning.json"
    if target.exists():
        raise FileExistsError(target)
    record = read(RESULTS / "aeq_exports.json")["models"]["float"]["model"]
    data = pilot.load_datasets("mlsd")
    inputs = data["evaluation"][0].numpy()
    counts = [1, 2, 4, 8]
    times = {n: [] for n in counts}
    started = now()
    orders = []
    for cycle in range(5):
        order = counts[cycle % 4 :] + counts[: cycle % 4]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for count in order:
            runtime, function = create(record, num_threads=count)
            for x in inputs:
                function(input=x[None])
            for x in inputs:
                start = time.perf_counter_ns()
                function(input=x[None])
                times[count].append((time.perf_counter_ns() - start) / 1e6)
            del runtime, function
            gc.collect()
        print("THREAD ROUND", cycle + 1, "/5", flush=True)
    selected = min(counts, key=lambda n: np.median(times[n]))
    _runtime, function = create(record, num_threads=selected)
    output = infer(function, inputs)
    development = read(RESULTS / "aeq_development.json")["models"]["float"]
    baseline = np.load(ROOT / development["output"]["path"])
    parity = pilot.numerical(baseline, output)
    quality = pilot.line_metrics(output, data["ground_truth"])
    assert parity["relative_l2"] <= 0.02
    assert (
        abs(quality["sAP10_subset_percent"] - development["sAP10"]["sAP10_subset_percent"]) <= 0.5
    )
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "model": record,
            "protocol": "Linux LiteRT2.2 CPU/XNNPACK FP32 same artifact, only num_threads changes;5rotated/reversed rounds16warmups+16timings;one interpreter at a time;shared float application I/O;loading/preprocess/line decoder excluded;no competing experiment. Select fastest median then verify development output/AP; no regression selection.",
            "orders": orders,
            "conditions": {
                str(n): {
                    "samples_ms": x,
                    "runs": len(x),
                    "median_ms": float(np.median(x)),
                    "p95_ms": float(np.percentile(x, 95)),
                }
                for n, x in times.items()
            },
            "selected_num_threads": selected,
            "selected_output": save("thread_tuned_development", output),
            "selected_parity": parity,
            "selected_sAP10": quality,
            "scope": "Host CPU tuning only; no INT8 speedup claim and not measured on Android.",
            "source_code": info(__file__),
        },
    )
    print(
        "THREAD SELECTED", selected, {n: float(np.median(x)) for n, x in times.items()}, flush=True
    )


if __name__ == "__main__":
    main()
