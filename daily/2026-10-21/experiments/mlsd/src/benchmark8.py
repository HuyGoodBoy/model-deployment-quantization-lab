"""Matched8thread confirmation for development-quality eligible recipes only."""

import gc
import time

import numpy as np
from measure21 import save
from runtime21 import Runner, infer
from support21 import RESULTS, checked, info, now, pilot_data, read, write


def main():
    target = RESULTS / "benchmark8.json"
    if target.exists():
        raise FileExistsError(target)
    registry = read(RESULTS / "exports.json")["models"]
    dev = read(RESULTS / "development.json")
    names = ["float", *dev["quality_eligible"]]
    assert names == ["float", "head_float"]
    pilot, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    samples = {n: [] for n in names}
    started, orders = now(), []
    for cycle in range(5):
        order = names if cycle % 2 == 0 else names[::-1]
        orders.append(order)
        for name in order:
            runner = Runner(checked(registry[name]["model"]), threads=8)
            for image in inputs:
                runner.run(image)
            for image in inputs:
                before = time.perf_counter_ns()
                runner.run(image)
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
            del runner
            gc.collect()
        print("BENCHMARK8", cycle + 1, "/5", flush=True)
    # Check precision after timing, no output serialization or metrics in timed loops.
    rows = {}
    for name in names:
        runner = Runner(checked(registry[name]["model"]), threads=8)
        output = infer(runner, inputs)
        with np.load(checked(dev["models"][name]["output"])) as saved:
            reference = saved["output"]
        exact = np.array_equal(reference, output)
        quality = pilot.line_metrics(output, data["ground_truth"])
        assert (
            abs(
                quality["sAP10_subset_percent"]
                - dev["models"][name]["sAP10"]["sAP10_subset_percent"]
            )
            < 0.05
        )
        rows[name] = {
            "model": registry[name]["model"],
            "samples_ms": samples[name],
            "runs": len(samples[name]),
            "median_ms": float(np.median(samples[name])),
            "p95_ms": float(np.percentile(samples[name], 95)),
            "thread4_vs8_output_exact": exact,
            "sAP10": quality,
            "output": save(name + "_threads8_development", output),
        }
        del runner, output
        gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": rows,
            "orders": orders,
            "protocol": "Linux LiteRT2.2 CPU/XNNPACK8threads,5alternating rounds16warmups+16timings;one runtime at a time,no competing work;float application I/O adapters included;loading/preprocess/line decoder excluded. Variants eligible by development before this8thread comparison. No regression-based choice.",
            "source_code": info(__file__),
        },
    )
    print("BENCHMARK8 RESULT", {n: r["median_ms"] for n, r in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
