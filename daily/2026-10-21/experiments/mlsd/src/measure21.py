"""Development gate, matched performance and pre-locked reused regression."""

import argparse
import gc
import time

import numpy as np
from runtime21 import Runner, infer, io_details
from support21 import ART, RESULTS, ROOT, checked, info, now, pilot_data, read, write


def save(name, values):
    path = ART / (name + ".npz")
    assert not path.exists(), path
    np.savez_compressed(path, output=values)
    return info(path)


def development():
    target = RESULTS / "development.json"
    if target.exists():
        raise FileExistsError(target)
    pilot, data = pilot_data()
    registry = read(RESULTS / "exports.json")
    outputs, rows = {}, {}
    for name, item in registry["models"].items():
        runner = Runner(checked(item["model"]))
        output = infer(runner, data["evaluation"][0].numpy())
        rows[name] = {
            "output": save(name + "_development", output),
            "sAP10": pilot.line_metrics(output, data["ground_truth"]),
            "native_IO": io_details(runner),
            "model": item["model"],
        }
        outputs[name] = output
        print("DEVELOPMENT", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
        del runner
        gc.collect()
    for name, row in rows.items():
        row["vs_float"] = {
            "all": pilot.numerical(outputs["float"], outputs[name]),
            "center": pilot.numerical(outputs["float"][..., :1], outputs[name][..., :1]),
            "displacement": pilot.numerical(outputs["float"][..., 1:], outputs[name][..., 1:]),
        }
    baseline = rows["float"]["sAP10"]["sAP10_subset_percent"]
    eligible = [
        name
        for name in ["head_float", "split_int8"]
        if rows[name]["sAP10"]["sAP10_subset_percent"] >= baseline - 0.5
    ]
    write(
        target,
        {
            "finished_at_utc": now(),
            "models": rows,
            "images": 16,
            "quality_gate": "Within0.5pp of same-checkpoint FP32 on development; benchmark must additionally beat matched FP32. Not deployment proof.",
            "quality_eligible": eligible,
            "exports": info(RESULTS / "exports.json"),
            "source_code": info(__file__),
        },
    )


def benchmark():
    target = RESULTS / "benchmark.json"
    if target.exists():
        raise FileExistsError(target)
    registry = read(RESULTS / "exports.json")["models"]
    _, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    names = ["float", "static", "head_float", "split_int8", "split_float"]
    samples = {n: [] for n in names}
    started, orders = now(), []
    for cycle in range(5):
        order = names[cycle % len(names) :] + names[: cycle % len(names)]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            runner = Runner(checked(registry[name]["model"]), threads=4)
            for image in inputs:
                runner.run(image)
            for image in inputs:
                before = time.perf_counter_ns()
                runner.run(image)
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
            del runner
            gc.collect()
        print("BENCHMARK", cycle + 1, "/5", flush=True)
    rows = {
        name: {
            "model": registry[name]["model"],
            "samples_ms": values,
            "runs": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "round_medians_ms": [float(np.median(values[n : n + 16])) for n in range(0, 80, 16)],
        }
        for name, values in samples.items()
    }
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "orders": orders,
            "models": rows,
            "protocol": "Linux LiteRT2.2 CPU/XNNPACK4threads,one interpreter at a time;5rotated/reversed rounds16warmups+16timings. Same float application NCHW5-channel output contract; includes actual quantize/dequantize, NHWC->NCHW adaptation and split-output merge when required. Excludes loading,preprocess,line decoder;no competing workload,profiling off.",
            "source_code": info(__file__),
        },
    )
    dev = read(RESULTS / "development.json")
    eligible = [
        n for n in dev["quality_eligible"] if rows[n]["median_ms"] < rows["float"]["median_ms"]
    ]
    selected = min(eligible, key=lambda n: rows[n]["median_ms"]) if eligible else "float"
    write(
        RESULTS / "selection.json",
        {
            "locked_at_utc": now(),
            "selected": selected,
            "eligible": eligible,
            "regression_used": False,
            "criterion": "Fastest declared INT8 alternative within0.5pp of float development AP and faster than matched float4threads,otherwise keep float. No selection by regression.",
            "development": info(RESULTS / "development.json"),
            "benchmark": info(target),
        },
    )
    print("LOCKED", selected, flush=True)


def regression():
    target = RESULTS / "regression.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read(RESULTS / "selection.json")
    registry = read(RESULTS / "exports.json")["models"]
    pilot, _ = pilot_data()
    manifest_path = ROOT / "daily/2026-10-16/experiments/finalize/results/mlsd_manifest.json"
    manifest = read(manifest_path)
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    started, rows = now(), {}
    # Show both declared candidates without choosing again after looking at regression.
    for name in ["float", "head_float", "split_int8"]:
        runner = Runner(checked(registry[name]["model"]))
        outputs = infer(runner, inputs)
        rows[name] = {
            "output": save(name + "_regression", outputs),
            "sAP10": pilot.line_metrics(outputs, truth),
            "model": registry[name]["model"],
        }
        print("REGRESSION", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
        del runner, outputs
        gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "images": len(inputs),
            "models": rows,
            "selection": info(RESULTS / "selection.json"),
            "selected_before_regression": selection["selected"],
            "manifest": info(manifest_path),
            "scope": "54 already-used YorkUrban images,regression not fresh holdout;show both declared candidates,no post-regression reselection",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["development", "benchmark", "regression"])
    args = parser.parse_args()
    {"development": development, "benchmark": benchmark, "regression": regression}[args.stage]()
