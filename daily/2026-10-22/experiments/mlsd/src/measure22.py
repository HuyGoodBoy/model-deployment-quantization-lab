"""Matched application I/O timing; selection is frozen before reused regression."""

import argparse
import gc
import time

import numpy as np
from mlsd22 import DAY21, RESULTS, ROOT, checked, info, now, pilot_data, read, save, write
from refinement_measure import ContiguousRunner
from runtime21 import infer


def benchmark(threads, session):
    target = RESULTS / f"benchmark_t{threads}_s{session}.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read(RESULTS / "quality_selection.json")
    models = selection["models"]
    _, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    names = list(models)
    samples = {n: [] for n in names}
    started, orders = now(), []
    for cycle in range(5):
        offset = (cycle + session - 1) % len(names)
        order = names[offset:] + names[:offset]
        if (cycle + session) % 2 == 0:
            order.reverse()
        orders.append(order)
        for name in order:
            runner = ContiguousRunner(checked(models[name]), threads=threads)
            assert runner.run(inputs[0]).flags.c_contiguous
            for image in inputs:
                runner.run(image)
            for image in inputs:
                before = time.perf_counter_ns()
                runner.run(image)
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
            del runner
            gc.collect()
        print("MLSD BENCHMARK", threads, session, cycle + 1, "/5", flush=True)
    rows = {
        name: {
            "model": models[name],
            "samples_ms": values,
            "runs": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "round_medians_ms": [float(np.median(values[i : i + 16])) for i in range(0, 80, 16)],
        }
        for name, values in samples.items()
    }
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": rows,
            "threads": threads,
            "session": session,
            "orders": orders,
            "quality_selection": info(RESULTS / "quality_selection.json"),
            "protocol": "Linux LiteRT2.2/XNNPACK;one runtime at a time,no competing experiment;new process per thread/session.5rotated/reversed rounds16warmups+16samples;one additional output-contract check before warmups. Same floatNCHW input/C-contiguousNCHW5 output;includes actual inputquant/copy/outputadapter;excludes load/preprocess/line decoder. Two sessions same host/day do not establish cross-device/day stability.",
            "source_code": info(__file__),
        },
    )
    print("MLSD TIMING", threads, session, {n: r["median_ms"] for n, r in rows.items()}, flush=True)


def paired(a, b, seed):
    a, b = np.asarray(a).reshape(5, 16), np.asarray(b).reshape(5, 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 5, size=(2000, 5))
    gain = [100 * (1 - np.median(b[i]) / np.median(a[i])) for i in indices]
    return {
        "median_reduction_percent": float(100 * (1 - np.median(b) / np.median(a))),
        "paired_round_bootstrap95_percent": np.percentile(gain, [2.5, 97.5]).tolist(),
        "faster_rounds": int(np.sum(np.median(b, axis=1) < np.median(a, axis=1))),
        "rounds": 5,
        "resamples": 2000,
        "seed": seed,
    }


def lock():
    target = RESULTS / "speed_selection.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "configuration_validation.json")["passed"]
    quality = read(RESULTS / "quality_selection.json")
    comparisons, sources = {}, {}
    benchmarks = {}
    for threads in [4, 8]:
        for session in [1, 2]:
            key = f"t{threads}_s{session}"
            path = RESULTS / ("benchmark_" + key + ".json")
            rows = read(path)["models"]
            benchmarks[key] = rows
            sources[key] = info(path)
            comparisons[key] = {}
            pairs = [("C1", "C1_roundtrip")]
            pairs += [
                (control, candidate)
                for candidate in quality["quality_eligible"]
                for control in ["float", "C1", "C1_roundtrip"]
            ]
            if all(n in quality["quality_eligible"] for n in ["M1", "M2"]):
                pairs.append(("M1", "M2"))
            for index, (reference, candidate) in enumerate(pairs):
                comparisons[key][candidate + "_vs_" + reference] = paired(
                    rows[reference]["samples_ms"],
                    rows[candidate]["samples_ms"],
                    20261022 + threads * 100 + session * 10 + index,
                )
    eligible = []
    for candidate in quality["quality_eligible"]:
        if all(
            benchmarks[f"t8_s{s}"][candidate]["median_ms"]
            < benchmarks[f"t8_s{s}"][ref]["median_ms"]
            and comparisons[f"t8_s{s}"][candidate + "_vs_" + ref][
                "paired_round_bootstrap95_percent"
            ][0]
            > 0
            for s in [1, 2]
            for ref in ["float", "C1"]
        ):
            eligible.append(candidate)
    selected = (
        min(eligible, key=lambda n: sum(benchmarks[f"t8_s{s}"][n]["median_ms"] for s in [1, 2]))
        if eligible
        else "float"
    )
    if selected == "M2" and "M1" in eligible:
        incremental = [comparisons[f"t8_s{s}"]["M2_vs_M1"] for s in [1, 2]]
        if not all(
            r["median_reduction_percent"] >= 1 and r["paired_round_bootstrap95_percent"][0] > 0
            for r in incremental
        ):
            selected = "M1"
    write(
        target,
        {
            "locked_at_utc": now(),
            "selected": selected,
            "eligible_8threads": eligible,
            "quality_eligible": quality["quality_eligible"],
            "comparisons": comparisons,
            "benchmarks": sources,
            "rule": "Within0.5pp developmentAP;8thread candidate must beat FP32 and oldC1 in both sessions with positive lower paired-round95interval. Choose lowest mean of two medians;prefer M1 unless M2 incremental gain>=1% and positive interval in both. Approximate5cluster intervals,not deployment certification. No regression-based selection.",
            "source_code": info(__file__),
        },
    )
    print("MLSD SELECTED", selected, "eligible", eligible, flush=True)


def regression():
    target = RESULTS / "regression.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read(RESULTS / "speed_selection.json")
    models = read(RESULTS / "quality_selection.json")["models"]
    pilot, _ = pilot_data()
    manifest_path = ROOT / "daily/2026-10-16/experiments/finalize/results/mlsd_manifest.json"
    manifest = read(manifest_path)
    images = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    reference = read(DAY21 / "results/refinement_regression.json")["models"]["C1"]
    with np.load(checked(reference["output"])) as saved:
        original = saved["output"]
    started, rows = now(), {}
    for name in selection["quality_eligible"]:
        runner = ContiguousRunner(checked(models[name]), threads=8)
        output = infer(runner, images)
        rows[name] = {
            "model": models[name],
            "sAP10": pilot.line_metrics(output, truth),
            "output": save(name + "_regression", output),
            "vs_C1": pilot.numerical(original, output),
        }
        del runner
        gc.collect()
        print("MLSD REGRESSION", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": rows,
            "selection": info(RESULTS / "speed_selection.json"),
            "selected_before_regression": selection["selected"],
            "manifest": info(manifest_path),
            "images": len(images),
            "scope": "54 already-used images,not independent holdout;quality eligible only,no reselection after seeing regression.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["benchmark", "lock", "regression"])
    parser.add_argument("--threads", type=int, choices=[4, 8], default=8)
    parser.add_argument("--session", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.stage == "benchmark":
        benchmark(args.threads, args.session)
    else:
        {"lock": lock, "regression": regression}[args.stage]()
