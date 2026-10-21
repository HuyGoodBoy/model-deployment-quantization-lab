"""Only quality-eligible candidates receive matched timing and reused regression."""

import argparse
import gc
import time

import numpy as np
from refinement_mlsd import RefinedRunner, save
from runtime21 import infer
from support21 import RESULTS, ROOT, checked, info, now, pilot_data, read, write


class ContiguousRunner(RefinedRunner):
    def run(self, image):
        return np.ascontiguousarray(super().run(image))


def registry():
    previous = read(RESULTS / "exports.json")["models"]
    candidates = read(RESULTS / "refinement_quality_selection.json")["quality_eligible"]
    return {
        "float": previous["float"]["model"],
        "head_float": previous["head_float"]["model"],
        **{n: read(RESULTS / ("refinement_" + n + ".json"))["model"] for n in candidates},
    }


def benchmark(threads, session):
    target = RESULTS / f"refinement_contiguous_benchmark_t{threads}_s{session}.json"
    if target.exists():
        raise FileExistsError(target)
    models = registry()
    pilot, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    samples = {n: [] for n in models}
    started, orders = now(), []
    for cycle in range(5):
        names = list(models)
        offset = (cycle + session - 1) % len(names)
        order = names[offset:] + names[:offset]
        if (cycle + session) % 2 == 0:
            order.reverse()
        orders.append(order)
        for name in order:
            runner = ContiguousRunner(checked(models[name]), threads=threads)
            for image in inputs:
                runner.run(image)
            for image in inputs:
                before = time.perf_counter_ns()
                runner.run(image)
                samples[name].append((time.perf_counter_ns() - before) / 1e6)
            del runner
            gc.collect()
        print("REFINED BENCHMARK", threads, session, cycle + 1, "/5", flush=True)
    rows = {}
    for name, record in models.items():
        values = samples[name]
        runner = ContiguousRunner(checked(record), threads=threads)
        assert runner.run(inputs[0]).flags.c_contiguous
        output = infer(runner, inputs)
        reference_record = (
            read(RESULTS / "development.json")["models"][name]["output"]
            if name in ["float", "head_float"]
            else read(RESULTS / ("refinement_" + name + ".json"))["development_output"]
        )
        with np.load(checked(reference_record)) as stored:
            reference = stored["output"]
        quality = pilot.line_metrics(output, data["ground_truth"])
        assert np.allclose(reference, output, rtol=1e-4, atol=1e-4)
        rows[name] = {
            "model": record,
            "samples_ms": values,
            "runs": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "round_medians_ms": [float(np.median(values[i : i + 16])) for i in range(0, 80, 16)],
            "vs_development_exact": bool(np.array_equal(reference, output)),
            "sAP10": quality,
        }
        del runner, output
        gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "threads": threads,
            "session": session,
            "orders": orders,
            "models": rows,
            "quality_selection": info(RESULTS / "refinement_quality_selection.json"),
            "protocol": "Linux LiteRT2.2/XNNPACK;new Python process per thread/session;one runtime at a time;5rotated/reversed rounds16warmups+16timings. Shared application floatC-contiguousNCHW5 contract,actual adapters/copy included,load/preprocess/line decode excluded. np.ascontiguousarray applied to ALL variants. No profiling/competing experiment. Sessions on same host/day are not independent machines/days. Earlier refinement_benchmark_t*_s1 files are exploratory with a strided C1 output,excluded from final selection.",
            "source_code": info(__file__),
        },
    )
    print(
        "REFINED TIMING", threads, session, {n: r["median_ms"] for n, r in rows.items()}, flush=True
    )


def lock():
    target = RESULTS / "refinement_speed_selection.json"
    if target.exists():
        raise FileExistsError(target)
    names = read(RESULTS / "refinement_quality_selection.json")["quality_eligible"]
    benchmarks = {
        f"t{t}_s{s}": read(RESULTS / f"refinement_contiguous_benchmark_t{t}_s{s}.json")
        for t in [4, 8]
        for s in [1, 2]
    }
    # A consistent benefit over the fastest reference is required for a replacement proposal.
    eligible = [
        n
        for n in names
        if all(
            benchmarks[f"t8_s{s}"]["models"][n]["median_ms"]
            < benchmarks[f"t8_s{s}"]["models"]["float"]["median_ms"]
            for s in [1, 2]
        )
    ]
    write(
        target,
        {
            "locked_at_utc": now(),
            "quality_eligible": names,
            "faster_than_float8_in_both_sessions": eligible,
            "replacement_proposal": eligible[0] if eligible else "float8",
            "benchmarks": {
                k: info(RESULTS / f"refinement_contiguous_benchmark_{k}.json") for k in benchmarks
            },
            "rule": "Quality screened first;replacement needs lower matched8thread median in both sessions. Point-median screen only,not proof of general/significant speedup. Regression cannot rescue a candidate or change selection.",
            "source_code": info(__file__),
        },
    )
    print("SPEED LOCKED", eligible or ["float8"], flush=True)


def regression():
    target = RESULTS / "refinement_regression.json"
    if target.exists():
        raise FileExistsError(target)
    selection = read(RESULTS / "refinement_speed_selection.json")
    models = registry()
    pilot, _ = pilot_data()
    manifest_path = ROOT / "daily/2026-10-16/experiments/finalize/results/mlsd_manifest.json"
    manifest = read(manifest_path)
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    started, rows = now(), {}
    for name in selection["quality_eligible"]:
        runner = RefinedRunner(checked(models[name]), threads=4)
        output = infer(runner, inputs)
        rows[name] = {
            "model": models[name],
            "output": save(name + "_regression", output),
            "sAP10": pilot.line_metrics(output, truth),
        }
        del runner, output
        gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": rows,
            "images": len(inputs),
            "manifest": info(manifest_path),
            "selection": info(RESULTS / "refinement_speed_selection.json"),
            "scope": "Only development-quality-eligible candidates;54 reused images,not a fresh test. Original float/A references remain in regression.json. No reselection after regression.",
            "source_code": info(__file__),
        },
    )
    print(
        "REFINED REGRESSION",
        {n: r["sAP10"]["sAP10_subset_percent"] for n, r in rows.items()},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["benchmark", "lock", "regression"])
    parser.add_argument("--threads", type=int, choices=[4, 8], default=4)
    parser.add_argument("--session", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.stage == "benchmark":
        benchmark(args.threads, args.session)
    else:
        {"lock": lock, "regression": regression}[args.stage]()
