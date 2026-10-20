"""Matched LiteRT ablation; lock by development before reused regression data."""

import gc
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from mlsd_support import ART, RESULTS, ROOT, info, now, pilot, read, sha, write


def checked(record):
    path = ROOT / record["path"]
    assert sha(path) == record["sha256"]
    return path


def create(record, num_threads=4):
    runtime = Interpreter(model_path=str(checked(record)), num_threads=num_threads)
    runtime.allocate_tensors()
    signature = runtime.get_signature_runner("serving_default")
    input_details = signature.get_input_details()["input"]
    output_details = signature.get_output_details()["output"]

    def function(*, input):
        value = input
        if input_details["dtype"] != np.float32:
            scale, zero = input_details["quantization"]
            assert scale > 0
            limits = np.iinfo(input_details["dtype"])
            value = np.clip(np.rint(input / scale) + zero, limits.min, limits.max).astype(
                input_details["dtype"]
            )
        output = signature(input=value)["output"]
        if output_details["dtype"] != np.float32:
            scale, zero = output_details["quantization"]
            assert scale > 0
            output = (output.astype(np.float32) - zero) * scale
        return {"output": output}

    return runtime, function


def infer(function, inputs):
    output = np.concatenate([function(input=image[None])["output"] for image in inputs])
    assert output.shape == (len(inputs), 5, 256, 256) and np.isfinite(output).all()
    return output.transpose(0, 2, 3, 1)


def save(name, output):
    path = ART / (name + ".npy")
    if path.exists():
        assert np.array_equal(np.load(path), output), "Existing output changed"
    else:
        np.save(path, output, allow_pickle=False)
    return info(path)


def main():
    registry = read(RESULTS / "aeq_exports.json")["models"]
    data = pilot.load_datasets("mlsd")
    inputs = data["evaluation"][0].numpy()
    if not (RESULTS / "aeq_development.json").exists():
        rows = {}
        for name, item in registry.items():
            runtime, function = create(item["model"])
            output = infer(function, inputs)
            rows[name] = {
                "output": save(name + "_development", output),
                "sAP10": pilot.line_metrics(output, data["ground_truth"]),
            }
            print("DEVELOPMENT", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
            del runtime, function, output
            gc.collect()
        write(
            RESULTS / "aeq_development.json",
            {"finished_at_utc": now(), "images": 16, "models": rows},
        )
    if not (RESULTS / "aeq_benchmark.json").exists():
        timings = {name: [] for name in registry}
        names, orders = list(registry), []
        started = now()
        for cycle in range(5):
            order = names[cycle % len(names) :] + names[: cycle % len(names)]
            if cycle % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                runtime, function = create(registry[name]["model"])
                for image in inputs:
                    function(input=image[None])
                for image in inputs:
                    start = time.perf_counter_ns()
                    function(input=image[None])
                    timings[name].append((time.perf_counter_ns() - start) / 1e6)
                del runtime, function
                gc.collect()
            print("BENCHMARK", cycle + 1, "/5", flush=True)
        write(
            RESULTS / "aeq_benchmark.json",
            {
                "started_at_utc": started,
                "finished_at_utc": now(),
                "orders": orders,
                "protocol": "Linux LiteRT2.2 Interpreter CPU/XNNPACK, 4 threads, batch1; one interpreter at a time; 5 rotated/reversed rounds;16 warmups+16 timings/round. Shared float application I/O: adapter quantizes/dequantizes when model I/O is INT8; timing includes this adapter, signature binding/output copy, excludes load/preprocess/decoder; no profiling or other experiment active.",
                "models": {
                    name: {
                        "samples_ms": values,
                        "runs": len(values),
                        "median_ms": float(np.median(values)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "round_medians_ms": [
                            float(np.median(values[i : i + 16])) for i in range(0, 80, 16)
                        ],
                        "model": registry[name]["model"],
                    }
                    for name, values in timings.items()
                },
            },
        )
    dev = read(RESULTS / "aeq_development.json")["models"]
    bench = read(RESULTS / "aeq_benchmark.json")["models"]
    baseline_ap = dev["static"]["sAP10"]["sAP10_subset_percent"]
    eligible = [
        n
        for n in ["selective", "dynamic"]
        if dev[n]["sAP10"]["sAP10_subset_percent"] >= baseline_ap - 0.5
        and bench[n]["median_ms"] < bench["static"]["median_ms"]
    ]
    if not (RESULTS / "aeq_selection.json").exists():
        write(
            RESULTS / "aeq_selection.json",
            {
                "locked_at_utc": now(),
                "selected": min(eligible, key=lambda n: bench[n]["median_ms"])
                if eligible
                else "static",
                "eligible": eligible,
                "regression_used": False,
                "criterion": "Fastest alternative within0.5pp of static development sAP10 and faster than static; otherwise static. Also compare all results to float; selection is not deployment approval.",
                "development": info(RESULTS / "aeq_development.json"),
                "benchmark": info(RESULTS / "aeq_benchmark.json"),
            },
        )
    target = RESULTS / "aeq_regression.json"
    if target.exists():
        return
    selection = read(RESULTS / "aeq_selection.json")
    manifest_path = ROOT / "daily/2026-10-16/experiments/finalize/results/mlsd_manifest.json"
    manifest = read(manifest_path)
    inputs = np.load(checked(manifest["inputs"]))
    truth = read(checked(manifest["ground_truth"]))
    started = now()
    rows = {}
    for name, item in registry.items():
        runtime, function = create(item["model"])
        output = infer(function, inputs)
        rows[name] = {
            "output": save(name + "_regression", output),
            "sAP10": pilot.line_metrics(output, truth),
        }
        print("REGRESSION", name, rows[name]["sAP10"]["sAP10_subset_percent"], flush=True)
        del runtime, function, output
        gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "images": len(inputs),
            "selected_before_regression": selection["selected"],
            "models": rows,
            "selection": info(RESULTS / "aeq_selection.json"),
            "manifest": info(manifest_path),
            "scope": "54 previously-used YorkUrban test images; reused regression only, not fresh independent holdout. No reselection after results.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
