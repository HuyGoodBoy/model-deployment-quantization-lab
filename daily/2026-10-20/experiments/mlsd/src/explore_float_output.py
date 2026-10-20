"""Post-diagnosis exploratory candidate, development only; no regression tuning."""

import gc
import re
import time

import numpy as np
from ai_edge_quantizer import calibrator, qtyping, quantizer, recipe
from evaluate_litert import create, infer, save
from mlsd_support import ART, RESULTS, info, now, pilot, read, write
from quantize_litert import graph


def main():
    target = RESULTS / "aeq_float_output_development.json"
    if target.exists():
        raise FileExistsError(target)
    protocol = RESULTS / "aeq_float_output_protocol.json"
    write(
        protocol,
        {
            "declared_at_utc": now(),
            "scope": "Exploratory after main candidate lock; development16 only. Do not run/reselect on54regression images.",
            "hypothesis": "Shared output scale2.335 destroys center logits. Keep final Conv and output transpose float, plus the3 ORT-selected slow integer scopes. Compare to original selective and float on development; gate within0.5pp of float development sAP10 and faster than float. No deployment claim without independent validation.",
        },
    )
    exports = read(RESULTS / "aeq_exports.json")
    float_path = ART / "float_conversion/qat_weights_fp32_float32.tflite"
    scopes = graph(float_path)
    last = [r for r in scopes if "block16/conv3/Conv_conv_filter" in r["weight"]]
    assert len(last) == 1
    selected = exports["selected"] + last
    data = pilot.load_datasets("mlsd")
    q = quantizer.Quantizer(float_path, recipe.static_wi8_ai8())
    for row in selected:
        q.update_quantization_recipe(
            "^" + re.escape(row["scope"]) + ";$",
            qtyping.TFLOperationName.ALL_SUPPORTED,
            algorithm_key="no_quantize",
        )
    q.update_quantization_recipe(
        "^output;$", qtyping.TFLOperationName.ALL_SUPPORTED, algorithm_key="no_quantize"
    )
    calibration = [{"input": x[None]} for x in data["calibration"][0].numpy()]
    statistics = q.calibrate(
        {"serving_default": calibration},
        num_threads=4,
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
    )
    result = q.quantize(calibration_result=statistics, enable_progress_bar=False)
    path = ART / "aeq_float_output.tflite"
    result.export_model(str(path), overwrite=False)
    rows = graph(path)
    for row in selected:
        matches = [r for r in rows if r["scope"] == row["scope"]]
        assert len(matches) == 1 and "float32" in matches[0]["weight_dtype"]
        assert "float32" in matches[0]["output_dtype"]
    runtime, function = create(info(path))
    # Inspect native I/O, not only the float application adapter.
    signature = runtime.get_signature_runner("serving_default")
    output_details = signature.get_output_details()["output"]
    assert output_details["dtype"] == np.float32
    inputs = data["evaluation"][0].numpy()
    output = infer(function, inputs)
    quality = pilot.line_metrics(output, data["ground_truth"])
    write(
        target,
        {
            "finished_at_utc": now(),
            "protocol": info(protocol),
            "model": info(path),
            "graph": rows,
            "native_output_dtype": str(output_details["dtype"]),
            "sAP10": quality,
            "output": save("float_output_development", output),
            "development_only": True,
            "source_code": info(__file__),
        },
    )
    del runtime, function, signature
    gc.collect()
    print("FLOAT OUTPUT DEVELOPMENT", quality["sAP10_subset_percent"], flush=True)
    # Benchmarked in a separate process after all other compute finishes.


def benchmark():
    target = RESULTS / "aeq_float_output_benchmark.json"
    if target.exists():
        raise FileExistsError(target)
    exports = read(RESULTS / "aeq_exports.json")["models"]
    candidate = read(RESULTS / "aeq_float_output_development.json")
    records = {
        "float": exports["float"]["model"],
        "selective": exports["selective"]["model"],
        "float_output": candidate["model"],
    }
    inputs = pilot.load_datasets("mlsd")["evaluation"][0].numpy()
    times = {n: [] for n in records}
    started = now()
    orders = []
    for cycle in range(5):
        names = list(records)
        order = names[cycle % 3 :] + names[: cycle % 3]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            runtime, function = create(records[name])
            for x in inputs:
                function(input=x[None])
            for x in inputs:
                start = time.perf_counter_ns()
                function(input=x[None])
                times[name].append((time.perf_counter_ns() - start) / 1e6)
            del runtime, function
            gc.collect()
        print("FLOAT OUTPUT BENCHMARK", cycle + 1, "/5", flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "orders": orders,
            "protocol": "Linux LiteRT2.2 CPU/XNNPACK4threads; same float application I/O adapter;5rotated/reversed rounds16warmups+16timings; excludes loading/preprocess/line decoder; no competing experiment.",
            "models": {
                n: {
                    "model": records[n],
                    "samples_ms": x,
                    "runs": len(x),
                    "median_ms": float(np.median(x)),
                    "p95_ms": float(np.percentile(x, 95)),
                }
                for n, x in times.items()
            },
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    benchmark() if args.benchmark else main()
