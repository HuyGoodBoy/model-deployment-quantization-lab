"""Application-stage wall times and delegate boundary inventory, NOT node profiling."""

import gc
import time
from collections import Counter

import numpy as np
from refinement_measure import ContiguousRunner, registry
from runtime21 import dequantize, quantize
from support21 import RESULTS, checked, info, now, pilot_data, write


def boundary(runtime):
    tensors = {int(t["index"]): t for t in runtime.get_tensor_details()}
    ops = runtime._get_ops_details()
    delegates = []
    for op in ops:
        if op["op_name"] != "DELEGATE":
            continue

        def describe(ids):
            return [
                {
                    "index": int(i),
                    "name": tensors[int(i)]["name"],
                    "dtype": str(tensors[int(i)]["dtype"]),
                    "shape": tensors[int(i)]["shape"].tolist(),
                }
                for i in ids
                if int(i) >= 0
            ]

        delegates.append(
            {
                "node_index": int(op["index"]),
                "inputs": describe(op["inputs"]),
                "outputs": describe(op["outputs"]),
            }
        )
    return {
        "static_original_op_counts": dict(
            Counter(op["op_name"] for op in ops if op["op_name"] != "DELEGATE")
        ),
        "delegate_node_count": len(delegates),
        "delegate_boundaries": delegates,
        "limitation": "_get_ops_details includes original nodes even after delegation;this is NOT the active execution plan. Original Q/DQ counts are not CPU kernel invocation counts.",
    }


def staged(runner, image):
    runtime = runner.runtime
    start = time.perf_counter_ns()
    value = quantize(image[None], runner.inputs["input"])
    after_quant = time.perf_counter_ns()
    runtime.set_tensor(int(runner.inputs["input"]["index"]), value)
    after_set = time.perf_counter_ns()
    runtime.invoke()
    after_invoke = time.perf_counter_ns()
    outputs = {k: runtime.get_tensor(int(d["index"])) for k, d in runner.outputs.items()}
    after_get = time.perf_counter_ns()
    outputs = {k: dequantize(v, runner.outputs[k]) for k, v in outputs.items()}
    if "output" in outputs:
        output = outputs["output"]
        if output.shape == (1, 256, 256, 5):
            output = output.transpose(0, 3, 1, 2)
    else:
        for name, channels in [("center", 1), ("displacement", 4)]:
            if outputs[name].shape == (1, 256, 256, channels):
                outputs[name] = outputs[name].transpose(0, 3, 1, 2)
        output = np.concatenate([outputs["center"], outputs["displacement"]], axis=1)
    output = np.ascontiguousarray(output)
    finish = time.perf_counter_ns()
    stamps = [start, after_quant, after_set, after_invoke, after_get, finish]
    names = [
        "application_quantize",
        "set_tensor_copy",
        "native_invoke",
        "get_tensor_copy",
        "application_output_adapter",
    ]
    times = {name: (b - a) / 1e6 for name, a, b in zip(names, stamps[:-1], stamps[1:], strict=True)}
    times["total"] = (finish - start) / 1e6
    assert output.shape == (1, 5, 256, 256)
    return output, times


def main():
    target = RESULTS / "refinement_io_profile.json"
    if target.exists():
        raise FileExistsError(target)
    models = registry()
    _, data = pilot_data()
    inputs = data["evaluation"][0].numpy()[:8]
    started, rows = now(), {}
    for threads in [4, 8]:
        for name, record in models.items():
            runner = ContiguousRunner(checked(record), threads=threads)
            inventory = boundary(runner.runtime)
            reference = runner.run(inputs[0])
            # SignatureRunner owns a native-wrapper reference. Release it before
            # direct Interpreter.invoke() so the normal safety guard remains on.
            del runner.signature
            gc.collect()
            output, _ = staged(runner, inputs[0])
            exact = bool(np.array_equal(reference, output))
            assert exact and output.flags.c_contiguous and reference.flags.c_contiguous, (
                threads,
                name,
            )
            for image in inputs:
                staged(runner, image)
            samples = []
            for _ in range(3):
                for image in inputs:
                    _, times = staged(runner, image)
                    samples.append(times)
            medians = {k: float(np.median([r[k] for r in samples])) for k in samples[0]}
            rows[f"{name}_t{threads}"] = {
                "model": record,
                "threads": threads,
                "inventory": inventory,
                "signature_vs_staged_output_exact": exact,
                "samples_ms": samples,
                "medians_ms": medians,
            }
            print(
                "IO PROFILE",
                name,
                threads,
                medians,
                "delegate_nodes",
                inventory["delegate_node_count"],
                flush=True,
            )
            del runner, output, reference
            gc.collect()
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "models": rows,
            "protocol": "Same LiteRT2.2 CPU/XNNPACK4/8threads;24staged samples/model after8warmups on first8development inputs. Different direct-set/invoke/get plumbing from signature benchmark;diagnostic only,not a replacement speed benchmark. Native invoke includes delegate,builtin,Q/DQ and compute;no per-op attribution. Median total computed per whole sample,not sum of stage medians. No competing experiment.",
            "native_profiler_availability": "Pinned Python Interpreter has no exposed profiler-result/execution-plan API. No node times or unsupported pointer hooks invented. Official C++ benchmark build needed for deeper per-kernel attribution.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
