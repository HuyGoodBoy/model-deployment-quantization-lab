"""Application-stage wall times and delegate boundary inventory, NOT node profiling."""

import gc

import numpy as np
from mlsd22 import RESULTS, checked, info, now, pilot_data, read, write
from refinement_io_profile import boundary, staged
from refinement_measure import ContiguousRunner


def main():
    target = RESULTS / "io_profile.json"
    if target.exists():
        raise FileExistsError(target)
    models = read(RESULTS / "quality_selection.json")["models"]
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
