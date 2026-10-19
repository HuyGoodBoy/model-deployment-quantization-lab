"""Same-artifact control: disable only the named QDQ selector/action pass."""

import gc
import time

import numpy as np
from support import (
    RESULTS,
    baselines,
    checked,
    info,
    now,
    pilot,
    profile_model,
    save_output,
    session,
    write,
)


def main():
    if (RESULTS / "fusion_control.json").exists():
        raise FileExistsError("Control already recorded")
    pass_name = "QDQSelectorActionTransformer"
    source = baselines()["qdq_all"]
    write(
        RESULTS / "fusion_control_protocol.json",
        {
            "locked_at_utc": now(),
            "model": source["model"],
            "disabled_pass": pass_name,
            "scope": "Supplementary causal control of this optimizer pass on the same graph, all other optimization levels retained. Does not isolate each Conv fusion from all other rules in the pass. Not used to select A/B.",
            "protocol": "Two conditions, 5 alternating-order rounds, 16 warmups plus16 timed development inputs/round; profiling in separate sessions after timing; 4 intra/1 inter threads, spinning off",
            "source_code": info(__file__),
        },
    )
    data = pilot.load_datasets("mlsd")
    inputs = data["evaluation"][0].numpy()
    conditions = {"enabled": None, "disabled": [pass_name]}
    timings = {name: [] for name in conditions}
    outputs = {}
    for cycle in range(5):
        order = list(conditions) if cycle % 2 == 0 else list(conditions)[::-1]
        for name in order:
            runtime = session(checked(source["model"]), disabled_optimizers=conditions[name])
            for image in inputs:
                runtime.run(None, {"input": image[None]})
            result = []
            for image in inputs:
                start = time.perf_counter_ns()
                output = runtime.run(None, {"input": image[None]})[0]
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
                if cycle == 0:
                    result.append(output)
            if cycle == 0:
                outputs[name] = np.concatenate(result).transpose(0, 2, 3, 1)
            del runtime
            gc.collect()
        print("FUSION CONTROL", cycle + 1, "/5", flush=True)
    profiles = {}
    for name, disabled_passes in conditions.items():
        item = {**source, "disabled_optimizers": disabled_passes}
        profiles[name] = profile_model("fusion_pass_" + name, item, inputs)
    enabled = profiles["enabled"]["unique_operators"].get("QLinearConv", 0)
    disabled = profiles["disabled"]["unique_operators"].get("QLinearConv", 0)
    # Evidence must show the installed runtime recognized and disabled this pass.
    if enabled != 22 or disabled != 0:
        raise RuntimeError(f"Cannot validate pass control: {enabled} vs {disabled}")
    write(
        RESULTS / "fusion_control.json",
        {
            "measured_at_utc": now(),
            "protocol": info(RESULTS / "fusion_control_protocol.json"),
            "same_artifact": source["model"],
            "disabled_pass": pass_name,
            "named_pass_effect_confirmed": True,
            "conditions": {
                name: {
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "samples_ms": values,
                    "runs": len(values),
                    "profile": profiles[name],
                    "development_output": save_output(
                        "fusion_pass_" + name + "_development", outputs[name]
                    ),
                    "sAP10": pilot.line_metrics(outputs[name], data["ground_truth"]),
                }
                for name, values in timings.items()
            },
            "numeric_difference": pilot.numerical(outputs["disabled"], outputs["enabled"]),
            "scope": "Effect of named QDQ selector/action optimizer pass, not all optimization and not pure Conv-kernel cost. Same artifact/precision specification; numeric kernel rounding may differ.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
