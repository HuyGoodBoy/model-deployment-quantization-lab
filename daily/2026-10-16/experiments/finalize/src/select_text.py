"""Select one static text export on development before final evaluation."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
from common import ART, RESULTS, checked, info, profile_summary, read, session, write
from export_parity import explicit_rounding
from export_text import infer
from train_continue import pilot


def main():
    if (RESULTS / "text_selection.json").exists():
        raise FileExistsError("Text selection is already locked")
    if (RESULTS / "text_final.json").exists():
        raise RuntimeError("Cannot select after final evaluation")
    export = read(RESULTS / "text_export.json")
    source = checked(export["models"]["qat_static_integer"]["model"])
    path = ART / "text_qat_static_explicit_rounding.onnx"
    model, quantizers = explicit_rounding(onnx.load(source))
    onnx.save(model, path)
    data = pilot.load_datasets("text")
    inputs = tuple(x.numpy() for x in data["evaluation"])
    reference = np.load(checked(export["fake_quant_reference"]))
    candidates = {}
    for name, candidate in {"static_integer": source, "static_explicit_rounding": path}.items():
        runtime = session(candidate)
        outputs = infer(runtime, inputs)
        del runtime
        target = ART / f"text_{name}_selection_development.npy"
        np.save(target, outputs, allow_pickle=False)
        parity = pilot.numerical(reference, outputs)
        agreement = float(np.mean(outputs.argmax(-1) == reference.argmax(-1)))
        delta = float(
            (
                np.mean(outputs.argmax(-1) == data["labels"])
                - np.mean(reference.argmax(-1) == data["labels"])
            )
            * 100
        )
        passed = parity["relative_l2"] <= 0.02 and agreement >= 0.99 and abs(delta) <= 1.0 + 1e-10
        candidates[name] = {
            "model": info(candidate),
            "output": info(target),
            "parity": parity,
            "label_agreement": agreement,
            "accuracy_percent": float(np.mean(outputs.argmax(-1) == data["labels"]) * 100),
            "accuracy_delta_pp": delta,
            "gate_passed": passed,
        }
        print("TEXT CANDIDATE", name, parity["relative_l2"], agreement, passed, flush=True)
    passed_names = [name for name, item in candidates.items() if item["gate_passed"]]
    chosen = min(
        passed_names or list(candidates),
        key=lambda n: (-candidates[n]["label_agreement"], candidates[n]["parity"]["relative_l2"]),
    )
    runtime = session(checked(candidates[chosen]["model"]), profile=ART / "text_selected_profile")
    runtime.run(None, {"input_ids": inputs[0][:1], "attention_mask": inputs[1][:1]})
    profile = profile_summary(Path(runtime.end_profiling()))
    del runtime
    assert (
        sum(
            count
            for op, count in profile["unique_executed_operators"].items()
            if "MatMulInteger" in op
        )
        == 38
    )
    write(
        RESULTS / "text_selection.json",
        {
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "rule": "Development only: prefer unchanged task-specific gate passing; then highest label agreement, then lowest relative L2; if none pass preserve failed gate",
            "holdout_used": False,
            "selected": chosen,
            "candidates": candidates,
            "quantizers_rewritten": len(quantizers),
            "runtime": profile,
            "release_gate_passed": candidates[chosen]["gate_passed"],
            "source_code": info(__file__),
        },
    )
    print("LOCKED TEXT", chosen, flush=True)


if __name__ == "__main__":
    main()
