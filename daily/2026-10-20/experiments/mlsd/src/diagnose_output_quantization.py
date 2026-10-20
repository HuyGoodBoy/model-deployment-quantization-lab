"""Development-only counterfactual: effect of the final shared INT8 output range."""

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from mlsd_support import RESULTS, ROOT, info, now, pilot, read, write


def main():
    exports = read(RESULTS / "aeq_exports.json")["models"]
    dev = read(RESULTS / "aeq_development.json")["models"]
    data = pilot.load_datasets("mlsd")
    floating = np.load(ROOT / dev["float"]["output"]["path"])
    rows = {}
    for name in ["static", "selective"]:
        runtime = Interpreter(model_path=str(ROOT / exports[name]["model"]["path"]), num_threads=4)
        runtime.allocate_tensors()
        details = runtime.get_signature_runner("serving_default").get_output_details()["output"]
        scale, zero = details["quantization"]
        limits = np.iinfo(details["dtype"])
        quantized = np.clip(np.rint(floating / scale) + zero, limits.min, limits.max)
        restored = (quantized - zero) * scale
        actual = np.load(ROOT / dev[name]["output"]["path"])
        rows[name] = {
            "output_scale": scale,
            "output_zero_point": zero,
            "dtype": str(details["dtype"]),
            "sAP10_only_output_roundtrip_on_float": pilot.line_metrics(
                restored, data["ground_truth"]
            ),
            "float_sAP10": dev["float"]["sAP10"]["sAP10_subset_percent"],
            "actual_sAP10": dev[name]["sAP10"]["sAP10_subset_percent"],
            "output_only_error": pilot.numerical(floating, restored),
            "actual_center_error": pilot.numerical(floating[..., :1], actual[..., :1]),
            "actual_displacement_error": pilot.numerical(floating[..., 1:], actual[..., 1:]),
        }
        print(
            "OUTPUT ROUNDTRIP",
            name,
            scale,
            rows[name]["sAP10_only_output_roundtrip_on_float"]["sAP10_subset_percent"],
            flush=True,
        )
    write(
        RESULTS / "aeq_output_diagnosis.json",
        {
            "finished_at_utc": now(),
            "development_only": True,
            "images": len(floating),
            "models": rows,
            "scope": "Apply the actual shared output INT8 scale to otherwise unchanged float output. This isolates this boundary's counterfactual error, not all error in the full quantized graph. No post-regression model selection or new release candidate.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
