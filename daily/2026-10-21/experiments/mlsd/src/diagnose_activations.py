"""Diagnostic no-delegate tensor comparison, distinct from performance execution."""

import csv

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from runtime21 import dequantize, quantize
from support21 import RESULTS, checked, info, inherited, now, pilot_data, write


def prepare(path):
    runtime = Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    runtime.allocate_tensors()
    tensors = {t["index"]: t for t in runtime.get_tensor_details()}
    records = []
    for op in runtime._get_ops_details():
        if op["op_name"] not in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
            continue
        output = tensors[int(op["outputs"][0])]
        records.append(
            {
                "scope": output["name"],
                "op": op["op_name"],
                "output": output,
                "weight": tensors[int(op["inputs"][1])]["name"],
            }
        )
    assert len(records) == 41
    return runtime, records


def error(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return {
        "relative_l2": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30)),
        "max_abs": float(np.max(np.abs(b - a))),
        "rmse": float(np.sqrt(np.mean((b - a) ** 2))),
    }


def main():
    target = RESULTS / "activation_diagnosis.json"
    if target.exists():
        raise FileExistsError(target)
    _pilot, data = pilot_data()
    models = inherited()
    float_runtime, floating = prepare(checked(models["float"]["model"]))
    quant_runtime, quantized = prepare(checked(models["static"]["model"]))
    quant_map = {x["scope"]: x for x in quantized}
    assert {r["scope"] for r in floating} == set(quant_map)
    records, final = [], []
    indices = [0, 1, 8, 15]
    for image_index in indices:
        image = data["evaluation"][0][image_index : image_index + 1].numpy()
        for runtime in [float_runtime, quant_runtime]:
            details = runtime.get_input_details()[0]
            runtime.set_tensor(details["index"], quantize(image, details))
            runtime.invoke()
        for order, row in enumerate(floating):
            qrow = quant_map[row["scope"]]
            a = float_runtime.get_tensor(row["output"]["index"])
            raw = quant_runtime.get_tensor(qrow["output"]["index"])
            b = dequantize(raw, qrow["output"])
            assert a.shape == b.shape
            scale, zero = qrow["output"]["quantization"]
            limits = np.iinfo(raw.dtype)
            low, high = (limits.min - zero) * scale, (limits.max - zero) * scale
            records.append(
                {
                    "image_index": image_index,
                    "order": order,
                    "scope": row["scope"],
                    "weight": row["weight"],
                    "shape": list(a.shape),
                    **error(a, b),
                    "quantization_scale": scale,
                    "representable_min": low,
                    "representable_max": high,
                    "float_outside_quantized_range_fraction": float(
                        np.mean((a < low) | (a > high))
                    ),
                }
            )
            if "block16/conv3/Conv_conv_filter" in row["weight"]:
                assert a.shape[-1] == 16
                final.append(
                    {
                        "image_index": image_index,
                        "center_channel7": error(a[..., 7:8], b[..., 7:8]),
                        "displacement_channels8_11": error(a[..., 8:12], b[..., 8:12]),
                        "float_center_range": [float(a[..., 7].min()), float(a[..., 7].max())],
                        "float_displacement_range": [
                            float(a[..., 8:12].min()),
                            float(a[..., 8:12].max()),
                        ],
                        "float_unused_range": [
                            float(np.concatenate([a[..., :7], a[..., 12:]], axis=-1).min()),
                            float(np.concatenate([a[..., :7], a[..., 12:]], axis=-1).max()),
                        ],
                        "conv_output_quantization_scale": scale,
                    }
                )
        print("ACTIVATION IMAGE", image_index, "done", flush=True)
    summary = []
    for row in floating:
        values = [r for r in records if r["scope"] == row["scope"]]
        summary.append(
            {
                "order": values[0]["order"],
                "scope": row["scope"],
                "weight": row["weight"],
                "mean_relative_l2": float(np.mean([r["relative_l2"] for r in values])),
                "max_relative_l2": max(r["relative_l2"] for r in values),
                "mean_reference_outside_range_fraction": float(
                    np.mean([r["float_outside_quantized_range_fraction"] for r in values])
                ),
                "quantization_scale": values[0]["quantization_scale"],
            }
        )
    write(
        target,
        {
            "finished_at_utc": now(),
            "models": {n: models[n]["model"] for n in ["float", "static"]},
            "diagnostic_image_indices": indices,
            "operators": 41,
            "records": records,
            "summary": summary,
            "active_final_channels": final,
            "protocol": "1thread no default delegates,preserve all tensors,4 fixed development images; match Conv output scope/shape; quantize native input and dequantize each tensor with actual scale. Different execution mode from timing. Errors include accumulation from upstream, not isolated causality. Float outside range is distinct from endpoint occupancy.",
            "data_manifest": data["manifest"],
            "source_code": info(__file__),
        },
    )
    with (RESULTS / "activation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print("FIRST5", summary[:5], flush=True)
    print("FINAL ACTIVE", final, flush=True)


if __name__ == "__main__":
    main()
