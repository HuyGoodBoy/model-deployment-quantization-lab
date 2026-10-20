"""New PTQ from the same QAT-weight float model; no reuse of ONNX QAT scales."""

import importlib.metadata
import re
from collections import Counter

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import calibrator, qtyping, quantizer, recipe
from mlsd_support import ART, RESULTS, info, model_paths, now, pilot, read, session, write


def graph(path):
    runtime = Interpreter(model_path=str(path), num_threads=4)
    runtime.allocate_tensors()
    tensors = {t["index"]: t for t in runtime.get_tensor_details()}
    rows = []
    for op in runtime._get_ops_details():
        if op["op_name"] not in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
            continue
        weight, output = (
            tensors[int(op[key][idx])] for key, idx in [("inputs", 1), ("outputs", 0)]
        )
        rows.append(
            {
                "op": op["op_name"],
                "weight": weight["name"],
                "scope": output["name"],
                "weight_dtype": str(weight["dtype"]),
                "output_dtype": str(output["dtype"]),
            }
        )
    return rows


def main():
    target = RESULTS / "aeq_exports.json"
    if target.exists():
        raise FileExistsError(target)
    path = ART / "float_conversion/qat_weights_fp32_float32.tflite"
    data = pilot.load_datasets("mlsd")
    inputs = data["evaluation"][0].numpy()
    runtime = Interpreter(model_path=str(path), num_threads=4)
    runtime.allocate_tensors()
    function = runtime.get_signature_runner("serving_default")
    original = session(model_paths()["qat_weights_fp32"])
    a, b = [], []
    for image in inputs:
        a.append(original.run(None, {"input": image[None]})[0])
        b.append(function(input=image[None])["output"])
    a, b = np.concatenate(a), np.concatenate(b)
    assert a.shape == b.shape == (16, 5, 256, 256)
    parity = pilot.numerical(a, b)
    assert parity["relative_l2"] < 0.02, parity
    del runtime, function, original
    scopes = graph(path)
    chosen = read(RESULTS / "ort_profile_comparison.json")["float_candidate_layers"]
    # ONNX weight identities survive conversion in the LiteRT filter tensor name.
    mapping = {
        "model.model.backbone.features.0.0": "features.0/features.0.0/Conv_conv_filter",
        "model.model.backbone.features.1.conv.0.0": "features.1/conv/conv.0/conv.0.0/Conv_depthwise_filter",
        "model.model.backbone.features.3.conv.1.0": "features.3/conv/conv.1/conv.1.0/Conv_depthwise_filter",
    }
    assert set(chosen) == set(mapping), chosen
    selected = []
    for layer, fragment in mapping.items():
        matches = [row for row in scopes if fragment in row["weight"]]
        assert len(matches) == 1, (layer, matches)
        selected.append({"onnx_layer": layer, **matches[0]})
    calibration = [{"input": image[None]} for image in data["calibration"][0].numpy()]
    calibrating = quantizer.Quantizer(path, recipe.static_wi8_ai8())
    statistics = calibrating.calibrate(
        {"serving_default": calibration},
        num_threads=4,
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
    )
    models = {"float": {"model": info(path), "graph": scopes}}
    for name in ["static", "selective", "dynamic"]:
        q = quantizer.Quantizer(
            path, recipe.dynamic_wi8_afp32() if name == "dynamic" else recipe.static_wi8_ai8()
        )
        if name == "selective":
            for row in selected:
                q.update_quantization_recipe(
                    "^" + re.escape(row["scope"]) + ";$",
                    qtyping.TFLOperationName(row["op"]),
                    qtyping.OpQuantizationConfig(),
                    algorithm_key="no_quantize",
                )
        output = ART / ("aeq_" + name + ".tflite")
        result = q.quantize(
            calibration_result=None if name == "dynamic" else statistics, enable_progress_bar=False
        )
        result.export_model(str(output), overwrite=False)
        rows = graph(output)
        checks = []
        for selected_row in selected:
            matches = [r for r in rows if r["scope"] == selected_row["scope"]]
            assert len(matches) == 1, (name, selected_row, matches)
            checks.append(matches[0])
        if name == "selective":
            assert all(
                "float32" in r["weight_dtype"] and "float32" in r["output_dtype"] for r in checks
            )
        if name == "static":
            assert all("int8" in r["weight_dtype"] and "int8" in r["output_dtype"] for r in checks)
        models[name] = {
            "model": info(output),
            "graph": rows,
            "selected_scopes_checked": checks,
            "conv_weight_dtypes": dict(Counter(r["weight_dtype"] for r in rows)),
        }
        print("EXPORTED", name, models[name]["conv_weight_dtypes"], flush=True)
    write(
        target,
        {
            "finished_at_utc": now(),
            "models": models,
            "selected": selected,
            "float_onnx_parity": parity,
            "calibration_images": len(calibration),
            "calibration_mode": "CALIBRATION_PROFILER_BASED",
            "development_images": len(inputs),
            "scope": "Fresh LiteRT PTQ from same QAT weights, not editing/preserving ONNX QDQ scales. ORT-selected scopes are hypotheses for LiteRT; speed must be measured independently.",
            "environment": {
                p: importlib.metadata.version(p)
                for p in ["ai-edge-litert", "ai-edge-quantizer", "onnxruntime"]
            },
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
