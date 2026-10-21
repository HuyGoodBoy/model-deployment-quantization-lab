"""Two declared PTQ experiments with unchanged32-image calibration."""

import gc
import importlib.metadata
import re
from collections import Counter

from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import calibrator, qtyping, quantizer, recipe
from runtime21 import Runner, infer, io_details
from support21 import ART, DAY20, RESULTS, checked, info, inherited, now, pilot_data, read, write


def graph(path):
    runtime = Interpreter(model_path=str(path), num_threads=4)
    runtime.allocate_tensors()
    tensors = {t["index"]: t for t in runtime.get_tensor_details()}
    rows = []
    for op in runtime._get_ops_details():
        if op["op_name"] == "DELEGATE":
            continue
        names = [tensors[int(i)]["name"] for i in op["outputs"] if i >= 0]
        scope = ";".join(names) + (";" if names else "")
        row = {
            "op": op["op_name"],
            "scope": scope,
            "output_dtypes": [str(tensors[int(i)]["dtype"]) for i in op["outputs"] if i >= 0],
        }
        if op["op_name"] in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
            weight = tensors[int(op["inputs"][1])]
            row.update(weight=weight["name"], weight_dtype=str(weight["dtype"]))
        rows.append(row)
    return rows


def calibrate(q, images):
    return q.calibrate(
        {"serving_default": [{"input": x[None]} for x in images]},
        num_threads=4,
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
    )


def main():
    target = RESULTS / "exports.json"
    if target.exists():
        raise FileExistsError(target)
    ART.mkdir(parents=True, exist_ok=True)
    pilot, data = pilot_data()
    inputs = data["evaluation"][0].numpy()
    calibration = data["calibration"][0].numpy()
    old = inherited()
    floating = checked(old["float"]["model"])
    split = checked(read(RESULTS / "split_conversion.json")["tflite"])
    original_runner, split_runner = Runner(floating), Runner(split)
    original_output = infer(original_runner, inputs)
    split_output = infer(split_runner, inputs)
    parity = {
        "all": pilot.numerical(original_output, split_output),
        "center": pilot.numerical(original_output[..., :1], split_output[..., :1]),
        "displacement": pilot.numerical(original_output[..., 1:], split_output[..., 1:]),
    }
    original_ap = pilot.line_metrics(original_output, data["ground_truth"])
    split_ap = pilot.line_metrics(split_output, data["ground_truth"])
    assert all(p["relative_l2"] < 1e-4 for p in parity.values()), parity
    assert abs(split_ap["sAP10_subset_percent"] - original_ap["sAP10_subset_percent"]) < 0.05
    write(
        RESULTS / "split_float_parity.json",
        {
            "finished_at_utc": now(),
            "original": info(floating),
            "split": info(split),
            "images": 16,
            "parity": parity,
            "original_sAP10": original_ap,
            "split_sAP10": split_ap,
            "native_IO": io_details(split_runner),
            "gate_passed": True,
        },
    )
    del original_runner, split_runner, original_output, split_output
    gc.collect()
    models = {
        "float": {"model": old["float"]["model"], "policy": "unchanged FP32 control"},
        "static": {"model": old["static"]["model"], "policy": "unchanged day20 AEQ static control"},
        "split_float": {"model": info(split), "policy": "two-output FP32 rewrite control"},
    }
    for name, path in [("head_float", floating), ("split_int8", split)]:
        q = quantizer.Quantizer(path, recipe.static_wi8_ai8())
        source_graph = graph(path)
        excluded = []
        if name == "head_float":
            for row in source_graph:
                backbone = "/backbone/" in row.get("weight", row["scope"])
                if not backbone:
                    assert row["scope"]
                    excluded.append(row["scope"])
                    q.update_quantization_recipe(
                        "^" + re.escape(row["scope"]) + "$",
                        qtyping.TFLOperationName.ALL_SUPPORTED,
                        algorithm_key="no_quantize",
                    )
        statistics = calibrate(q, calibration)
        result = q.quantize(calibration_result=statistics, enable_progress_bar=False)
        output = ART / (name + ".tflite")
        result.export_model(str(output), overwrite=False)
        actual = graph(output)
        convs = [r for r in actual if "weight" in r]
        if name == "head_float":
            assert len(convs) == 41
            for row in convs:
                dtype = "int8" if "/backbone/" in row["weight"] else "float32"
                assert dtype in row["weight_dtype"] and all(
                    dtype in x for x in row["output_dtypes"]
                ), row
            assert sum("float32" in r["weight_dtype"] for r in convs) == 11
            assert sum("int8" in r["weight_dtype"] for r in convs) == 30
        else:
            assert len(convs) == 42
            assert all("int8" in r["weight_dtype"] for r in convs)
        runner = Runner(output)
        native = io_details(runner)
        if name == "head_float":
            assert "float32" in native["outputs"]["output"]["dtype"]
        else:
            assert native["outputs"]["center"]["scale"] > 0
            assert native["outputs"]["displacement"]["scale"] > 0
            assert (
                native["outputs"]["center"]["scale"] != native["outputs"]["displacement"]["scale"]
            )
        models[name] = {
            "model": info(output),
            "graph": actual,
            "native_IO": native,
            "float_scopes": excluded,
            "conv_weight_dtypes": dict(Counter(r["weight_dtype"] for r in convs)),
            "calibration_images": len(calibration),
        }
        print("EXPORTED", name, models[name]["conv_weight_dtypes"], native, flush=True)
        del runner, q, result, statistics
        gc.collect()
    write(
        target,
        {
            "finished_at_utc": now(),
            "models": models,
            "data_manifest": data["manifest"],
            "calibration_images": len(calibration),
            "development_images": len(inputs),
            "calibration_sha256": __import__("hashlib").sha256(calibration.tobytes()).hexdigest(),
            "development_input_sha256": __import__("hashlib").sha256(inputs.tobytes()).hexdigest(),
            "source_control": info(DAY20 / "results/aeq_exports.json"),
            "scope": "New PTQ recipes, same weights/data. A30backboneINT8+11headfloat; Bsplit active output filters before PTQ with independent native outputs. Model adapter concatenates only after dequantizing. No holdout/regression used to build.",
            "environment": {
                p: importlib.metadata.version(p) for p in ["ai-edge-litert", "ai-edge-quantizer"]
            },
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
