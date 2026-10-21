"""Recompute day21 metrics and prove graph changes and inherited source integrity."""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-21"
MLSD = DAY / "experiments/mlsd/results"
ASR = DAY / "experiments/zipformer/results"
sys.path.insert(0, str(DAY / "experiments/mlsd/src"))

# isort: split
from support21 import checked, pilot_data, read, sha


def equal(a, b):
    if isinstance(b, dict):
        assert set(a) == set(b)
        for key in b:
            equal(a[key], b[key])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def stats(row, key="samples_ms", median="median_ms", p95="p95_ms"):
    values = row[key]
    assert len(values) == row["runs"]
    equal(float(np.median(values)), row[median])
    equal(float(np.percentile(values, 95)), row[p95])


def load(record):
    with np.load(checked(record)) as data:
        values = data["output"].copy()
    assert np.isfinite(values).all()
    return values


def verify_mlsd():
    pilot, data = pilot_data()
    exports = read(MLSD / "exports.json")
    dev = read(MLSD / "development.json")
    reg = read(MLSD / "regression.json")
    manifest = read(checked(reg["manifest"]))
    truth = read(checked(manifest["ground_truth"]))
    checked(manifest["inputs"])
    assert (
        hashlib.sha256(data["calibration"][0].numpy().tobytes()).hexdigest()
        == exports["calibration_sha256"]
    )
    assert (
        hashlib.sha256(data["evaluation"][0].numpy().tobytes()).hexdigest()
        == exports["development_input_sha256"]
    )
    for result, ground, count in [(dev, data["ground_truth"], 16), (reg, truth, 54)]:
        assert result["images"] == count
        for item in result["models"].values():
            values = load(item["output"])
            assert values.shape == (count, 256, 256, 5)
            equal(pilot.line_metrics(values, ground), item["sAP10"])
    original = load(dev["models"]["float"]["output"])
    split = load(dev["models"]["split_float"]["output"])
    assert np.array_equal(original, split)
    assert read(MLSD / "split_float_parity.json")["gate_passed"]
    for name, count, integer, floating in [("head_float", 41, 30, 11), ("split_int8", 42, 42, 0)]:
        runtime = Interpreter(
            model_path=str(checked(exports["models"][name]["model"])), num_threads=4
        )
        runtime.allocate_tensors()
        tensors = {t["index"]: t for t in runtime.get_tensor_details()}
        convs = [
            o
            for o in runtime._get_ops_details()
            if o["op_name"] in ["CONV_2D", "DEPTHWISE_CONV_2D"]
        ]
        assert len(convs) == count
        pairs = [(tensors[int(o["inputs"][1])], tensors[int(o["outputs"][0])]) for o in convs]
        assert sum(w["dtype"] == y["dtype"] == np.int8 for w, y in pairs) == integer
        assert sum(w["dtype"] == y["dtype"] == np.float32 for w, y in pairs) == floating
        signature = runtime.get_signature_runner("serving_default")
        out = signature.get_output_details()
        if name == "split_int8":
            assert set(out) == {"center", "displacement"}
            equal(
                out["center"]["quantization"][0],
                exports["models"][name]["native_IO"]["outputs"]["center"]["scale"],
            )
            equal(
                out["displacement"]["quantization"][0],
                exports["models"][name]["native_IO"]["outputs"]["displacement"]["scale"],
            )
            assert out["center"]["quantization"][0] < out["displacement"]["quantization"][0]
    for item in exports["models"].values():
        checked(item["model"])
    bench = read(MLSD / "benchmark.json")
    for row in bench["models"].values():
        stats(row)
        assert row["runs"] == 80
    quality_eligible = [
        n
        for n in ["head_float", "split_int8"]
        if dev["models"][n]["sAP10"]["sAP10_subset_percent"]
        >= dev["models"]["float"]["sAP10"]["sAP10_subset_percent"] - 0.5
    ]
    assert quality_eligible == dev["quality_eligible"] == ["head_float"]
    selection = read(MLSD / "selection.json")
    checked(selection["development"])
    checked(selection["benchmark"])
    checked(reg["selection"])
    assert selection["selected"] == reg["selected_before_regression"] == "head_float"
    assert datetime.fromisoformat(selection["locked_at_utc"]) < datetime.fromisoformat(
        reg["started_at_utc"]
    )
    assert not selection["regression_used"]
    assert bench["models"]["head_float"]["median_ms"] < bench["models"]["float"]["median_ms"]
    bench8 = read(MLSD / "benchmark8.json")
    for name, row in bench8["models"].items():
        stats(row)
        values = load(row["output"])
        equal(pilot.line_metrics(values, data["ground_truth"]), row["sAP10"])
        assert (
            np.array_equal(values, load(dev["models"][name]["output"]))
            == row["thread4_vs8_output_exact"]
        )
    assert bench8["models"]["head_float"]["median_ms"] > bench8["models"]["float"]["median_ms"]
    spec = importlib.util.spec_from_file_location(
        "timing_check", DAY / "experiments/mlsd/src/analyze_timing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis = read(MLSD / "timing_analysis.json")
    equal(
        module.paired(
            bench["models"]["float"]["samples_ms"],
            bench["models"]["head_float"]["samples_ms"],
            20261021,
        ),
        analysis["head_float_vs_float_threads4"],
    )
    equal(
        module.paired(
            bench8["models"]["float"]["samples_ms"],
            bench8["models"]["head_float"]["samples_ms"],
            20261022,
        ),
        analysis["head_float_vs_float_threads8"],
    )
    diagnostic = read(MLSD / "activation_diagnosis.json")
    assert len(diagnostic["records"]) == 4 * 41
    for summary in diagnostic["summary"]:
        rows = [r for r in diagnostic["records"] if r["scope"] == summary["scope"]]
        assert len(rows) == 4
        equal(float(np.mean([r["relative_l2"] for r in rows])), summary["mean_relative_l2"])
    return {
        "development5_and_regression3_AP_recomputed": True,
        "split_float_output_exact": True,
        "head_float30INT8_11FP32_verified": True,
        "split42INT8_independent_output_scales_verified": True,
        "same32calibration16development_input_hashes": True,
        "selection_precedes_regression": True,
        "timing8threads_and_uncertainty_recomputed": True,
    }


def distance(reference, hypothesis):
    import re

    a, b = (re.sub(r"[^A-Z' ]", " ", text.upper()).split() for text in [reference, hypothesis])
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(a)


def check_wer(result):
    errors, words = 0, 0
    for row in result["records"]:
        e, n = distance(row["reference"], row["text"])
        assert e == row["word_errors"]["errors"] and n == row["word_errors"]["N"]
        errors += e
        words += n
    assert errors == result["word_errors"]["errors"] and words == result["word_errors"]["N"]
    equal(errors / words * 100, result["word_errors"]["WER_percent"])
    return errors, words


def verify_asr():
    rewrite = read(ASR / "encoder_fc_rewrite.json")
    old = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(checked(rewrite["original"]).read_bytes(), 0)
    )
    new = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(checked(rewrite["model"]).read_bytes(), 0)
    )
    a, b = old.subgraphs[0], new.subgraphs[0]
    assert np.array_equal(a.inputs, b.inputs) and np.array_equal(a.outputs, b.outputs)
    assert len(a.operators) == len(b.operators)
    changed = 0
    for original, candidate in zip(a.operators, b.operators, strict=True):
        op1 = old.operatorCodes[original.opcodeIndex].builtinCode
        op2 = new.operatorCodes[candidate.opcodeIndex].builtinCode
        if op1 != op2:
            assert (
                op1 == schema.BuiltinOperator.BATCH_MATMUL
                and op2 == schema.BuiltinOperator.FULLY_CONNECTED
            )
            assert candidate.builtinOptions.keepNumDims and int(candidate.inputs[-1]) == -1
            t1, t2 = a.tensors[int(original.inputs[1])], b.tensors[int(candidate.inputs[1])]
            w1 = np.frombuffer(old.buffers[t1.buffer].data, dtype=np.float32).reshape(t1.shape)
            w2 = np.frombuffer(new.buffers[t2.buffer].data, dtype=np.float32).reshape(t2.shape)
            assert np.array_equal(w1.T, w2)
            changed += 1
    assert changed == 112
    opcounts = Counter(new.operatorCodes[o.opcodeIndex].builtinCode for o in b.operators)
    assert opcounts[schema.BuiltinOperator.BATCH_MATMUL] == 40
    assert opcounts[schema.BuiltinOperator.FULLY_CONNECTED] == 112
    for fixture in rewrite["parity_on15_state_fixtures"]:
        assert len(fixture) == 36 and all(r["allclose_rtol1e_4_atol1e_4"] for r in fixture.values())
        assert all(r["exact"] for n, r in fixture.items() if n.startswith("new_cached_len_"))
    evaluations = {n: read(ASR / ("evaluation_" + n + ".json")) for n in ["baseline", "fc"]}
    for result in evaluations.values():
        assert check_wer(result) == (91, 1150)
        assert len(result["records"]) == 73
        assert all(
            r["original_tokens_equal"] and r["original_final_counters_equal"]
            for r in result["records"]
        )
        for model in result["models"].values():
            checked(model)
    continuous = read(ASR / "continuous_state_test.json")
    checked(continuous["input_audio"])
    assert continuous["tokens_equal"] and continuous["counters_exact"]
    assert continuous["encoder_parity"]["allclose_rtol1e_4_atol1e_4"]
    bench = read(ASR / "benchmark.json")
    for row in bench["models"].values():
        stats(row["encoder_ms"], "samples", "median", "p95")
        stats(row["pipeline_RTF"], "samples", "median", "p95")
        assert row["encoder_ms"]["runs"] == 160 and row["pipeline_RTF"]["runs"] == 3
        for r, rtf in zip(row["pipeline_rounds"], row["pipeline_RTF"]["samples"], strict=True):
            equal(r["pipeline_seconds"] / r["audio_seconds"], rtf)
    multi = read(ASR / "multispeaker_evaluation.json")
    manifest = read(checked(multi["manifest"]))
    for record in manifest["records"]:
        checked(record)
    assert (
        manifest["utterances"] == 100 and len({r["speaker_id"] for r in manifest["records"]}) == 20
    )
    assert datetime.fromisoformat(manifest["locked_at_utc"]) < datetime.fromisoformat(
        multi["started_at_utc"]
    )
    assert (
        manifest["no_overlap_with_previous73_ids"]
        and manifest["no_overlap_with_previous73_speakers"]
    )
    old_manifest = read(
        ROOT / "daily/2026-10-20/experiments/zipformer/results/evaluation_manifest.json"
    )
    assert not {r["id"] for r in manifest["records"]} & {r["id"] for r in old_manifest["records"]}
    assert not {r["speaker_id"] for r in manifest["records"]} & {
        r["speaker_id"] for r in old_manifest["records"]
    }
    assert set(Counter(r["speaker_id"] for r in manifest["records"]).values()) == {5}
    assert datetime.fromisoformat(rewrite["finished_at_utc"]) < datetime.fromisoformat(
        manifest["locked_at_utc"]
    )
    checked(manifest["archive"])
    for row in multi["models"].values():
        assert len(row["records"]) == 100
        for expected, actual in zip(manifest["records"], row["records"], strict=True):
            assert expected["id"] == actual["id"] and expected["text"] == actual["reference"]
        check_wer(row)
    for x, y in zip(
        multi["models"]["baseline"]["records"], multi["models"]["fc"]["records"], strict=True
    ):
        assert (
            x["id"] == y["id"]
            and x["tokens"] == y["tokens"]
            and x["final_cached_len"] == y["final_cached_len"]
        )
    return {
        "112weights_transposed_exactly_other_operator_types_unchanged": True,
        "public_inputs_outputs_preserved": True,
        "15state_fixtures36outputs_passed": True,
        "73utterance_WER_recomputed": True,
        "100utterance20speaker_WER_recomputed": True,
        "continuous79_5seconds_tokens_and_counters_match": True,
        "matched_encoder_and_RTF_samples_recomputed": True,
    }


def main():
    target = DAY / "VERIFICATION.json"
    assert not target.exists()
    previous = read(ROOT / "daily/2026-10-20/SNAPSHOT.json")
    count = 0
    for item in previous["files"]:
        if item["path"] != "README.md":
            checked(item)
            count += 1
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "previous_day_files_unchanged": count,
        "mlsd": verify_mlsd(),
        "zipformer": verify_asr(),
        "scope": "Evidence integrity/metric recomputation and graph rewrite checks;not a deployment certification. Corpus quality subsets and hardware/timing limits remain.",
        "source_sha256": sha(__file__),
    }
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
