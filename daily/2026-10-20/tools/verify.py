"""Recompute reported metrics and guards from immutable models, outputs and samples."""

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-20"
ASR = DAY / "experiments/zipformer/results"
MLSD = DAY / "experiments/mlsd/results"
sys.path.insert(0, str(DAY / "experiments/mlsd/src"))

# isort: split
from mlsd_support import pilot
from profile_onnx import profile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(item):
    path = ROOT / item["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item["sha256"], path
    if "bytes" in item:
        assert path.stat().st_size == item["bytes"]
    return path


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (actual, expected)
    else:
        assert actual == expected


def edit_distance(reference, hypothesis):
    # Independent scalar Levenshtein implementation; S/D/I tie choices may vary.
    a, b = (re.sub(r"[^A-Z' ]", " ", text.upper()).split() for text in [reference, hypothesis])
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(a)


def sample_stats(row, key="samples_ms", median="median_ms", p95="p95_ms"):
    values = row[key]
    equal(float(np.median(values)), row[median])
    equal(float(np.percentile(values, 95)), row[p95])
    assert len(values) == row["runs"]


def verify_asr():
    manifest = read(ASR / "source_manifest.json")
    for item in manifest["files"]:
        checked(item)
        if item["upstream_lfs_sha256"]:
            assert item["sha256"] == item["upstream_lfs_sha256"]
    evaluation_manifest = read(ASR / "evaluation_manifest.json")
    for item in evaluation_manifest["records"]:
        checked(item)
    evaluations = {v: read(ASR / f"evaluation_{v}.json") for v in ["onnx", "fixed", "optimized"]}
    for name, result in evaluations.items():
        for model in result["models"].values():
            checked(model)
        checked(result["manifest"])
        errors, words = 0, 0
        assert result["utterances"] == 73
        for record in result["records"]:
            e, n = edit_distance(record["reference"], record["text"])
            assert (e, n) == (record["word_errors"]["errors"], record["word_errors"]["N"])
            errors += e
            words += n
        assert (errors, words) == (91, 1150)
        equal(errors / words * 100, result["word_errors"]["WER_percent"])
        assert all(r["original_token_agreement"] for r in result["smoke"])
        if name != "onnx":
            for a, b in zip(evaluations["onnx"]["records"], result["records"], strict=True):
                assert a["id"] == b["id"] and a["tokens"] == b["tokens"]
                assert a["final_cached_len"] == b["final_cached_len"]
    bench = read(ASR / "matched_benchmark_v2.json")
    assert len(bench["captured_encoder_state_parity"]) == 15
    for fixture in bench["captured_encoder_state_parity"]:
        assert len(fixture) == 36
        assert all(row["allclose_rtol1e_4_atol1e_4"] for row in fixture.values())
        assert all(
            row["exact"] for name, row in fixture.items() if name.startswith("new_cached_len_")
        )
    for result in bench["models"].values():
        for row in result["components_ms"].values():
            sample_stats(row, "samples", "median", "p95")
            assert row["runs"] == 160
        sample_stats(result["pipeline_RTF"], "samples", "median", "p95")
        assert result["pipeline_RTF"]["runs"] == 3
        for row, rtf in zip(
            result["pipeline_rounds"], result["pipeline_RTF"]["samples"], strict=True
        ):
            equal(row["pipeline_seconds"] / row["audio_seconds"], rtf)
    for variant in ["raw", "static"]:
        validation = read(ASR / f"runtime_validation_{variant}.json")["models"]
        assert validation["decoder"]["status"] == "runtime_failed"
        assert validation["joiner"]["status"] == "allocated_and_invoked"
    assert (
        read(ASR / "runtime_validation_raw.json")["models"]["encoder"]["status"] == "runtime_failed"
    )
    assert (
        read(ASR / "runtime_validation_static.json")["models"]["encoder"]["status"]
        == "allocated_and_invoked"
    )
    repair = read(ASR / "decoder_conv2d_repair.json")
    lookup = read(ASR / "decoder_lookup_optimization.json")
    for result in [repair, lookup]:
        checked(result["tflite"])
        assert all(r["allclose_rtol1e_4_atol1e_4"] for r in result["lite_parity"])
    assert "CONV_2D" not in lookup["operators"]
    claude = read(ASR / "claude_execution.json")
    assert not claude["auth_status"]["loggedIn"] and claude["returncode"] == 1
    response = json.loads(claude["stdout"])
    assert response["is_error"] and not response["modelUsage"]
    assert response["usage"]["input_tokens"] == response["usage"]["output_tokens"] == 0
    return {
        "WER_percent": 91 / 1150 * 100,
        "all73tokens_and_final_counters_match": True,
        "state_fixtures": 15,
        "state_outputs_each": 36,
        "claude_inference_executed": False,
    }


def verify_mlsd():
    previous = read(ROOT / "daily/2026-10-19/experiments/selective/results/exports.json")
    for name in ["qat_weights_fp32", "qdq_all"]:
        checked(previous["models"][name]["model"])
    reproduction = read(MLSD / "float_conversion_reproduction.json")
    checked(reproduction["source_unchanged"])
    checked(reproduction["model"])
    assert (
        reproduction["model"]["sha256"]
        == reproduction["byte_identical_to_evaluated_model"]["sha256"]
    )
    data = pilot.load_datasets("mlsd")
    profiling = read(MLSD / "ort_profile_comparison.json")
    for name, row in profiling["profiles"].items():
        checked(row["trace"])
        checked(row["optimized"])
        rebuilt = profile(name, checked(row["model"]), data["evaluation"][0].numpy())
        equal(rebuilt, row)
        optimized = onnx.load(checked(row["optimized"]))
        counts = dict(Counter(n.op_type for n in optimized.graph.node))
        assert counts == row["unique_operators"]
        assert len(row["convolutions"]) == 41
        if name == "qdq_all":
            assert counts["QLinearConv"] == 22
            assert sum(n["op"] in ["Conv", "FusedConv"] for n in row["nodes"]) == 19
            initializers = {x.name: x for x in optimized.graph.initializer}
            for node in row["nodes"]:
                assert node["calls"] == 8
                if node["op"] == "QLinearConv":
                    matches = [
                        n for n in optimized.graph.node if n.name + "_kernel_time" == node["name"]
                    ]
                    assert len(matches) == 1
                    assert initializers[matches[0].input[3]].data_type in [
                        onnx.TensorProto.INT8,
                        onnx.TensorProto.UINT8,
                    ]
                    assert any(t in node["input_type_shape"][0] for t in ["int8", "uint8"])
    a = profiling["profiles"]["qat_weights_fp32"]["convolutions"]
    b = profiling["profiles"]["qdq_all"]["convolutions"]
    for row in profiling["comparison"]:
        equal(a[row["layer"]]["mean_ms"], row["fp32_mean_ms"])
        equal(b[row["layer"]]["mean_ms"], row["qdq_mean_ms"])
        equal(row["qdq_mean_ms"] - row["fp32_mean_ms"], row["extra_ms"])
        equal(row["qdq_mean_ms"] / row["fp32_mean_ms"], row["ratio"])
    exports = read(MLSD / "aeq_exports.json")
    dev = read(MLSD / "aeq_development.json")
    regression = read(MLSD / "aeq_regression.json")
    manifest = read(checked(regression["manifest"]))
    truth = read(checked(manifest["ground_truth"]))
    checked(manifest["inputs"])
    for result, ground, count in [(dev, data["ground_truth"], 16), (regression, truth, 54)]:
        assert result["images"] == count
        for item in result["models"].values():
            outputs = np.load(checked(item["output"]))
            assert outputs.shape == (count, 256, 256, 5) and np.isfinite(outputs).all()
            equal(pilot.line_metrics(outputs, ground), item["sAP10"])
    for name, item in exports["models"].items():
        runtime = Interpreter(model_path=str(checked(item["model"])), num_threads=4)
        runtime.allocate_tensors()
        tensors = {t["index"]: t for t in runtime.get_tensor_details()}
        rows = []
        for op in runtime._get_ops_details():
            if op["op_name"] in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
                weight = tensors[int(op["inputs"][1])]
                output = tensors[int(op["outputs"][0])]
                rows.append((output["name"], weight["dtype"], output["dtype"]))
        assert len(rows) == 41
        if name == "selective":
            assert sum(w == np.float32 for _, w, _ in rows) == 3
            assert sum(w == o == np.int8 for _, w, o in rows) == 38
            for selected in exports["selected"]:
                matches = [r for r in rows if r[0] == selected["scope"]]
                assert len(matches) == 1 and matches[0][1] == matches[0][2] == np.float32
        if name == "static":
            assert all(w == o == np.int8 for _, w, o in rows)
    bench = read(MLSD / "aeq_benchmark.json")
    for row in bench["models"].values():
        sample_stats(row)
        assert row["runs"] == 80
    selection = read(MLSD / "aeq_selection.json")
    checked(selection["development"])
    checked(selection["benchmark"])
    checked(regression["selection"])
    assert not selection["regression_used"]
    assert datetime.fromisoformat(selection["locked_at_utc"]) < datetime.fromisoformat(
        regression["started_at_utc"]
    )
    assert selection["selected"] == regression["selected_before_regression"] == "static"
    actual_eligible = [
        n
        for n in ["selective", "dynamic"]
        if dev["models"][n]["sAP10"]["sAP10_subset_percent"]
        >= dev["models"]["static"]["sAP10"]["sAP10_subset_percent"] - 0.5
        and bench["models"][n]["median_ms"] < bench["models"]["static"]["median_ms"]
    ]
    assert actual_eligible == selection["eligible"] == []
    diagnosis = read(MLSD / "aeq_output_diagnosis.json")
    floating = np.load(checked(dev["models"]["float"]["output"]))
    for row in diagnosis["models"].values():
        scale, zero = row["output_scale"], row["output_zero_point"]
        restored = (np.clip(np.rint(floating / scale) + zero, -128, 127) - zero) * scale
        equal(
            pilot.line_metrics(restored, data["ground_truth"]),
            row["sAP10_only_output_roundtrip_on_float"],
        )
    exploration = read(MLSD / "aeq_float_output_development.json")
    checked(exploration["model"])
    equal(
        pilot.line_metrics(np.load(checked(exploration["output"])), data["ground_truth"]),
        exploration["sAP10"],
    )
    assert exploration["development_only"]
    tuning = read(MLSD / "float_thread_tuning.json")
    for row in tuning["conditions"].values():
        sample_stats(row)
        assert row["runs"] == 80
    winner = str(tuning["selected_num_threads"])
    assert winner == min(tuning["conditions"], key=lambda n: tuning["conditions"][n]["median_ms"])
    equal(
        pilot.line_metrics(np.load(checked(tuning["selected_output"])), data["ground_truth"]),
        tuning["selected_sAP10"],
    )
    assert np.array_equal(np.load(checked(tuning["selected_output"])), floating)
    return {
        "all8_development_and_regression_metrics_recomputed": True,
        "AEQ_selective3float38int8_verified": True,
        "original_source_SHA256_restored": True,
        "selection_before_regression": True,
        "thread_tuning_selected": int(winner),
        "ORT_graph_and_raw_trace_recomputed": True,
        "thread_tuning_output_exact": True,
    }


def main():
    target = DAY / "VERIFICATION_final.json"
    assert not target.exists()
    results = {"asr": verify_asr(), "mlsd": verify_mlsd()}
    assert read(DAY / "tools/schedule_guard_verification.json")["passed"]
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "scope": "Evidence integrity and recomputation pass; not a claim that all quantized models passed quality/release gates or that Claude executed.",
        **results,
    }
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
