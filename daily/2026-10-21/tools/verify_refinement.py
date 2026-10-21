"""Independent metric/raw graph/chronology checks for the approved refinement."""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-21"
MLSD = DAY / "experiments/mlsd/results"
ASR = DAY / "experiments/zipformer/results"
sys.path.insert(0, str(DAY / "experiments/mlsd/src"))

from support21 import checked, info, pilot_data, read, write


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


initial = module("verify_initial", DAY / "tools/verify.py")
analysis = module("analyze_refinement", DAY / "tools/analyze_refinement.py")


def pack(obj):
    builder = flatbuffers.Builder(0)
    builder.Finish(obj.Pack(builder))
    return bytes(builder.Output())


def graph(record):
    return s.ModelT.InitFromObj(s.Model.GetRootAsModel(checked(record).read_bytes(), 0))


def verify_mlsd():
    pilot, data = pilot_data()
    prior = read(MLSD / "development.json")
    exports = read(MLSD / "exports.json")
    baseline = prior["models"]["float"]["sAP10"]["sAP10_subset_percent"]
    control = read(MLSD / "refinement_prune_parity.json")
    assert np.array_equal(
        initial.load(control["output"]), initial.load(prior["models"]["float"]["output"])
    )
    assert control["output_exact"]
    eligible = []
    for name, integer, floating in [("C1", 30, 11), ("C2", 40, 1), ("C3", 38, 3)]:
        row = read(MLSD / ("refinement_" + name + ".json"))
        output = initial.load(row["development_output"])
        initial.equal(pilot.line_metrics(output, data["ground_truth"]), row["sAP10"])
        initial.equal(baseline - row["sAP10"]["sAP10_subset_percent"], row["loss_pp"])
        assert row["quality_passed"] == (row["loss_pp"] <= 0.5)
        if row["quality_passed"]:
            eligible.append(name)
        assert row["calibration_sha256"] == exports["calibration_sha256"]
        assert row["development_input_sha256"] == exports["development_input_sha256"]
        runtime = Interpreter(model_path=str(checked(row["model"])), num_threads=4)
        runtime.allocate_tensors()
        tensors = {int(t["index"]): t for t in runtime.get_tensor_details()}
        convs = [
            op
            for op in runtime._get_ops_details()
            if op["op_name"] in ["CONV_2D", "DEPTHWISE_CONV_2D"]
        ]
        assert len(convs) == 41
        pairs = [
            (tensors[int(op["inputs"][1])]["dtype"], tensors[int(op["outputs"][0])]["dtype"])
            for op in convs
        ]
        assert pairs.count((np.int8, np.int8)) == integer
        assert pairs.count((np.float32, np.float32)) == floating
    assert eligible == ["C1"]
    quality = read(MLSD / "refinement_quality_selection.json")
    assert quality["quality_eligible"] == eligible
    assert quality["failed_skipped_full_benchmark_regression"] == ["C2", "C3"]
    timings = read(DAY / "REFINEMENT_ANALYSIS.json")["mlsd"]
    for threads in [4, 8]:
        for session in [1, 2]:
            bench = read(MLSD / f"refinement_contiguous_benchmark_t{threads}_s{session}.json")
            assert datetime.fromisoformat(quality["locked_at_utc"]) < datetime.fromisoformat(
                bench["started_at_utc"]
            )
            assert set(bench["models"]) == {"float", "head_float", "C1"}
            for name, row in bench["models"].items():
                initial.stats(row)
                assert row["runs"] == 80 and row["vs_development_exact"]
                expected = (
                    prior["models"][name]["sAP10"]
                    if name in prior["models"]
                    else read(MLSD / "refinement_C1.json")["sAP10"]
                )
                initial.equal(row["sAP10"], expected)
                if name != "float":
                    derived = analysis.paired(
                        bench["models"]["float"]["samples_ms"],
                        row["samples_ms"],
                        16,
                        20261021 + threads * 10 + session,
                    )
                    initial.equal(
                        derived, timings[f"threads{threads}_session{session}"][name + "_vs_float"]
                    )
    selection = read(MLSD / "refinement_speed_selection.json")
    regression = read(MLSD / "refinement_regression.json")
    assert datetime.fromisoformat(selection["locked_at_utc"]) < datetime.fromisoformat(
        regression["started_at_utc"]
    )
    assert selection["replacement_proposal"] == "C1"
    assert set(regression["models"]) == {"C1"}
    manifest = read(checked(regression["manifest"]))
    truth = read(checked(manifest["ground_truth"]))
    for row in regression["models"].values():
        initial.equal(pilot.line_metrics(initial.load(row["output"]), truth), row["sAP10"])
    profile = read(MLSD / "refinement_io_profile.json")
    for row in profile["models"].values():
        assert row["signature_vs_staged_output_exact"]
        for sample in row["samples_ms"]:
            initial.equal(sum(v for k, v in sample.items() if k != "total"), sample["total"])
        for key, median in row["medians_ms"].items():
            initial.equal(float(np.median([r[key] for r in row["samples_ms"]])), median)
    return {
        "all3_AP_precision_data_hashes_recomputed": True,
        "prune_float_output_exact": True,
        "C2_C3_failed_gate_not_relaxed": True,
        "four_contiguous_benchmarks_and_uncertainty_recomputed": True,
        "selection_precedes54image_regression": True,
        "staged_profile_output_parity_and_per_sample_total_checked": True,
    }


def verify_asr():
    rewrite = read(ASR / "refinement_bias_rewrite.json")
    before, after = graph(rewrite["original"]), graph(rewrite["model"])
    assert len(before.subgraphs) == len(after.subgraphs) == 1
    a, b = before.subgraphs[0], after.subgraphs[0]
    assert np.array_equal(a.inputs, b.inputs) and np.array_equal(a.outputs, b.outputs)
    assert [pack(t) for t in a.tensors] == [pack(t) for t in b.tensors]
    assert [pack(t) for t in before.signatureDefs] == [pack(t) for t in after.signatureDefs]
    assert [pack(x) for x in before.buffers] == [pack(x) for x in after.buffers]
    changes = {r["fc_index"]: r for r in rewrite["changes"]}
    removed = {r["removed_ADD_index"] for r in rewrite["changes"]}
    assert len(changes) == len(removed) == 92
    actual = iter(b.operators)
    for index, old in enumerate(a.operators):
        if index in removed:
            continue
        new = next(actual)
        if index in changes:
            row = changes[index]
            add = a.operators[row["removed_ADD_index"]]
            assert old.inputs[2] == -1
            assert int(old.outputs[0]) == row["old_fc_output"]
            assert row["bias_tensor"] in add.inputs and row["old_fc_output"] in add.inputs
            old.inputs = old.inputs.copy()
            old.inputs[2] = row["bias_tensor"]
            old.outputs = add.outputs.copy()
        assert pack(old) == pack(new), index
    assert next(actual, None) is None
    assert rewrite["tensor_gate_passed"] and rewrite["counters_exact"]
    assert len(rewrite["parity_on15_state_fixtures"]) == 15
    for fixture in rewrite["parity_on15_state_fixtures"]:
        assert len(fixture) == 36
        assert all(v["allclose_rtol1e_4_atol1e_4"] for v in fixture.values())
    quality = read(ASR / "refinement_bias_evaluation.json")
    assert quality["quality_passed"]
    references = {
        "73": read(ASR / "evaluation_fc.json")["records"],
        "100": read(ASR / "multispeaker_evaluation.json")["models"]["fc"]["records"],
    }
    for name, row in quality["subsets"].items():
        initial.check_wer(row)
        for record, reference in zip(row["records"], references[name], strict=True):
            assert record["id"] == reference["id"] and record["reference"] == reference["reference"]
            assert record["tokens"] == reference["tokens"]
            assert record["final_cached_len"] == reference["final_cached_len"]
    continuous = quality["continuous"]
    assert continuous["tokens_equal"] and continuous["counters_exact"]
    assert continuous["encoder_parity"]["allclose_rtol1e_4_atol1e_4"]
    assert continuous["records"]["fc"]["tokens"] == continuous["records"]["fc_bias"]["tokens"]
    bench = read(ASR / "refinement_benchmark.json")
    assert datetime.fromisoformat(quality["finished_at_utc"]) < datetime.fromisoformat(
        bench["started_at_utc"]
    )
    for row in bench["models"].values():
        initial.stats(row["encoder_ms"], "samples", "median", "p95")
        initial.stats(row["pipeline_RTF"], "samples", "median", "p95")
    cold = read(ASR / "refinement_coldstart_memory.json")
    pids = []
    for row in cold["models"].values():
        assert len(row["samples"]) == 5
        for sample in row["samples"]:
            pids.append(sample["pid"])
            for reading in sample["rss_bytes"].values():
                assert reading["VmHWM"] >= reading["VmRSS"] > 0
            t = sample["timings"]
            initial.equal(
                t["init_ms"], t["constructor_ms"] + t["allocation_ms"] + t["signature_metadata_ms"]
            )
            initial.equal(t["init_plus_first_ms"], t["init_ms"] + t["first_signature_invoke_ms"])
        for key, value in row["timing_medians_ms"].items():
            initial.equal(value, float(np.median([r["timings"][key] for r in row["samples"]])))
        initial.equal(
            row["median_peak_RSS_MiB"],
            float(np.median([r["peak_RSS_bytes"] for r in row["samples"]])) / 1024**2,
        )
    assert len(set(pids)) == len(pids) == 15
    storage = rewrite["serialization"]
    assert storage["baseline"]["buffer_payload_bytes"] == storage["FC"]["buffer_payload_bytes"]
    assert storage["FC"]["non_payload_bytes"] - storage["baseline"]["non_payload_bytes"] == 936
    return {
        "92FC_ADD_fusions_only_all_buffer_tensor_signature_bytes_preserved": True,
        "173utterance_WER_tokens_and_counters_recomputed": True,
        "15fixture_and79_5second_state_gates_checked": True,
        "matched_timing_and15fresh_process_memory_recomputed": True,
        "936byte_delta_is_non_payload": True,
    }


def main():
    target = DAY / "REFINEMENT_VERIFICATION.json"
    if target.exists():
        raise FileExistsError(target)
    unchanged = 0
    for row in read(ROOT / "daily/2026-10-20/SNAPSHOT.json")["files"]:
        if row["path"] != "README.md":
            checked(row)
            unchanged += 1
    # Reuse independent old checks without overwriting VERIFICATION.json or old results.
    initial_mlsd = initial.verify_mlsd()
    initial_asr = initial.verify_asr()
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "previous_day_files_unchanged": unchanged,
        "initial_checks_passed": {"mlsd": initial_mlsd, "zipformer": initial_asr},
        "mlsd": verify_mlsd(),
        "zipformer": verify_asr(),
        "source_code": info(__file__),
        "scope": "Evidence/metric/graph integrity,not deployment certification;native per-op LiteRT attribution remains unmeasured.",
    }
    write(target, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
