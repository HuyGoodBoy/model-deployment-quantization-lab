"""Recompute saved quality metrics and validate chronology without timing again."""

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DAY = Path(__file__).resolve().parents[1]
ROOT = DAY.parents[1]
MLSD = DAY / "experiments/mlsd/results"
ASR = DAY / "experiments/zipformer/results"
sys.path.insert(0, str(DAY / "experiments/mlsd/src"))

from mlsd22 import DAY21, checked, info, now, pilot_data, read, write


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


old = module("prior_verifier", ROOT / "daily/2026-10-21/tools/verify.py")


def before(a, b):
    assert datetime.fromisoformat(a) < datetime.fromisoformat(b), (a, b)


def speed_decision(selection, quality):
    """Check the rule rather than requiring the same winner on another run."""
    rows = {s: read(MLSD / f"benchmark_t8_s{s}.json")["models"] for s in [1, 2]}
    eligible = []
    for candidate in quality["quality_eligible"]:
        passes = []
        for session in [1, 2]:
            for reference in ["float", "C1", "C1_roundtrip"]:
                comparison = selection["comparisons"][f"t8_s{session}"][
                    candidate + "_vs_" + reference
                ]
                a = np.array(rows[session][reference]["samples_ms"]).reshape(5, 16)
                b = np.array(rows[session][candidate]["samples_ms"]).reshape(5, 16)
                indices = np.random.default_rng(comparison["seed"]).integers(0, 5, (2000, 5))
                gains = 100 * (
                    1 - np.median(b[indices], axis=(1, 2)) / np.median(a[indices], axis=(1, 2))
                )
                old.equal(
                    float(100 * (1 - np.median(b) / np.median(a))),
                    comparison["median_reduction_percent"],
                )
                old.equal(
                    np.percentile(gains, [2.5, 97.5]).tolist(),
                    comparison["paired_round_bootstrap95_percent"],
                )
                if reference in ["float", "C1"]:
                    passes.append(np.median(b) < np.median(a) and np.percentile(gains, 2.5) > 0)
        if all(passes):
            eligible.append(candidate)
    assert eligible == selection["eligible_8threads"]
    selected = (
        min(eligible, key=lambda n: sum(rows[s][n]["median_ms"] for s in [1, 2]))
        if eligible
        else "float"
    )
    if (
        selected == "M2"
        and "M1" in eligible
        and not all(
            selection["comparisons"][f"t8_s{s}"]["M2_vs_M1"]["median_reduction_percent"] >= 1
            and selection["comparisons"][f"t8_s{s}"]["M2_vs_M1"][
                "paired_round_bootstrap95_percent"
            ][0]
            > 0
            for s in [1, 2]
        )
    ):
        selected = "M1"
    assert selection["selected"] == selected
    return selected


def mlsd():
    pilot, data = pilot_data()
    quality = read(MLSD / "quality_selection.json")
    configured = read(MLSD / "configuration_validation.json")
    selection = read(MLSD / "speed_selection.json")
    assert configured["passed"]
    before(configured["finished_at_utc"], selection["locked_at_utc"])
    base = read(DAY21 / "results/development.json")["models"]["float"]
    reference = old.load(read(DAY21 / "results/refinement_C1.json")["development_output"])
    for name in ["M1", "M2"]:
        row = read(MLSD / f"quality_{name}.json")
        output = old.load(row["output"])
        old.equal(pilot.line_metrics(output, data["ground_truth"]), row["sAP10"])
        assert np.array_equal(output, reference)
        assert row["quality_passed"] == (
            base["sAP10"]["sAP10_subset_percent"] - row["sAP10"]["sAP10_subset_percent"] <= 0.5
        )
    for name, row in configured["models"].items():
        output = old.load(row["output"])
        assert np.array_equal(output, old.load(row["reference_output"]))
        old.equal(pilot.line_metrics(output, data["ground_truth"]), row["sAP10"])
    for threads in [4, 8]:
        for session in [1, 2]:
            bench = read(MLSD / f"benchmark_t{threads}_s{session}.json")
            before(quality["locked_at_utc"], bench["started_at_utc"])
            before(bench["finished_at_utc"], selection["locked_at_utc"])
            for row in bench["models"].values():
                checked(row["model"])
                old.stats(row)
                assert row["runs"] == 80
                old.equal(
                    row["round_medians_ms"],
                    np.median(np.array(row["samples_ms"]).reshape(5, 16), axis=1).tolist(),
                )
    reg = read(MLSD / "regression.json")
    before(selection["locked_at_utc"], reg["started_at_utc"])
    assert speed_decision(selection, quality) == reg["selected_before_regression"]
    truth = read(checked(read(checked(reg["manifest"]))["ground_truth"]))
    prior = old.load(read(DAY21 / "results/refinement_regression.json")["models"]["C1"]["output"])
    for row in reg["models"].values():
        output = old.load(row["output"])
        old.equal(pilot.line_metrics(output, truth), row["sAP10"])
        assert np.array_equal(output, prior)
    profile = read(MLSD / "io_profile.json")
    for row in profile["models"].values():
        assert row["signature_vs_staged_output_exact"]
        for sample in row["samples_ms"]:
            old.equal(sum(v for k, v in sample.items() if k != "total"), sample["total"])
        for key, value in row["medians_ms"].items():
            old.equal(float(np.median([s[key] for s in row["samples_ms"]])), value)
    assert read(MLSD / "quantize_probe.json")["rounding_gate_passed"]
    return {
        "development_and_regression_AP_recomputed": True,
        "t8_outputs_exact_vs_t4": True,
        "four_matched_benchmarks_recomputed": True,
        "selection_before_regression": True,
        "staged_profile_totals_recomputed": True,
    }


def asr():
    parity = read(ASR / "fixture_parity.json")
    assert parity["exact_passed"]
    assert (
        len(parity["encoder"]) == 15
        and len(parity["decoder"]) == 18
        and len(parity["joiner"]) == 30
    )
    assert all(
        len(row["outputs"]) == 36 and all(v["exact"] for v in row["outputs"].values())
        for row in parity["encoder"]
    )
    reg = read(ASR / "regression.json")
    assert reg["quality_passed"]
    prior_dir = ROOT / "daily/2026-10-21/experiments/zipformer/results"
    prior_records = {
        "73": read(prior_dir / "evaluation_fc.json")["records"],
        "100": read(prior_dir / "multispeaker_evaluation.json")["models"]["fc"]["records"],
    }
    count = 0
    for key, row in reg["subsets"].items():
        old.check_wer(row)
        assert len(row["records"]) == int(key)
        assert all(
            r["original_tokens_equal"] and r["original_counters_equal"] for r in row["records"]
        )
        for actual, reference in zip(row["records"], prior_records[key], strict=True):
            assert actual["id"] == reference["id"]
            assert actual["reference"] == reference["reference"]
            assert actual["tokens"] == reference["tokens"]
            assert actual["final_cached_len"] == reference["final_cached_len"]
        count += len(row["records"])
    continuous = read(ASR / "continuous_state.json")
    assert count == 173 and continuous["exact_passed"] and continuous["audio_seconds"] == 600
    a, b = continuous["records"]["Z0"], continuous["records"]["Z1"]
    assert a["tokens"] == b["tokens"] and a["final_cached_len"] == b["final_cached_len"]
    assert a["chunks"] == b["chunks"] == 1875 and continuous["encoded_parity"]["exact"]
    for session in [1, 2]:
        bench = read(ASR / f"benchmark_s{session}.json")
        before(continuous["finished_at_utc"], bench["started_at_utc"])
        for row in bench["models"].values():
            for sample in row["components_ms"].values():
                old.stats(sample, "samples", "median", "p95")
                assert sample["runs"] == 160
            old.stats(row["pipeline_RTF"], "samples", "median", "p95")
            for round_ in row["pipeline_rounds"]:
                old.equal(round_["RTF"], round_["pipeline_seconds"] / round_["audio_seconds"])
    return {
        "WER_from_transcripts_recomputed": True,
        "tokens_and_final_counters_173_checked": True,
        "fixtures_and_continuous_encoder_token_finalcounter_checked": True,
        "two_session_component_pipeline_statistics_recomputed": True,
        "limitation": "Continuous test does not compare all cached-state tensors at every chunk.",
    }


def documents():
    count = 0
    for name in ["REPORT.md", "RESEARCH.md", "RUNNING.md"]:
        path = DAY / name
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (name, target)
            count += 1
    report = (DAY / "REPORT.md").read_text(encoding="utf-8")
    for text in report.split("\n## ")[1:]:
        assert text.split("\n\n", 1)[1].startswith("- Evidence:")
    for term in [
        "Astra",
        "Codex",
        "Claude",
        "Gemini",
        "push",
        "commit",
        "local",
        "snapshot",
        "trò chuyện",
    ]:
        assert term.lower() not in report.lower(), term
    return {
        "relative_links_resolve": count,
        "evidence_first_per_project": True,
        "mentor_report_content_checked": True,
    }


def main():
    target = DAY / "VERIFICATION.json"
    assert not target.exists()
    unchanged = 0
    # The day21 final snapshot predates its later editorial report cleanup;
    # use Git's committed tree for old-day integrity, not obsolete editorial hashes.
    import subprocess

    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "daily/2026-10-20", "daily/2026-10-21"], cwd=ROOT, text=True
    )
    assert not changed
    for file in (ROOT / "daily/2026-10-21").rglob("*.py"):
        unchanged += 1
    value = {
        "verified_at_utc": now(),
        "passed": True,
        "old_day_git_diff_empty": True,
        "day21_python_files": unchanged,
        "mlsd": mlsd(),
        "zipformer": asr(),
        "documents": documents(),
        "source_code": info(__file__),
    }
    write(target, value)
    print(value)


if __name__ == "__main__":
    main()
