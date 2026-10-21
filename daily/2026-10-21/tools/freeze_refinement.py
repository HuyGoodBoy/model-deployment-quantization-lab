"""Check report tables/local links and freeze the local handoff without git writes."""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-21"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def info(path):
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def vn(value, digits=3):
    return f"{value:,.{digits}f}".replace(",", "~").replace(".", ",").replace("~", ".")


def main():
    assert read(DAY / "REFINEMENT_VERIFICATION.json")["passed"]
    report = (DAY / "REPORT.md").read_text(encoding="utf-8").replace("**", "")
    asr = DAY / "experiments/zipformer/results"
    mlsd = DAY / "experiments/mlsd/results"
    timing = read(asr / "refinement_benchmark.json")["models"]
    cold = read(asr / "refinement_coldstart_memory.json")["models"]
    for key, label in [("baseline", "Baseline"), ("fc", "FC"), ("fc_bias", "FC + bias")]:
        row = timing[key]
        expected = f"| {label} | {vn(row['encoder_ms']['median'])} / {vn(row['encoder_ms']['p95'])} | {vn(row['pipeline_RTF']['median'], 5)} |"
        assert expected in report, expected
        row = cold[key]
        t = row["timing_medians_ms"]
        expected = f"| {label} | {vn(t['init_ms'], 1)} | {vn(t['first_signature_invoke_ms'], 2)} | {vn(row['median_peak_RSS_MiB'], 2)} / {vn(row['max_peak_RSS_MiB'], 2)} |"
        assert expected in report, expected
    for name, label in [("float", "FP32"), ("head_float", "A"), ("C1", "C1")]:
        values = [
            read(mlsd / f"refinement_contiguous_benchmark_t{t}_s{s}.json")["models"][name][
                "median_ms"
            ]
            for t, s in [(4, 1), (4, 2), (8, 1), (8, 2)]
        ]
        expected = "| " + " | ".join([label, *[vn(v) for v in values]]) + " |"
        assert expected in report, expected
    for name, label, precision in [
        ("C1", "C1: prune 5 kênh, toàn head float", "30 / 11"),
        ("C2", "C2: prune, chỉ Conv cuối/output float", "40 / 1"),
        ("C3", "C3: prune, suffix block16 float", "38 / 3"),
    ]:
        row = read(mlsd / f"refinement_{name}.json")
        regression = (
            (
                vn(
                    read(mlsd / "refinement_regression.json")["models"][name]["sAP10"][
                        "sAP10_subset_percent"
                    ],
                    4,
                )
                + "%"
            )
            if name == "C1"
            else "—"
        )
        size = f"{row['model']['bytes']:,}".replace(",", ".")
        expected = f"| {label} | {precision} | {size} | {vn(row['sAP10']['sAP10_subset_percent'], 4)}% | {regression} |"
        assert expected in report, expected
    profile = read(mlsd / "refinement_io_profile.json")["models"]
    for name, label in [("float", "FP32"), ("head_float", "A"), ("C1", "C1")]:
        row = profile[name + "_t8"]["medians_ms"]
        values = [
            row[k]
            for k in [
                "application_quantize",
                "native_invoke",
                "application_output_adapter",
                "total",
            ]
        ]
        expected = "| " + " | ".join([label, *[vn(v) for v in values]]) + " |"
        assert expected in report, expected
    c1 = read(mlsd / "refinement_C1.json")["model"]["bytes"]
    floating = read(mlsd / "exports.json")["models"]["float"]["model"]["bytes"]
    assert vn(100 * (1 - c1 / floating), 1) + "%" in report
    sections = re.split(
        r"^## ", (DAY / "REPORT.md").read_text(encoding="utf-8"), flags=re.MULTILINE
    )[1:]
    assert len(sections) == 2
    assert sections[0].startswith("ZipFormer") and sections[1].startswith("M-LSD")
    for section in sections:
        lines = [line.strip() for line in section.splitlines()[1:] if line.strip()]
        assert lines[0].startswith("- Evidence:"), section.splitlines()[0]
    local_links = 0
    for path in [
        DAY / name for name in ["REPORT.md", "RESEARCH.md", "RUNNING.md", "PROTOCOL_REFINEMENT.md"]
    ]:
        content = path.read_text(encoding="utf-8")
        assert not any(ord(c) < 32 and c not in "\n\r\t" for c in content)
        for target in re.findall(r"\[[^\]]*\]\(([^)\n]+)\)", content):
            if target.startswith(("https://", "http://", "#")):
                continue
            target = target.strip("<>").split("#", 1)[0]
            assert (path.parent / target).resolve().exists(), (path, target)
            local_links += 1
    initial = read(DAY / "SNAPSHOT.initial.json")
    editable = {
        "README.md",
        *["daily/2026-10-21/" + n for n in ["REPORT.md", "RESEARCH.md", "RUNNING.md"]],
    }
    old_preserved = 0
    for row in initial["files"]:
        if row["path"] in editable:
            continue
        assert info(ROOT / row["path"]) == row, row["path"]
        old_preserved += 1
    files = [ROOT / "README.md"]
    for path in sorted(DAY.rglob("*")):
        if not path.is_file() or any(p in {"artifacts", "__pycache__"} for p in path.parts):
            continue
        if path == DAY / "SNAPSHOT.json":
            continue
        assert path.stat().st_size < 2_000_000, (path, "unexpected large tracked artifact")
        files.append(path)
    records = [info(path) for path in files]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    result = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Day21 local handoff after approved refinement. No commit/push;original results/sources and day20 preserved. Source-at-execution hashes may precede final Ruff formatting. Excludes this snapshot itself and ignored model/audio/tensor artifacts.",
        "git_head_unchanged": head,
        "initial_snapshot": info(DAY / "SNAPSHOT.initial.json"),
        "initial_source_result_files_preserved": old_preserved,
        "document_checks": {
            "numeric_tables_and_C1_size_reduction_match_results": True,
            "project_groups_start_with_Evidence": True,
            "local_links_resolve": local_links,
        },
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
    }
    (DAY / "SNAPSHOT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=True, indent=2)
    )


if __name__ == "__main__":
    main()
