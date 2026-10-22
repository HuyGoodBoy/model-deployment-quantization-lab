"""Check report tables and save a manifest of reviewable day22 files."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

DAY = Path(__file__).resolve().parents[1]
ROOT = DAY.parents[1]


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
    assert read(DAY / "VERIFICATION.json")["passed"]
    review_path = DAY / "verification/astra_structure_review.json"
    review = read(review_path)
    assert review["passed"] and review["zipformer_independent_review"]["passed"]
    for row in review["sources"]:
        assert info(ROOT / row["path"]) == row, row["path"]
    report = (DAY / "REPORT.md").read_text(encoding="utf-8").replace("**", "")
    mlsd = DAY / "experiments/mlsd/results"
    asr = DAY / "experiments/zipformer/results"
    for name, label in [
        ("float", "FP32"),
        ("C1", "C1"),
        ("C1_roundtrip", "C1 đóng gói lại"),
        ("M1", "M1"),
        ("M2", "M2"),
    ]:
        values = [
            read(mlsd / f"benchmark_t{t}_s{s}.json")["models"][name]
            for t, s in [(4, 1), (4, 2), (8, 1), (8, 2)]
        ]
        expected = f"| {label} | {vn(values[0]['median_ms'])} / {vn(values[1]['median_ms'])} | {vn(values[2]['median_ms'])} / {vn(values[3]['median_ms'])} | {vn(values[2]['p95_ms'])} / {vn(values[3]['p95_ms'])} |"
        assert expected in report, expected
    for name, label in [
        ("C1", "C1"),
        ("C1_roundtrip", "C1 đóng gói lại"),
        ("M1", "M1"),
        ("M2", "M2"),
    ]:
        row = read(mlsd / "io_profile.json")["models"][name + "_t8"]["medians_ms"]
        values = [
            row[k]
            for k in [
                "application_quantize",
                "set_tensor_copy",
                "native_invoke",
                "application_output_adapter",
                "total",
            ]
        ]
        expected = "| " + " | ".join([label, *[vn(x) for x in values]]) + " |"
        assert expected in report, expected
    for name, label in [("Z0", "Z0: SignatureRunner"), ("Z1", "Z1: gọi trực tiếp")]:
        rows = [read(asr / f"benchmark_s{s}.json")["models"][name] for s in [1, 2]]
        expected = f"| {label} | {vn(rows[0]['components_ms']['encoder']['median'])} / {vn(rows[1]['components_ms']['encoder']['median'])} | {vn(rows[0]['pipeline_RTF']['median'], 5)} / {vn(rows[1]['pipeline_RTF']['median'], 5)} |"
        assert expected in report, expected
    files = [ROOT / "README.md"]
    for path in sorted(DAY.rglob("*")):
        if (
            not path.is_file()
            or any(x in {"artifacts", "__pycache__"} for x in path.parts)
            or path.name == "SNAPSHOT.json"
        ):
            continue
        assert path.stat().st_size < 2_000_000, path
        files.append(path)
    records = [info(path) for path in files]
    target = DAY / "SNAPSHOT.json"
    assert not target.exists()
    value = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Reviewable source, documents and small evidence; excludes model/audio/tensor artifacts and this manifest.",
        "verification": info(DAY / "VERIFICATION.json"),
        "independent_structure_statistics_review": info(review_path),
        "report_tables_match_raw_results": True,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
    }
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print({k: v for k, v in value.items() if k != "files"})


if __name__ == "__main__":
    main()
