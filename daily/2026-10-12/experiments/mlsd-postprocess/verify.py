"""Compare Java with pinned NAVER post-processing, then benchmark adjacency search.

Run with the baseline environment (NumPy 1.26 / TensorFlow 2.15.1).
Only output decoding is needed when refreshing real fixtures; no model inference.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import tensorflow as tf

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[3]
CACHE = ROOT / ".cache/day12-postprocess"
FIXTURES = TASK / "fixtures"
RESULTS = TASK / "results"
UPSTREAM_URL = "https://raw.githubusercontent.com/navervision/mlsd/453cafa09467d0272760578d35c1fda38e8895a5/utils.py"
UPSTREAM_SHA = "2e7caac40d7831f6a9b5e9ab97eed436d80f60dd03bf6d2572deac6277e4b15a"
DEFAULTS = {
    "score": 0.06,
    "outside_ratio": 0.28,
    "inside_ratio": 0.45,
    "w_overlap": 0.0,
    "w_degree": 1.95,
    "w_length": 0.0,
    "w_area": 1.86,
    "w_center": 0.14,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def refresh_fixtures() -> None:
    """Small, committed CSV fixtures make verification independent of model artifacts."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = []

    def add(name, lines, description, robust=False, **metadata):
        path = FIXTURES / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x1", "y1", "x2", "y2"])
            writer.writerows(lines)
        manifest.append(
            dict(
                name=name,
                description=description,
                source_compatible=not robust,
                sha256=sha(path),
                **metadata,
            )
        )

    add(
        "fragmented",
        [
            [20.3, 30.4, 80.2, 30.4],
            [80, 30.4, 140.5, 30.4],
            [141, 30.4, 220.8, 30.4],
            [220.8, 30.4, 220.8, 220.7],
            [220.8, 220.7, 120, 220.7],
            [120, 220.7, 20.3, 220.7],
            [20.3, 220.7, 20.3, 30.4],
        ],
        "Một hình chữ nhật, cạnh trên bị chia ba và cạnh dưới bị chia hai.",
    )
    angle = np.deg2rad(18)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.array([[-70, -45], [70, -45], [70, 45], [-70, 45]]) @ rotation.T + 128.3
    add("rotated", [np.r_[points[k], points[(k + 1) % 4]] for k in range(4)], "Box xoay 18 độ.")
    integer = [[20, 30, 220, 30], [220, 30, 220, 220], [220, 220, 20, 220], [20, 220, 20, 30]]
    add("integer", integer, "Endpoint nguyên để kiểm tra epsilon và ép INT32 ở giao điểm.")
    add("reversed", [s[2:] + s[:2] for s in integer], "Đảo thứ tự hai đầu mút của từng đoạn.")
    add(
        "endpoint_gaps",
        [
            [24.2, 20.3, 197.6, 20.3],
            [201.7, 24.4, 201.7, 196.5],
            [197.6, 200.7, 24.2, 200.7],
            [20.1, 196.5, 20.1, 24.4],
        ],
        "Bốn cạnh có khoảng hở ở góc.",
    )
    add("parallel", [[20.3, 30.2, 230.7, 30.2], [20.3, 70.2, 230.7, 70.2]], "Hai đường song song.")
    add(
        "near_parallel",
        [[20.3, 30.2, 230.7, 30.2], [20.3, 31.2, 230.7, 31.20000001]],
        "Hai đường gần song song, giao điểm ở rất xa.",
    )
    add(
        "border",
        [
            [0.1, 0.1, 220.4, 0.1],
            [220.4, 0.1, 220.4, 220.4],
            [220.4, 220.4, 0.1, 220.4],
            [0.1, 220.4, 0.1, 0.1],
        ],
        "Đường gần biên accumulator Hough.",
    )
    coords = [25.3 + 30 * k for k in range(8)]
    grid = [[coords[0], y, coords[-1], y] for y in coords]
    grid += [[x, coords[0], x, coords[-1]] for x in coords]
    add("grid", grid, "Lưới 8 ngang × 8 dọc để thử số tổ hợp khi ghép box.")
    add("empty", [], "Không có đoạn thẳng.", robust=True)
    add("zero_length", [[30, 30, 30, 30]], "Đoạn có chiều dài bằng 0.", robust=True)
    add(
        "invalid",
        [[float("nan"), 0, 100, 0], [0, 0, float("inf"), 100]],
        "NaN/Inf: Java cần bỏ qua, source Python có thể báo lỗi.",
        robust=True,
    )
    add(
        "outside_accumulator",
        [[600, 0, 600, 100], [0, 600, 100, 600]],
        "Đường ngoài kích thước bảng Hough: Java bỏ qua thay vì truy cập ngoài mảng.",
        robust=True,
    )
    source_results = ROOT / "daily/2026-10-09/experiments/03-mlsd/results/run-01"
    for variant in ["decoded_fp32_old_t4", "official_old_t4", "static_old_t4"]:
        path = source_results / f"demo_{variant}.npz"
        data = np.load(path, allow_pickle=False)
        lines = []
        for (y, x), score in zip(data["points"][0], data["scores"][0]):
            dx1, dy1, dx2, dy2 = data["vmap"][0, int(y), int(x)]
            if score > 0.06 and np.linalg.norm(np.array([dx1 - dx2, dy1 - dy2])) > 20:
                # Match pred_squares: displacement promoted to float64 by d_arrow=1.0.
                lines.append(
                    [
                        float(x) + float(dx1),
                        float(y) + float(dy1),
                        float(x) + float(dx2),
                        float(y) + float(dy2),
                    ]
                )
        measurement = json.loads(
            (source_results / f"measure_{variant}.json").read_text(encoding="utf-8")
        )
        add(
            "real_" + variant,
            lines,
            "Đoạn đã decode từ output M-LSD trên frame_1.jpg đã lưu ngày 09.",
            raw_output_sha256=sha(path),
            model_sha256=measurement["model_sha256"],
            input_sha256=measurement["input_sha256"],
        )
    write_json(
        FIXTURES / "manifest.json",
        {
            "coordinates": "map 256x256, before final image scaling",
            "thresholds": {"score": 0.06, "length": 20},
            "fixtures": manifest,
        },
    )


def upstream_reference():
    """Execute upstream algorithm unchanged after its model/decode boundary.

    AST extraction skips only image preprocessing, inference and line decoding.
    Capture locals after execution to compare intermediate steps, not just top box.
    """
    path = ROOT / ".cache/mlsd/utils_453cafa.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(UPSTREAM_URL, timeout=30).read())
    if sha(path) != UPSTREAM_SHA:
        raise RuntimeError("Upstream source hash mismatch")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pred_squares"
    )
    start = next(
        i
        for i, n in enumerate(original.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "point" for t in n.targets)
    )
    body = original.body[start:]
    assert isinstance(body[-1], ast.Return)
    body[-1] = ast.Return(
        value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])
    )
    prefix = ast.parse(
        "pts=np.zeros((200,2)); input_shape=[512,512]; original_shape=[256,256]"
    ).body
    function = ast.FunctionDef(
        name="reference",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="segments"), ast.arg(arg="params")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=prefix + body,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {"np": np, "tf": tf}
    exec(compile(module, str(path), "exec"), scope)  # noqa: S102 -- pinned source verified by SHA-256 above
    return scope["reference"]


def reference_output(reference, lines, weights):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = reference(np.asarray(lines, dtype=np.float64).reshape(-1, 4), weights)
    groups = trace["corner_dict"]
    corners = [[t, *list(c)] for t in range(4) for c in groups[t]]
    scores = np.asarray(trace.get("score_array", [])).reshape(-1)
    squares = np.asarray(trace.get("squares", [])).reshape(-1, 4, 2)
    boxes = []
    for i in range(len(scores)):
        idx = trace["sorted_idx"][i]
        boxes.append(
            {
                "score": float(scores[i]),
                "corners": squares[i].tolist(),
                "edges": trace["connect_array"][idx].tolist(),
                "components": [
                    float(trace[name][idx])
                    for name in [
                        "overlap_scores",
                        "degree_scores",
                        "area_scores",
                        "center_scores",
                        "length_scores",
                    ]
                ],
            }
        )
    return {
        "merged": np.asarray(trace["new_segments"]).reshape(-1, 4).tolist(),
        "corners": corners,
        "boxes": boxes,
    }


def compare(reference, java):
    a = np.asarray(reference["merged"]).reshape(-1, 4)
    b = np.asarray(java["merged"]).reshape(-1, 4)
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=0):
        raise AssertionError(f"Merged lines differ: {a.shape} vs {b.shape}")
    if reference["corners"] != java["corners"]:
        raise AssertionError(
            f"Corners differ: reference {reference['corners']} vs Java {java['corners']}"
        )

    def key(box):
        return tuple(np.asarray(box["corners"]).reshape(-1)) + tuple(box["edges"])

    left = sorted(reference["boxes"], key=key)
    right = sorted(java["boxes"], key=key)
    if len(left) != len(right) or [key(b) for b in left] != [key(b) for b in right]:
        raise AssertionError(f"Candidate set differs: {len(left)} vs {len(right)}")
    error = 0.0
    for expected, actual in zip(left, right):
        error = max(
            error,
            abs(expected["score"] - actual["score"]),
            float(np.max(np.abs(np.array(expected["components"]) - actual["components"]))),
        )
    if error > 1e-8:
        raise AssertionError(f"Scoring difference {error}")
    # Tie order may differ, but ranking scores and all candidate identities must agree.
    if not np.allclose(
        [b["score"] for b in reference["boxes"]],
        [b["score"] for b in java["boxes"]],
        atol=1e-8,
        rtol=0,
    ):
        raise AssertionError("Ranking differs outside score ties")
    return {
        "merged_max_abs_error": float(np.max(np.abs(a - b))) if a.size else 0.0,
        "max_score_component_error": error,
        "corner_coordinates_exact": True,
        "candidate_set_exact": True,
        "ranking_scores_match": True,
        "tie_order": "not required",
    }


def java_tools():
    home = os.environ.get("JAVA_HOME")
    candidates = [Path(home) / "bin"] if home else []
    if shutil.which("javac"):
        candidates.append(Path(shutil.which("javac")).parent)
    candidates += sorted((ROOT / ".cache/android-tools/java").glob("*/bin"))
    suffix = ".exe" if os.name == "nt" else ""
    for folder in candidates:
        if (folder / ("javac" + suffix)).exists() and (folder / ("java" + suffix)).exists():
            return folder / ("javac" + suffix), folder / ("java" + suffix)
    raise RuntimeError("JDK 11+ required; set JAVA_HOME or put javac on PATH")


def cpu_name():
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def main(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.refresh_fixtures:
        refresh_fixtures()
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    reference = upstream_reference()
    javac, java = java_tools()
    classes = CACHE / "classes"
    classes.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(javac),
            "--release",
            "8",
            "-d",
            str(classes),
            *map(str, sorted((TASK / "src").glob("*.java"))),
        ],
        check=True,
    )
    record = {
        "date": "2026-10-12",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "desktop JVM post-processing; no model inference or Android benchmark",
        "source": {"url": UPSTREAM_URL, "sha256": UPSTREAM_SHA},
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu_name(),
            "logical_processors": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "java": subprocess.run(
                [str(java), "-version"], capture_output=True, text=True, check=True
            ).stderr.strip(),
            "single_java_application_thread": True,
        },
        "protocol": {
            "warmup_per_mode": args.warmup,
            "runs_per_mode": args.runs,
            "order": "paired measurements; alternate naive/adjacency first each iteration",
            "includes": "Hough merge, corners, adjacency construction/search, scoring and sorting",
            "excludes": "decode model output, input/output CSV, model inference, image remapping",
            "timing": "System.nanoTime per stage; total from first to last stage",
        },
        "code_sha256": {
            p.name: sha(p) for p in [*sorted((TASK / "src").glob("*.java")), Path(__file__)]
        },
        "fixtures": [],
        "all_passed": True,
    }
    timing_rows = []
    for item in manifest["fixtures"]:
        name = item["name"]
        path = FIXTURES / f"{name}.csv"
        if sha(path) != item["sha256"]:
            raise AssertionError(f"Fixture changed: {name}")
        with path.open(newline="", encoding="utf-8") as stream:
            lines = [
                [float(r[k]) for k in ["x1", "y1", "x2", "y2"]] for r in csv.DictReader(stream)
            ]
        output = CACHE / f"{name}.json"
        subprocess.run(
            [
                str(java),
                "-Xms128m",
                "-Xmx512m",
                "-cp",
                str(classes),
                "PostprocessBench",
                str(path),
                str(output),
                str(args.warmup),
                str(args.runs),
            ],
            check=True,
        )
        actual = json.loads(output.read_text(encoding="utf-8"))
        row = {
            "name": name,
            "input_sha256": item["sha256"],
            "input_segments": len(lines),
            "merged_segments": len(actual["adjacency"]["merged"]),
            "corner_counts": [
                sum(c[0] == i for c in actual["adjacency"]["corners"]) for i in range(4)
            ],
            "boxes": len(actual["adjacency"]["boxes"]),
            "naive_checks": actual["naive"]["connection_checks"],
            "adjacency_checks": actual["adjacency"]["connection_checks"],
            "adjacency_output_exact": True,
        }
        try:
            if item["source_compatible"]:
                expected = reference_output(reference, lines, DEFAULTS.copy())
                row["reference_check"] = compare(expected, actual["adjacency"])
            else:
                assert not actual["adjacency"]["boxes"] and not actual["adjacency"]["merged"]
                try:
                    reference_output(reference, lines, DEFAULTS.copy())
                except (IndexError, ValueError, ZeroDivisionError, OverflowError) as exc:
                    row["upstream_behavior"] = type(exc).__name__ + ": " + str(exc)
                else:
                    row["upstream_behavior"] = "returned without exception"
                row["reference_check"] = (
                    "robustness case: empty Java result required, not upstream equivalence"
                )
            row["passed"] = True
        except (
            AssertionError,
            IndexError,
            KeyError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ) as exc:
            row["passed"] = False
            row["error"] = str(exc)
            record["all_passed"] = False
        row["benchmark"] = {}
        for mode, stages in actual["benchmark"].items():
            row["benchmark"][mode] = {
                stage: {k: v for k, v in values.items() if k != "samples_ms"}
                for stage, values in stages.items()
            }
            for run in range(args.runs):
                timing_rows.append(
                    [
                        name,
                        mode,
                        run,
                        *[
                            stages[s]["samples_ms"][run]
                            for s in ["merge", "corners", "search", "score", "total"]
                        ],
                    ]
                )
        record["fixtures"].append(row)
    # Explicitly turn on overlap and length once; default zero weights could hide errors.
    record["nondefault_weights_checks"] = []
    for name in ["fragmented", "grid", "real_decoded_fp32_old_t4"]:
        path = FIXTURES / f"{name}.csv"
        output = CACHE / f"{name}_all_components.json"
        subprocess.run(
            [
                str(java),
                "-cp",
                str(classes),
                "PostprocessBench",
                str(path),
                str(output),
                "1",
                "1",
                "all-components",
            ],
            check=True,
        )
        with path.open(newline="", encoding="utf-8") as stream:
            lines = [
                [float(r[k]) for k in ["x1", "y1", "x2", "y2"]] for r in csv.DictReader(stream)
            ]
        weights = DEFAULTS.copy()
        weights.update(w_overlap=0.7, w_length=0.6)
        try:
            check = compare(
                reference_output(reference, lines, weights),
                json.loads(output.read_text())["adjacency"],
            )
            record["nondefault_weights_checks"].append(dict(name=name, passed=True, **check))
        except (
            AssertionError,
            IndexError,
            KeyError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ) as exc:
            record["nondefault_weights_checks"].append(
                {"name": name, "passed": False, "error": str(exc)}
            )
            record["all_passed"] = False
    with (RESULTS / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "fixture",
                "mode",
                "run",
                "merge_ms",
                "corners_ms",
                "search_ms",
                "score_ms",
                "total_ms",
            ]
        )
        writer.writerows(timing_rows)
    record["timings_sha256"] = sha(RESULTS / "timings.csv")
    write_json(RESULTS / "verification.json", record)
    print("All passed:", record["all_passed"], flush=True)
    for row in record["fixtures"]:
        print(row["name"], "OK" if row["passed"] else row["error"], flush=True)
    if not record["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-fixtures", action="store_true")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=400)
    args = parser.parse_args()
    if min(args.warmup, args.runs) < 1:
        parser.error("Positive warmup/runs required")
    main(args)
