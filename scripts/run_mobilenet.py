"""Chay tat ca hoac tung buoc cua bai thuc hanh."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("prepare", "convert", "evaluate", "benchmark", "report")


def execute(module, *arguments):
    print(f"\n>>> {module}", flush=True)
    subprocess.run(
        [sys.executable, "-m", f"lab.{module}", *map(str, arguments)], cwd=ROOT, check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--images-dir", type=Path, help="Thu muc anh that cua ban")
    parser.add_argument("--labels", type=Path, help="CSV filename,class_index; ImageNet 0..999")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--strict", action="store_true", help="Exit 2 neu output khong dat allclose"
    )
    args = parser.parse_args()
    if min(args.limit, args.threads, args.warmup, args.runs) < 1:
        parser.error("limit, threads, warmup, runs phai >= 1")
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol va rtol phai >= 0")
    for step in STEPS:
        if step not in args.steps:
            continue
        if step == "prepare":
            extra = ["--images-dir", args.images_dir.resolve()] if args.images_dir else []
            execute(step, "--limit", args.limit, *extra)
        elif step == "convert":
            execute(step)
        elif step == "evaluate":
            extra = ["--labels", args.labels.resolve()] if args.labels else []
            execute(
                step, "--threads", args.threads, "--atol", args.atol, "--rtol", args.rtol, *extra
            )
        elif step == "benchmark":
            for runtime in ("tensorflow", "onnx", "tflite"):
                execute(
                    step,
                    "--runtime",
                    runtime,
                    "--threads",
                    args.threads,
                    "--warmup",
                    args.warmup,
                    "--runs",
                    args.runs,
                )
        else:
            execute(step)
    if args.strict and "evaluate" in args.steps:
        data = json.loads((ROOT / "results" / "evaluation.json").read_text(encoding="utf-8"))
        failed = [r for r, values in data["comparisons"].items() if not values["allclose"]]
        if failed:
            print(
                f"FP32 tolerance check failed: {failed}. See results/evaluation.json.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"Stage failed (exit {exc.returncode}). Read the error above; artifacts are retained.",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
