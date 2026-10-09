"""Run separate stages/processes, preserve failed variants, produce the handoff."""

import argparse
import subprocess
import sys

from qlab.common import RESULTS, ROOT, VARIANTS, read_json


def launch(task, stage, variant=None, required=True, extra=()):
    name = stage + ("_" + variant if variant else "")
    path = ROOT / "logs" / task / (name + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "qlab." + stage]
    if task != "all":
        command += ["--task", task]
    if variant:
        command += ["--variant", variant]
    command += list(extra)
    print(f"START {task}/{name}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"FAILED {task}/{name}: see {path.relative_to(ROOT)}", flush=True)
        if required:
            raise RuntimeError(f"{task}/{name} failed")
    else:
        print(f"PASS {task}/{name}", flush=True)
    return result.returncode == 0


def converted(task, variant):
    if variant == "tensorflow_fp32":
        return True
    path = RESULTS / task / f"conversion_{variant}.json"
    return path.exists() and read_json(path)["status"] == "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cv", "text", "all"), default="all")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "download",
            "prepare",
            "convert",
            "evaluate",
            "benchmark",
            "diagnose",
            "report",
            "verify",
        ),
        default=[
            "download",
            "prepare",
            "convert",
            "evaluate",
            "benchmark",
            "diagnose",
            "report",
            "verify",
        ],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    tasks = ("cv", "text") if args.task == "all" else (args.task,)
    if "download" in args.stages:
        launch(args.task, "download")
    for task in tasks:
        if "prepare" in args.stages:
            launch(task, "prepare")
        if "convert" in args.stages:
            for variant in VARIANTS[1:]:
                launch(task, "convert", variant, required=variant in ("onnx_fp32", "tflite_fp32"))
        if "evaluate" in args.stages:
            for variant in VARIANTS:
                if converted(task, variant):
                    launch(task, "evaluate", variant)
            launch(task, "evaluate")
        if "benchmark" in args.stages:
            for variant in VARIANTS:
                if converted(task, variant):
                    launch(
                        task,
                        "benchmark",
                        variant,
                        extra=(
                            "--threads",
                            str(args.threads),
                            "--warmup",
                            str(args.warmup),
                            "--runs",
                            str(args.runs),
                        ),
                    )
        if "diagnose" in args.stages and converted(task, "tflite_static"):
            launch(task, "diagnose", "tflite_static")
    for stage in ("report", "verify"):
        if stage in args.stages:
            launch("all", stage)


if __name__ == "__main__":
    main()
