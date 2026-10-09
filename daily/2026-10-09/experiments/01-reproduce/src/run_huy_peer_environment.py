"""Measure Huy's unchanged models in the fully installed peer environment.

Run sequentially in the Linux CPU image after peer conversions have finished.
The 30/200 protocol intentionally matches Huy's existing Windows matrix.
"""

import json
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[1]


def main():
    run_id = "run-04-huy-linux"
    folder = TASK / "results" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for task in ("cv", "text"):
        for variant in (
            "onnx_fp32",
            "onnx_dynamic",
            "onnx_static",
            "tflite_fp32",
            "tflite_dynamic",
            "tflite_static",
        ):
            command = [
                sys.executable,
                str(Path(__file__).with_name("runtime_compare.py")),
                "--task",
                task,
                "--variant",
                variant,
                "--runtime",
                "new",
                "--threads",
                "4",
                "--protocol",
                "matched",
                "--warmup",
                "30",
                "--runs",
                "200",
                "--run-id",
                run_id,
            ]
            start = time.perf_counter()
            with (folder / f"{task}_{variant}.log").open("w", encoding="utf-8") as log:
                p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            record = {
                "task": task,
                "variant": variant,
                "command": command,
                "returncode": p.returncode,
                "seconds": time.perf_counter() - start,
            }
            records.append(record)
            (folder / "execution.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"{task}/{variant}: exit {p.returncode}", flush=True)
    return int(any(r["returncode"] for r in records))


if __name__ == "__main__":
    sys.exit(main())
