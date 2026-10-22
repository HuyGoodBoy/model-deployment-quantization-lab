"""Capture the actual pinned container, packages and profiler prerequisites."""

import argparse
import importlib.metadata as md
import inspect
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter


def main(name):
    day = Path(__file__).resolve().parents[1]
    target = day / "environments" / (name + ".json")
    assert not target.exists()
    packages = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    prior = day.parent / "2026-10-21/environments" / (name + "-requirements-lock.txt")
    normalized = lambda text: sorted(x.strip() for x in text.splitlines() if x.strip())
    cpu = Path("/proc/cpuinfo").read_text()
    value = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": sorted(
            {x.split(":", 1)[1].strip() for x in cpu.splitlines() if x.startswith("model name")}
        ),
        "litert": md.version("ai-edge-litert"),
        "packages_match_day21": normalized(packages) == normalized(prior.read_text()),
        "interpreter_signature": str(inspect.signature(Interpreter)),
        "public_profiler_methods": [
            n for n in dir(Interpreter) if "profil" in n.lower() and not n.startswith("_")
        ],
        "build_tools": {
            n: shutil.which(n) for n in ["c++", "cmake", "bazel", "nm", "perf", "benchmark_model"]
        },
        "scope": "Read-only environment inventory; no native profiler build or kernel measurement performed.",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")
    (target.parent / (name + "-requirements-lock.txt")).write_text(packages)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["mlsd", "zipformer"])
    main(parser.parse_args().name)
