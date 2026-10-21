"""Day21 streaming paths; read-only reuse of verified day20 runtime wrappers."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
DAY20 = ROOT / "daily/2026-10-20/experiments/zipformer"
ART = TASK / "artifacts"
RESULTS = TASK / "results"
sys.path.append(str(DAY20 / "src"))

# isort: split
from support import (  # noqa: F401 -- runtime API re-exports
    LiteRunner,
    OnnxRunner,
    numeric,
    source,
    zeros_for_model,
)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(path):
    path = Path(path)
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def paths(variant):
    return {
        "encoder": DAY20 / "artifacts/conversion/static/encoder/output/input_float32.tflite"
        if variant == "baseline"
        else ART / "encoder_fc.tflite",
        "decoder": DAY20 / "artifacts/conversion/conv2d/decoder/output/input_float32.tflite",
        "joiner": DAY20 / "artifacts/conversion/static/joiner/output/input_float32.tflite",
    }


def create(variant):
    return {name: LiteRunner(path) for name, path in paths(variant).items()}
