"""Isolated day21 paths; inherited checkpoints and data are read-only."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
DAY20 = ROOT / "daily/2026-10-20/experiments/mlsd"
ART = TASK / "artifacts"
RESULTS = TASK / "results"
sys.path.append(str(DAY20 / "src"))


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(record):
    path = ROOT / record["path"]
    assert sha(path) == record["sha256"], path
    return path


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


def inherited():
    return read(DAY20 / "results/aeq_exports.json")["models"]


def pilot_data():
    from mlsd_support import pilot

    return pilot, pilot.load_datasets("mlsd")
