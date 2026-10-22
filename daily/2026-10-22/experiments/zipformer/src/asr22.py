"""Day22 paths; graph artifacts are read-only references to day20/day21."""

import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
DAY21 = ROOT / "daily/2026-10-21/experiments/zipformer"
ART = TASK / "artifacts"
RESULTS = TASK / "results"
sys.path.append(str(DAY21 / "src"))

from asr21 import (  # noqa: F401
    DAY20,
    LiteRunner,
    info,
    now,
    numeric,
    paths,
    read,
    sha,
    source,
    write,
    zeros_for_model,
)
