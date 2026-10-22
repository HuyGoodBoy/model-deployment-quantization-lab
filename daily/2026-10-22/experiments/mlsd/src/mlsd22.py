"""Day22 experiment paths and read-only reuse of day21 data/metric utilities."""

import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
DAY21 = ROOT / "daily/2026-10-21/experiments/mlsd"
ART = TASK / "artifacts"
RESULTS = TASK / "results"
sys.path.append(str(DAY21 / "src"))

from support21 import checked, info, now, pilot_data, read, sha, write  # noqa: F401


def references():
    old = read(DAY21 / "results/exports.json")["models"]
    return {
        "float": old["float"]["model"],
        "C1": read(DAY21 / "results/refinement_C1.json")["model"],
    }


def save(name, values):
    import numpy as np

    path = ART / (name + ".npz")
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, output=values)
    return info(path)
