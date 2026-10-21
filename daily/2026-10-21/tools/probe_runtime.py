"""Read-only runtime/tool inventory for the refinement profiler design."""

import importlib.metadata
import inspect
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import ai_edge_litert
import ai_edge_quantizer
from ai_edge_litert.interpreter import Interpreter

litert = Path(ai_edge_litert.__file__).parent
quant = Path(ai_edge_quantizer.__file__).parent
inventory = {
    "recorded_at_utc": datetime.now(UTC).isoformat(),
    "versions": {n: importlib.metadata.version(n) for n in ["ai-edge-litert", "ai-edge-quantizer"]},
    "interpreter_signature": str(inspect.signature(Interpreter)),
    "litert_files": [
        str(p.relative_to(litert))
        for p in litert.rglob("*")
        if p.is_file() and any(t in p.name.lower() for t in ["profil", "benchmark", ".so"])
    ],
    "quantizer_profile_files": [
        str(p.relative_to(quant))
        for p in quant.rglob("*")
        if p.is_file() and any(t in p.name.lower() for t in ["profil", "calibrat"])
    ],
    "tools": {n: shutil.which(n) for n in ["nm", "c++", "cmake", "bazel", "perf", "taskset"]},
    "python_profiler_methods": [n for n in dir(Interpreter) if "profil" in n.lower()],
    "scope": "Read-only pinned-image inventory;no supported per-op result API exposed by Interpreter. No custom runtime build or private pointer hook used;application-stage diagnostic is not native per-node profiling.",
}
root = Path(__file__).resolve().parents[3]
target = root / "daily/2026-10-21/environments/refinement_profiler_inventory.json"
assert not target.exists()
target.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(inventory, indent=2))
