"""Record installed packages without importing model runtimes or running inference."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

DAY = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "onnx",
    "onnxruntime",
    "ai-edge-litert",
    "litert-torch",
    "ai-edge-quantizer",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DAY / "experiments/01-reproduce/results/environment-probe/environment.json",
        help="Separate inventory file; does not overwrite historical run-01 evidence",
    )
    args = parser.parse_args()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "missing_packages": [name for name, version in packages.items() if version is None],
        "peer_revision": "536b9ae1f9e1b3829ac4776a98fb05677df9b6aa",
        "scope": "Installed-package inventory only; not proof of conversion, inference or reproduction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
