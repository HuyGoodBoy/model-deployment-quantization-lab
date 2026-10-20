"""Save exact installed packages and image identities without container secrets."""

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "daily/2026-10-20/environments"


def command(args):
    return subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    ).stdout.strip()


def main():
    OUTPUT.mkdir(exist_ok=True)
    records = {}
    for name, image in {
        "zipformer": "huy-day20-zipformer:onnx2tf269",
        "mlsd": "huy-peer-day09:cpu",
    }.items():
        details = json.loads(command(["docker", "image", "inspect", image]))[0]
        packages = command(
            ["docker", "run", "--rm", "--network", "none", image, "python", "-m", "pip", "freeze"]
        )
        (OUTPUT / (name + "-requirements-lock.txt")).write_text(
            packages + "\n", encoding="utf-8", newline="\n"
        )
        system = json.loads(
            command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    image,
                    "python",
                    "-c",
                    "import json,platform,os; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),'cpu_model':[x.split(':',1)[1].strip() for x in open('/proc/cpuinfo') if x.startswith('model name')][:1]}))",
                ]
            )
        )
        records[name] = {
            "image": image,
            "image_id": details["Id"],
            "repo_digests": details["RepoDigests"],
            **system,
        }
    result = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host_platform": platform.platform(),
        "host_python": platform.python_version(),
        "environments": records,
    }
    (OUTPUT / "runtime.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
