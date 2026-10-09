"""Create or check an isolated, version-pinned environment using uv."""

import argparse
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "baseline": ROOT / ".venv",
    "runtime": ROOT / ".venv-day09",
    "gpu": ROOT / "quantization/.venv-gpu",
    "mobilenet": ROOT / ".venv-mobilenet",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="baseline")
    parser.add_argument("--offline", action="store_true", help="Use installed packages/cache only")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="Check installed dependencies only")
    actions.add_argument("--dry-run", action="store_true", help="Show sync plan without installing")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    if platform.system() != "Windows":
        parser.error(
            "These captured locks target Windows x64; use environments/peer-linux for Linux"
        )

    environment = PROFILES[args.profile]
    interpreter = environment / "Scripts/python.exe"
    command = [uv, "--cache-dir", str(ROOT / ".cache/uv")]
    if args.offline:
        command.append("--offline")
    if not interpreter.is_file():
        if args.check or args.dry_run:
            parser.error(
                f"Environment does not exist: {environment}. Run setup without the check flags"
            )
        subprocess.run([*command, "venv", "--python", "3.11", str(environment)], check=True)

    if args.check:
        subprocess.run([*command, "pip", "check", "--python", str(interpreter)], check=True)
    else:
        lock = ROOT / "environments" / args.profile / "requirements-lock.txt"
        sync = [*command, "pip", "sync", "--python", str(interpreter), str(lock), "--strict"]
        if args.dry_run:
            sync.append("--dry-run")
        subprocess.run(sync, check=True)
    print(f"{args.profile}: {interpreter}")


if __name__ == "__main__":
    main()
