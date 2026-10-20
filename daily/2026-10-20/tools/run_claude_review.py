"""Record actual Claude CLI execution and tool evidence; never substitute another AI."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-20"


def main():
    executable = Path.home() / ".local/bin/claude.exe"
    target = DAY / "experiments/zipformer/results/claude_execution.json"
    if target.exists():
        raise FileExistsError(target)
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=True
    )
    auth = subprocess.run(
        [str(executable), "auth", "status"], capture_output=True, text=True, check=False
    )
    prompt = (DAY / "tools/claude_prompt.txt").read_text(encoding="utf-8")
    cmd = [
        str(executable),
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "plan",
        "--tools",
        "Read,Grep,Glob",
        "--no-session-persistence",
    ]
    run = subprocess.run(
        cmd,
        input=prompt,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    record = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "version": version.stdout.strip(),
        "auth_status": json.loads(auth.stdout),
        "command": cmd,
        "prompt": prompt,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "attribution": "Existing experiment and rewrites are Codex-authored. No Claude contribution is claimed unless this response contains an actual authenticated model execution.",
    }
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("CLAUDE", run.returncode, run.stdout[:2000], run.stderr[:500])


if __name__ == "__main__":
    main()
