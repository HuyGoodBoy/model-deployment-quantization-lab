"""Commit the reviewed, already-staged snapshot once; never stage or push here."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / ".cache/scheduled-commit-2026-10-20"


def git(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    log = {"started_at_utc": datetime.now(UTC).isoformat(), "status": "skipped"}
    try:
        ready_path = STATE / "ready.json"
        if not ready_path.exists():
            raise RuntimeError("No reviewed snapshot ready; no commit made")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        assert ready["astra_reviewed"] and ready["validation_passed"]
        assert datetime.now(UTC) >= datetime.fromisoformat("2026-10-20T10:00:00+00:00")
        assert git("rev-parse", "HEAD") == ready["parent"], "HEAD changed after review"
        assert git("branch", "--show-current") == ready["branch"], "Branch changed after review"
        assert git("write-tree") == ready["tree"], "Staged snapshot changed after review"
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False)
        assert unstaged.returncode == 0, "Tracked working files changed after review"
        paths = git("diff", "--cached", "--name-only").splitlines()
        assert paths and all(p == "README.md" or p.startswith("daily/2026-10-20/") for p in paths)
        log["commit_output"] = git("commit", "-m", "20/10")
        log.update(status="committed", commit=git("rev-parse", "HEAD"), push=False)
    except Exception as error:  # noqa: BLE001 -- fail closed and log any scheduled-run failure
        log["reason"] = str(error)
    log["finished_at_utc"] = datetime.now(UTC).isoformat()
    (STATE / "result.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
