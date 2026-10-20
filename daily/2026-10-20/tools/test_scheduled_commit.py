"""Integration checks in an isolated disposable repo; never commit the real repo."""

import importlib.util
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    spec = importlib.util.spec_from_file_location(
        "scheduled", Path(__file__).with_name("scheduled_commit.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class AfterDeadline(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 20, 10, 0, 1, tzinfo=UTC)

    module.datetime = AfterDeadline
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="schedule-guard-test-", dir=cache) as temporary:
        folder = Path(temporary).resolve()
        assert folder.is_relative_to(cache.resolve())
        module.ROOT = folder
        module.STATE = folder / "state"

        def git(*args):
            return subprocess.run(
                ["git", *args], cwd=folder, capture_output=True, text=True, check=True
            ).stdout.strip()

        git("init", "-b", "main")
        git("config", "user.name", "Guard Test")
        git("config", "user.email", "guard-test@example.invalid")
        (folder / "README.md").write_text("baseline\n")
        git("add", "README.md")
        git("commit", "-m", "baseline")
        parent = git("rev-parse", "HEAD")
        module.main()
        result = json.loads((module.STATE / "result.json").read_text())
        assert result["status"] == "skipped" and git("rev-parse", "HEAD") == parent
        (folder / "README.md").write_text("reviewed\n")
        git("add", "README.md")
        marker = {
            "parent": parent,
            "branch": "main",
            "tree": "wrong-tree",
            "astra_reviewed": True,
            "validation_passed": True,
        }
        (module.STATE / "ready.json").write_text(json.dumps(marker))
        module.main()
        result = json.loads((module.STATE / "result.json").read_text())
        assert result["status"] == "skipped" and git("rev-parse", "HEAD") == parent
        marker["tree"] = git("write-tree")
        (module.STATE / "ready.json").write_text(json.dumps(marker))
        (folder / "README.md").write_text("changed after review\n")
        module.main()
        result = json.loads((module.STATE / "result.json").read_text())
        assert result["status"] == "skipped" and git("rev-parse", "HEAD") == parent
        (folder / "README.md").write_text("reviewed\n")
        module.main()
        result = json.loads((module.STATE / "result.json").read_text())
        assert result["status"] == "committed" and result["push"] is False
        assert git("log", "-1", "--format=%s") == "20/10"
        assert git("rev-parse", "HEAD") != parent
        committed = git("rev-parse", "HEAD")
        module.main()
        result = json.loads((module.STATE / "result.json").read_text())
        assert result["status"] == "skipped" and git("rev-parse", "HEAD") == committed
    output = Path(__file__).with_name("schedule_guard_verification.json")
    output.write_text(
        json.dumps(
            {
                "verified_at_utc": datetime.now(UTC).isoformat(),
                "isolated_repo_only": True,
                "checks": [
                    "missing marker skips",
                    "changed index skips",
                    "unstaged tracked change skips",
                    "reviewed snapshot commits once with20/10 and no push",
                ],
                "passed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("SCHEDULE GUARDS PASSED; real repository untouched")


if __name__ == "__main__":
    main()
