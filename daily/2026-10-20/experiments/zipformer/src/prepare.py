"""Pin and download the requested public Zipformer artifacts, without editing them."""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
ART = TASK / "artifacts"
RESULTS = TASK / "results"
REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
REVISION = "d42f2d9f7ca24806fb667456a18a9f1b60f70d16"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    target = RESULTS / "source_manifest.json"
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        for item in data["files"]:
            assert sha(ROOT / item["path"]) == item["sha256"]
        print("Existing source hashes verified")
        return
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60
    ) as response:
        data = json.load(response)
    assert data["sha"] == REVISION
    records = []
    for item in data["siblings"]:
        name = item["rfilename"]
        if ".int8." in name or not (
            name.endswith((".onnx", ".wav", ".txt", ".sh")) or name == "README.md"
        ):
            continue
        path = ART / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        expected = item.get("lfs", {}).get("sha256")
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            partial.replace(path)
        digest = sha(path)
        assert not expected or digest == expected, name
        assert path.stat().st_size == item["size"], name
        records.append(
            {
                "name": name,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "url": url,
                "upstream_lfs_sha256": expected,
            }
        )
        print("PREPARED", name, path.stat().st_size, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "locked_at_utc": datetime.now(UTC).isoformat(),
                "repo": REPO,
                "revision": REVISION,
                "files": records,
                "source_code_sha256": sha(__file__),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
