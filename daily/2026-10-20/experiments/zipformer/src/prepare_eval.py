"""Lock all public dummy LibriSpeech utterances before evaluating conversion patches."""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"


def main():
    target = TASK / "results/evaluation_manifest.json"
    if target.exists():
        print("Preserved locked evaluation manifest")
        return
    folder = TASK / "artifacts/evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    name = "clean/validation-00000-of-00001.parquet"
    url = f"https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/{REVISION}/{name}"
    path = folder / "source.parquet"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda row: row["id"])
    records = []
    for row in rows:
        audio = row["audio"]
        assert audio["bytes"]
        suffix = Path(audio["path"]).suffix or ".flac"
        output = folder / (row["id"] + suffix)
        output.write_bytes(audio["bytes"])
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "speaker_id": row.get("speaker_id"),
                "chapter_id": row.get("chapter_id"),
                "path": output.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(audio["bytes"]).hexdigest(),
                "bytes": len(audio["bytes"]),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "locked_at_utc": datetime.now(UTC).isoformat(),
                "dataset": "hf-internal-testing/librispeech_asr_dummy clean validation",
                "revision": REVISION,
                "selection": "All rows sorted by ID, no model-based filtering",
                "utterances": len(records),
                "speaker_count": len({x["speaker_id"] for x in records}),
                "parquet_url": url,
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "records": records,
                "scope": "Small public test subset; WER valid only here, not full LibriSpeech benchmark or proof of unseen pretraining data",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("LOCKED", len(records), "utterances", len({x["speaker_id"] for x in records}), "speakers")


if __name__ == "__main__":
    main()
