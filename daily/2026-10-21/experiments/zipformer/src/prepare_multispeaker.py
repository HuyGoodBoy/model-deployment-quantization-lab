"""Lock a deterministic20speaker100utterance test-clean subset before inference."""

import hashlib
import random
import re
import tarfile
import time
import urllib.request
from collections import defaultdict

from asr21 import ART, DAY20, RESULTS, ROOT, info, now, read, write

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
EXPECTED_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"


def main():
    target = RESULTS / "multispeaker_manifest.json"
    if target.exists():
        raise FileExistsError(target)
    archive = ROOT / ".cache/librispeech/test-clean.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".gz.partial")
        started = time.monotonic()
        last = started
        downloaded = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            URL, headers={"Range": f"bytes={downloaded}-"} if downloaded else {}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if downloaded and response.status == 206:
                assert response.headers["Content-Range"].startswith(f"bytes {downloaded}-")
                mode = "ab"
            else:
                downloaded, mode = 0, "wb"
            with partial.open(mode) as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Dataset download budget exceeded")
                    stream.write(block)
                    downloaded += len(block)
                    if time.monotonic() - last > 15:
                        print("DOWNLOAD", round(downloaded / 1e6, 1), "MB", flush=True)
                        last = time.monotonic()
        partial.replace(archive)
    md5 = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
    assert md5.hexdigest() == EXPECTED_MD5
    old = read(DAY20 / "results/evaluation_manifest.json")
    excluded_speakers = {int(r["speaker_id"]) for r in old["records"]}
    excluded_ids = {r["id"] for r in old["records"]}
    transcripts = {}
    grouped = defaultdict(list)
    audio_paths = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".trans.txt"):
                for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                    identity, text = line.split(" ", 1)
                    transcripts[identity] = text
            elif member.isfile() and member.name.endswith(".flac"):
                identity = member.name.rsplit("/", 1)[-1][:-5]
                assert re.fullmatch(r"\d+-\d+-\d+", identity)
                speaker = int(identity.split("-")[0])
                if speaker not in excluded_speakers and identity not in excluded_ids:
                    grouped[speaker].append(identity)
                    audio_paths[identity] = member.name
        rng = random.Random(20261021)
        speakers = sorted(rng.sample(sorted(grouped), 20))
        selected = sorted(
            identity for speaker in speakers for identity in rng.sample(sorted(grouped[speaker]), 5)
        )
        assert len(selected) == 100 and not set(selected) & excluded_ids
        folder = ART / "multispeaker"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        ordered = sorted(
            selected, key=lambda identity: tar.getmember(audio_paths[identity]).offset_data
        )
        for identity in ordered:
            path = folder / (identity + ".flac")
            assert path.resolve().is_relative_to(folder.resolve())
            path.write_bytes(tar.extractfile(audio_paths[identity]).read())
            rows.append(
                {
                    "id": identity,
                    "speaker_id": int(identity.split("-")[0]),
                    "text": transcripts[identity],
                    **info(path),
                }
            )
    rows.sort(key=lambda row: row["id"])
    write(
        target,
        {
            "locked_at_utc": now(),
            "dataset": "LibriSpeech test-clean subset from official OpenSLR12",
            "source_url": URL,
            "publisher_MD5": EXPECTED_MD5,
            "archive": info(archive),
            "checksum_source": "https://www.openslr.org/resources/12/md5sum.txt",
            "license": "CC BY4.0",
            "seed": 20261021,
            "selection": "Random20speakers then5utterances/speaker;fixed after patch, before evaluation;exclude previous speaker/IDs",
            "speakers": speakers,
            "speaker_count": 20,
            "utterances": 100,
            "records": rows,
            "no_overlap_with_previous73_ids": True,
            "no_overlap_with_previous73_speakers": True,
            "reference_words": sum(
                len(re.sub(r"[^A-Z' ]", " ", r["text"].upper()).split()) for r in rows
            ),
            "scope": "New to these experiments, not proof of unseen pretraining;100utterance subset,not complete test-clean benchmark. No model-based filtering or patch tuning on this set.",
            "source_code": info(__file__),
        },
    )
    print("LOCKED MULTISPEAKER", len(rows), "utterances", len(speakers), "speakers", flush=True)


if __name__ == "__main__":
    main()
