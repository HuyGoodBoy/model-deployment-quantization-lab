"""Additional frozen multi-speaker quality confirmation, never optimization feedback."""

import gc

import numpy as np
from asr21 import RESULTS, ROOT, create, info, now, paths, read, write
from streaming import errors, transcribe


def main():
    target = RESULTS / "multispeaker_evaluation.json"
    if target.exists():
        raise FileExistsError(target)
    manifest_path = RESULTS / "multispeaker_manifest.json"
    manifest = read(manifest_path)
    models = {}
    started = now()
    for variant in ["baseline", "fc"]:
        runners = create(variant)
        rows = []
        for index, row in enumerate(manifest["records"]):
            result, encoded = transcribe(runners, ROOT / row["path"])
            assert np.isfinite(encoded).all()
            result.update(
                id=row["id"],
                speaker_id=row["speaker_id"],
                reference=row["text"],
                word_errors=errors(row["text"], result["text"]),
            )
            rows.append(result)
            if (index + 1) % 20 == 0:
                print("MULTISPEAKER", variant, index + 1, "/100", flush=True)
        total = {k: sum(r["word_errors"][k] for r in rows) for k in ["S", "D", "I", "N", "errors"]}
        total["WER_percent"] = total["errors"] / total["N"] * 100
        models[variant] = {
            "records": rows,
            "word_errors": total,
            "artifacts": {c: info(p) for c, p in paths(variant).items()},
        }
        del runners
        gc.collect()
    for a, b in zip(models["baseline"]["records"], models["fc"]["records"], strict=True):
        assert a["id"] == b["id"] and a["tokens"] == b["tokens"]
        assert a["final_cached_len"] == b["final_cached_len"]
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "manifest": info(manifest_path),
            "models": models,
            "utterances": 100,
            "speakers": 20,
            "tokens_all_equal": True,
            "final_counters_exact": True,
            "scope": "Frozen subset,after patch completed,no tuning/selection from results;not full test-clean WER. Pipeline unchanged from73utterance tests. Timing not used as benchmark.",
            "source_code": info(__file__),
        },
    )
    print("MULTISPEAKER RESULT", {n: r["word_errors"] for n, r in models.items()}, flush=True)


if __name__ == "__main__":
    main()
