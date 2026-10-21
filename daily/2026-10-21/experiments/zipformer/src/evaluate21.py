"""Same73utterance pipeline and a continuous repeated-audio state test."""

import argparse
import gc

import numpy as np
import soundfile as sf
from asr21 import ART, DAY20, RESULTS, ROOT, create, info, now, numeric, paths, read, write
from streaming import errors, transcribe


def main(variant):
    target = RESULTS / ("evaluation_" + variant + ".json")
    if target.exists():
        raise FileExistsError(target)
    runners = create(variant)
    reference = read(DAY20 / "results/evaluation_onnx.json")
    manifest_path = DAY20 / "results/evaluation_manifest.json"
    manifest = read(manifest_path)
    started, records = now(), []
    for index, (row, expected) in enumerate(
        zip(manifest["records"], reference["records"], strict=True)
    ):
        assert row["id"] == expected["id"]
        result, _ = transcribe(runners, ROOT / row["path"])
        result.update(
            id=row["id"],
            reference=row["text"],
            word_errors=errors(row["text"], result["text"]),
            original_tokens_equal=result["tokens"] == expected["tokens"],
            original_final_counters_equal=result["final_cached_len"]
            == expected["final_cached_len"],
        )
        assert result["original_tokens_equal"] and result["original_final_counters_equal"]
        records.append(result)
        if (index + 1) % 10 == 0:
            print("WER", variant, index + 1, "/73", flush=True)
    total = {k: sum(r["word_errors"][k] for r in records) for k in ["S", "D", "I", "N", "errors"]}
    total["WER_percent"] = total["errors"] / total["N"] * 100
    assert total["errors"] == 91 and total["N"] == 1150
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "variant": variant,
            "utterances": len(records),
            "models": {name: info(path) for name, path in paths(variant).items()},
            "manifest": info(manifest_path),
            "word_errors": total,
            "records": records,
            "scope": "Same locked73public dummy utterances,one speaker;shared frontend/context[0,0]/greedy/state rules. Runtime observed time is not the matched benchmark. No claim of full LibriSpeech WER.",
            "source_code": info(__file__),
        },
    )
    print("RESULT", variant, total, flush=True)


def continuous():
    target = RESULTS / "continuous_state_test.json"
    if target.exists():
        raise FileExistsError(target)
    source = DAY20 / "artifacts/source/test_wavs/0.wav"
    waveform, rate = sf.read(source, dtype="float32", always_2d=True)
    assert rate == 16000
    repeat = 12
    repeated = np.tile(waveform[:, 0], repeat)
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "continuous_repeated.wav"
    sf.write(path, repeated, rate, subtype="FLOAT")
    rows, encoded = {}, {}
    for variant in ["baseline", "fc"]:
        runners = create(variant)
        result, tensor = transcribe(runners, path)
        assert np.isfinite(tensor).all()
        rows[variant], encoded[variant] = result, tensor
        del runners
        gc.collect()
    assert rows["baseline"]["tokens"] == rows["fc"]["tokens"]
    assert rows["baseline"]["final_cached_len"] == rows["fc"]["final_cached_len"]
    parity = numeric(encoded["baseline"], encoded["fc"])
    assert parity["allclose_rtol1e_4_atol1e_4"]
    write(
        target,
        {
            "finished_at_utc": now(),
            "source_audio": info(source),
            "repeat_count": repeat,
            "input_audio": info(path),
            "audio_seconds": len(repeated) / rate,
            "records": rows,
            "encoder_parity": parity,
            "tokens_equal": True,
            "counters_exact": True,
            "scope": "One uninterrupted synthetic repetition of an existing utterance,no reset at repetition boundaries;state stability test,not new-speaker WER or INT32-overflow safety proof.",
            "source_code": info(__file__),
        },
    )
    print(
        "CONTINUOUS PASS",
        len(repeated) / rate,
        "seconds",
        rows["fc"]["chunks"],
        "chunks",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["baseline", "fc", "continuous"])
    args = parser.parse_args()
    continuous() if args.variant == "continuous" else main(args.variant)
