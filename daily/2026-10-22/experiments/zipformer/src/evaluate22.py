"""Exact wrapper parity, reused utterance regression and continuous ten-minute state."""

import argparse
import gc

import numpy as np
import soundfile as sf
from asr22 import ART, DAY20, DAY21, RESULTS, ROOT, info, now, numeric, paths, read, write
from runtime22 import create
from streaming import errors, transcribe


def fixtures():
    target = RESULTS / "fixture_parity.json"
    if target.exists():
        raise FileExistsError(target)
    reference = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in reference for p in row["captures"]]
    runners = {v: create(v) for v in ["Z0", "Z1"]}
    encoder_rows, decoder_rows, joiner_rows = [], [], []
    rng = np.random.default_rng(20261022)
    contexts = [
        np.array([[0, 0]], np.int64),
        *[rng.integers(0, 500, size=(1, 2), dtype=np.int64) for _ in range(17)],
    ]
    decoder_outputs = {}
    for index, context in enumerate(contexts):
        values = {v: r["decoder"].run({"y": context})["decoder_out"] for v, r in runners.items()}
        parity = numeric(values["Z0"], values["Z1"])
        decoder_rows.append({"context": context.tolist(), "parity": parity})
        decoder_outputs[index] = values
    for index, capture in enumerate(captures):
        with np.load(ROOT / capture["path"]) as saved:
            feed = {k: v.copy() for k, v in saved.items()}
        values = {v: r["encoder"].run(feed) for v, r in runners.items()}
        parity = {k: numeric(values["Z0"][k], values["Z1"][k]) for k in values["Z0"]}
        encoder_rows.append({"input": capture, "outputs": parity})
        for frame in range(2):
            logits = {
                v: r["joiner"].run(
                    {
                        "encoder_out": values[v]["encoder_out"][0, frame][None],
                        "decoder_out": decoder_outputs[index % 18][v],
                    }
                )["logit"]
                for v, r in runners.items()
            }
            joiner_rows.append(
                {"capture": index, "frame": frame, "parity": numeric(logits["Z0"], logits["Z1"])}
            )
    passed = all(v["exact"] for row in encoder_rows for v in row["outputs"].values()) and all(
        row["parity"]["exact"] for row in decoder_rows + joiner_rows
    )
    write(
        target,
        {
            "finished_at_utc": now(),
            "encoder": encoder_rows,
            "decoder": decoder_rows,
            "joiner": joiner_rows,
            "exact_passed": passed,
            "model_files": {n: info(p) for n, p in paths("fc").items()},
            "source_code": info(__file__),
        },
    )
    print(
        "WRAPPER FIXTURES",
        len(encoder_rows),
        len(decoder_rows),
        len(joiner_rows),
        "exact",
        passed,
        flush=True,
    )


def regression():
    target = RESULTS / "regression.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "fixture_parity.json")["exact_passed"]
    started, subsets = now(), {}
    references = [
        (
            "73",
            DAY20 / "results/evaluation_manifest.json",
            read(DAY21 / "results/evaluation_fc.json")["records"],
        ),
        (
            "100",
            DAY21 / "results/multispeaker_manifest.json",
            read(DAY21 / "results/multispeaker_evaluation.json")["models"]["fc"]["records"],
        ),
    ]
    for name, manifest_path, expected in references:
        manifest = read(manifest_path)
        runner = create("Z1")
        rows = []
        for index, (record, prior) in enumerate(zip(manifest["records"], expected, strict=True)):
            assert record["id"] == prior["id"]
            value, encoded = transcribe(runner, ROOT / record["path"])
            assert np.isfinite(encoded).all()
            value.update(
                id=record["id"],
                reference=record["text"],
                word_errors=errors(record["text"], value["text"]),
                original_tokens_equal=value["tokens"] == prior["tokens"],
                original_counters_equal=value["final_cached_len"] == prior["final_cached_len"],
            )
            rows.append(value)
            if (index + 1) % 20 == 0:
                print("DIRECT REGRESSION", name, index + 1, flush=True)
        total = {k: sum(r["word_errors"][k] for r in rows) for k in ["S", "D", "I", "N", "errors"]}
        total["WER_percent"] = total["errors"] / total["N"] * 100
        subsets[name] = {
            "records": rows,
            "word_errors": total,
            "tokens_all_equal": all(r["original_tokens_equal"] for r in rows),
            "counters_all_equal": all(r["original_counters_equal"] for r in rows),
            "manifest": info(manifest_path),
        }
        del runner
        gc.collect()
    passed = all(r["tokens_all_equal"] and r["counters_all_equal"] for r in subsets.values())
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "subsets": subsets,
            "quality_passed": passed,
            "scope": "Known173utterance regression;not fresh WER holdout. Exact token/counter gate before timing.",
            "source_code": info(__file__),
        },
    )
    print("DIRECT QUALITY", passed, {k: v["word_errors"] for k, v in subsets.items()}, flush=True)


def continuous():
    target = RESULTS / "continuous_state.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "fixture_parity.json")["exact_passed"]
    source = DAY20 / "artifacts/source/test_wavs/0.wav"
    waveform, rate = sf.read(source, dtype="float32", always_2d=True)
    assert rate == 16000
    samples = rate * 600
    repeats = (samples + len(waveform) - 1) // len(waveform)
    repeated = np.tile(waveform[:, 0], repeats)[:samples]
    ART.mkdir(parents=True, exist_ok=True)
    audio = ART / "continuous_600_seconds.wav"
    sf.write(audio, repeated, rate, subtype="FLOAT")
    started, rows, encoded = now(), {}, {}
    for variant in ["Z0", "Z1"]:
        runner = create(variant)
        rows[variant], encoded[variant] = transcribe(runner, audio)
        assert np.isfinite(encoded[variant]).all()
        del runner
        gc.collect()
        print("CONTINUOUS", variant, rows[variant]["chunks"], "chunks", flush=True)
    a, b = rows["Z0"], rows["Z1"]
    parity = numeric(encoded["Z0"], encoded["Z1"])
    passed = (
        a["tokens"] == b["tokens"]
        and a["final_cached_len"] == b["final_cached_len"]
        and parity["exact"]
    )
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "source_audio": info(source),
            "audio": info(audio),
            "audio_seconds": 600,
            "records": rows,
            "encoded_parity": parity,
            "exact_passed": passed,
            "scope": "Uninterrupted synthetic600second waveform;each wrapper starts from zero and independently carries state/context throughout. Not concatenated per-utterance transcripts,new WER corpus or universal INT32 overflow test.",
            "source_code": info(__file__),
        },
    )
    print("CONTINUOUS EXACT", passed, parity, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fixtures", "regression", "continuous"])
    {"fixtures": fixtures, "regression": regression, "continuous": continuous}[
        parser.parse_args().stage
    ]()
