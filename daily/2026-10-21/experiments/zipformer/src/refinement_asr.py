"""Regression/state and matched timing for the conservative bias fusion."""

import argparse
import gc

import numpy as np
from asr21 import ART, DAY20, RESULTS, ROOT, LiteRunner, info, now, numeric, paths, read, write
from streaming import errors, transcribe


def model_paths(variant):
    value = paths("baseline" if variant == "baseline" else "fc")
    if variant == "fc_bias":
        value["encoder"] = ART / "refinement/encoder_fc_bias.tflite"
    return value


def create(variant):
    return {name: LiteRunner(path) for name, path in model_paths(variant).items()}


def quality():
    target = RESULTS / "refinement_bias_evaluation.json"
    if target.exists():
        raise FileExistsError(target)
    rewrite = read(RESULTS / "refinement_bias_rewrite.json")
    assert rewrite["tensor_gate_passed"] and rewrite["counters_exact"]
    rows = {}
    started = now()
    for subset, manifest_path, expected in [
        (
            "73",
            DAY20 / "results/evaluation_manifest.json",
            read(RESULTS / "evaluation_fc.json")["records"],
        ),
        (
            "100",
            RESULTS / "multispeaker_manifest.json",
            read(RESULTS / "multispeaker_evaluation.json")["models"]["fc"]["records"],
        ),
    ]:
        manifest = read(manifest_path)
        runners = create("fc_bias")
        records = []
        for index, (row, prior) in enumerate(zip(manifest["records"], expected, strict=True)):
            assert row["id"] == prior["id"]
            result, encoded = transcribe(runners, ROOT / row["path"])
            assert np.isfinite(encoded).all()
            result.update(
                id=row["id"],
                reference=row["text"],
                word_errors=errors(row["text"], result["text"]),
                original_tokens_equal=result["tokens"] == prior["tokens"],
                original_counters_equal=result["final_cached_len"] == prior["final_cached_len"],
            )
            records.append(result)
            if (index + 1) % 20 == 0:
                print("BIAS REGRESSION", subset, index + 1, flush=True)
        total = {
            k: sum(r["word_errors"][k] for r in records) for k in ["S", "D", "I", "N", "errors"]
        }
        total["WER_percent"] = total["errors"] / total["N"] * 100
        rows[subset] = {
            "records": records,
            "word_errors": total,
            "manifest": info(manifest_path),
            "tokens_all_equal": all(r["original_tokens_equal"] for r in records),
            "counters_all_equal": all(r["original_counters_equal"] for r in records),
        }
        del runners
        gc.collect()
    audio = ART / "continuous_repeated.wav"
    continuous_rows, encoded_rows = {}, {}
    for variant in ["fc", "fc_bias"]:
        runners = create(variant)
        continuous_rows[variant], encoded_rows[variant] = transcribe(runners, audio)
        del runners
        gc.collect()
    a, b = (continuous_rows[v] for v in ["fc", "fc_bias"])
    parity = numeric(encoded_rows["fc"], encoded_rows["fc_bias"])
    continuous = {
        "records": continuous_rows,
        "audio": info(audio),
        "encoder_parity": parity,
        "tokens_equal": a["tokens"] == b["tokens"],
        "counters_exact": a["final_cached_len"] == b["final_cached_len"],
    }
    passed = all(v["tokens_all_equal"] and v["counters_all_equal"] for v in rows.values())
    passed = passed and continuous["tokens_equal"] and continuous["counters_exact"]
    passed = passed and parity["allclose_rtol1e_4_atol1e_4"]
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "subsets": rows,
            "continuous": continuous,
            "quality_passed": passed,
            "artifacts": {n: info(p) for n, p in model_paths("fc_bias").items()},
            "scope": "Known73/100utterance regression sets;unchanged context[0,0]/frontend/greedy. No new holdout or full-test-clean claim. Timing not benchmark.",
            "source_code": info(__file__),
        },
    )
    print("BIAS QUALITY", passed, {n: v["word_errors"] for n, v in rows.items()}, flush=True)


def stats(values):
    return {
        "samples": values,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "runs": len(values),
    }


def benchmark():
    target = RESULTS / "refinement_benchmark.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "refinement_bias_evaluation.json")["quality_passed"]
    reference = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in reference for p in row["captures"]]
    feeds = [{k: v.copy() for k, v in np.load(ROOT / p["path"]).items()} for p in captures]
    times = {v: [] for v in ["baseline", "fc", "fc_bias"]}
    rtf, pipelines = {v: [] for v in times}, {v: [] for v in times}
    started, orders = now(), []
    for cycle in range(5):
        names = list(times)
        order = names[cycle % 3 :] + names[: cycle % 3]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for variant in order:
            runners = create(variant)
            for index in range(16):
                runners["encoder"].run(feeds[index % len(feeds)])
            runners["encoder"].times.clear()
            for index in range(32):
                runners["encoder"].run(feeds[index % len(feeds)])
            times[variant].extend(runners["encoder"].times)
            if cycle < 3:
                seconds, audio = 0.0, 0.0
                for row in reference:
                    result, _ = transcribe(
                        runners, DAY20 / "artifacts/source/test_wavs" / row["name"]
                    )
                    assert result["tokens"] == row["tokens"]
                    seconds += result["pipeline_seconds"]
                    audio += result["audio_seconds"]
                rtf[variant].append(seconds / audio)
                pipelines[variant].append({"pipeline_seconds": seconds, "audio_seconds": audio})
            del runners
            gc.collect()
        print("BIAS BENCHMARK", cycle + 1, "/5", flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "orders": orders,
            "models": {
                v: {
                    "artifacts": {c: info(p) for c, p in model_paths(v).items()},
                    "encoder_ms": stats(times[v]),
                    "pipeline_RTF": stats(rtf[v]),
                    "pipeline_rounds": pipelines[v],
                    "round_medians_ms": [
                        float(np.median(times[v][i : i + 32])) for i in range(0, 160, 32)
                    ],
                }
                for v in times
            },
            "protocol": "Linux LiteRT2.1.2 CPU4threads;one pipeline at a time;5rotated/reversed rounds16encoderwarmups+32timings=160samples;first3rounds3smokeaudiosRTF. Same API/frontend/state/context. Load excluded,frontend/read/greedy included. Encoder explicitly warmed,first decoder/joiner invoke remains included equally. Profiling off,no competing experiment.",
            "source_code": info(__file__),
        },
    )
    print(
        "BIAS TIMING",
        {
            v: {"encoder": float(np.median(times[v])), "RTF": float(np.median(rtf[v]))}
            for v in times
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["quality", "benchmark"])
    args = parser.parse_args()
    {"quality": quality, "benchmark": benchmark}[args.stage]()
