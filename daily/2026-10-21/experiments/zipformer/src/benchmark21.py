"""Matched baseline/FC comparisons after all quality and state gates."""

import gc

import numpy as np
from asr21 import (
    DAY20,
    RESULTS,
    ROOT,
    create,
    info,
    now,
    paths,
    read,
    source,
    write,
    zeros_for_model,
)
from streaming import transcribe


def stats(values):
    return {
        "samples": values,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "runs": len(values),
    }


def main():
    target = RESULTS / "benchmark.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "continuous_state_test.json")["tokens_equal"]
    assert read(RESULTS / "evaluation_fc.json")["word_errors"]["errors"] == 91
    reference = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in reference for p in row["captures"]]
    feeds = [{k: v.copy() for k, v in np.load(ROOT / p["path"]).items()} for p in captures]
    zeros_for_model(source("encoder"))
    times = {v: [] for v in ["baseline", "fc"]}
    rtf = {v: [] for v in times}
    pipelines = {v: [] for v in times}
    started, orders = now(), []
    for cycle in range(5):
        order = list(times) if cycle % 2 == 0 else list(times)[::-1]
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
        print("ASR BENCHMARK", cycle + 1, "/5", flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "orders": orders,
            "models": {
                v: {
                    "artifacts": {c: info(p) for c, p in paths(v).items()},
                    "encoder_ms": stats(times[v]),
                    "pipeline_RTF": stats(rtf[v]),
                    "pipeline_rounds": pipelines[v],
                }
                for v in times
            },
            "protocol": "Linux LiteRT2.1.2 CPU4threads,batch1;baseline day20encoder vs FC encoder,identical decoder/joiner/frontend/state rules.5alternating-order rounds,16encoderwarmups+32timings=160samples;first3rounds also entire3smoke audio pipeline. Model/metadata load excluded,frontend/file reading/greedy included in RTF;no profiling/competing workload.",
            "source_code": info(__file__),
        },
    )
    print(
        "RESULT",
        {
            v: {"encoder_ms": float(np.median(times[v])), "RTF": float(np.median(rtf[v]))}
            for v in times
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
