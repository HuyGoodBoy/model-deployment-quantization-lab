"""Rotated matched CPU benchmarks; no conversion/profiling in timed regions."""

import gc

import numpy as np
from evaluate_asr import create, paths_for
from streaming import transcribe
from support import RESULTS, ROOT, SOURCE, info, now, numeric, read, source, write, zeros_for_model


def stats(values):
    return {
        "samples": values,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "runs": len(values),
    }


def main():
    target = RESULTS / "matched_benchmark_v2.json"
    if target.exists():
        raise FileExistsError(target)
    variants = ["onnx", "fixed", "optimized"]
    baseline = read(RESULTS / "onnx_baseline.json")
    captures = [p for r in baseline["smoke_audio"] for p in r["captures"]]
    feeds = [{k: v.copy() for k, v in np.load(ROOT / item["path"]).items()} for item in captures]
    evaluations = {v: read(RESULTS / f"evaluation_{v}.json") for v in variants}
    agreement = {}
    for v in variants[1:]:
        pairs = list(zip(evaluations["onnx"]["records"], evaluations[v]["records"], strict=True))
        assert all(
            a["id"] == b["id"]
            and a["tokens"] == b["tokens"]
            and a["final_cached_len"] == b["final_cached_len"]
            for a, b in pairs
        )
        agreement[v] = {
            "utterances": len(pairs),
            "all_tokens_equal": True,
            "all_final_cached_lengths_equal": True,
        }
    ref, candidate = create("onnx"), create("fixed")
    state_parity = []
    for feed in feeds:
        a, b = ref["encoder"].run(feed), candidate["encoder"].run(feed)
        comparison = {k: numeric(a[k], b[k]) for k in a}
        assert all(p["allclose_rtol1e_4_atol1e_4"] for p in comparison.values())
        assert all(p["exact"] for k, p in comparison.items() if k.startswith("new_cached_len_"))
        state_parity.append(comparison)
    contexts = [np.array([[0, 0]], np.int64)]
    contexts += [
        np.array([r["tokens"][i : i + 2]], np.int64)
        for r in evaluations["onnx"]["records"][:8]
        for i in range(min(4, len(r["tokens"]) - 1))
    ]
    decoder_feeds = [{"y": x} for x in contexts]
    frames = np.load(ROOT / baseline["smoke_audio"][0]["encoded_output"]["path"])[0]
    decoder_out = ref["decoder"].run(decoder_feeds[0])["decoder_out"]
    joiner_feeds = [
        {"encoder_out": frame[None], "decoder_out": decoder_out} for frame in frames[:32]
    ]
    fixtures = {"encoder": feeds, "decoder": decoder_feeds, "joiner": joiner_feeds}
    del ref, candidate
    gc.collect()
    times = {v: {c: [] for c in fixtures} for v in variants}
    rtf = {v: [] for v in variants}
    pipeline_samples = {v: [] for v in variants}
    orders = []
    # This parses the88.8MB source once. Metadata/cache preparation is not inference.
    zeros_for_model(source("encoder"))
    started = now()
    for cycle in range(5):
        order = variants[cycle % 3 :] + variants[: cycle % 3]
        if cycle % 2:
            order.reverse()
        orders.append(order)
        for v in order:
            runners = create(v)
            for c, runner in runners.items():
                for n in range(16):
                    runner.run(fixtures[c][n % len(fixtures[c])])
                runner.times.clear()
                for n in range(32):
                    runner.run(fixtures[c][n % len(fixtures[c])])
                times[v][c].extend(runner.times)
            if cycle < 3:
                seconds, audio = 0.0, 0.0
                for row in baseline["smoke_audio"]:
                    result, _ = transcribe(runners, SOURCE / "test_wavs" / row["name"])
                    assert result["tokens"] == row["tokens"]
                    seconds += result["pipeline_seconds"]
                    audio += result["audio_seconds"]
                rtf[v].append(seconds / audio)
                pipeline_samples[v].append({"pipeline_seconds": seconds, "audio_seconds": audio})
            del runners
            gc.collect()
        print("BENCHMARK", cycle + 1, "/5", flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "protocol": "Linux CPU, 4 threads; one runtime set at a time; 5 rotated/reversed rounds, 16 warmups then32 component calls each/round; no profile. Prime cached encoder input_specs outside timing. First3 rounds also time complete shared pipeline on all3 bundled audio; model and metadata loading excluded. Only decoder differs fixed/optimized.",
            "orders": orders,
            "agreement": agreement,
            "captured_encoder_state_parity": state_parity,
            "models": {
                v: {
                    "components_ms": {c: stats(x) for c, x in times[v].items()},
                    "pipeline_RTF": stats(rtf[v]),
                    "pipeline_rounds": pipeline_samples[v],
                    "artifacts": {c: info(p) for c, p in paths_for(v).items()},
                }
                for v in variants
            },
            "source_code": info(__file__),
        },
    )
    print(
        "SUMMARY",
        {
            v: {
                "RTF": float(np.median(rtf[v])),
                "decoder_ms": float(np.median(times[v]["decoder"])),
            }
            for v in variants
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
