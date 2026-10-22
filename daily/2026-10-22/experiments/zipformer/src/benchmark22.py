"""Matched two-session fixed-graph wrapper timing, with all components warmed."""

import argparse
import gc

import numpy as np
from asr22 import DAY20, RESULTS, ROOT, info, now, read, source, write, zeros_for_model
from runtime22 import create
from streaming import transcribe


def stats(values):
    return {
        "samples": values,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "runs": len(values),
    }


def prepare():
    smoke = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in smoke for p in row["captures"]]
    encoder = [{k: v.copy() for k, v in np.load(ROOT / p["path"]).items()} for p in captures]
    contexts = read(RESULTS / "fixture_parity.json")["decoder"]
    decoder = [{"y": np.array(r["context"], np.int64)} for r in contexts]
    runners = create("Z0")
    outputs = [runners["decoder"].run(feed)["decoder_out"] for feed in decoder]
    joiner = []
    for index, feed in enumerate(encoder):
        encoded = runners["encoder"].run(feed)["encoder_out"]
        for frame in range(2):
            joiner.append(
                {
                    "encoder_out": encoded[0, frame][None],
                    "decoder_out": outputs[index % len(outputs)],
                }
            )
    del runners
    gc.collect()
    zeros_for_model(source("encoder"))
    return smoke, {"encoder": encoder, "decoder": decoder, "joiner": joiner}


def benchmark(session):
    target = RESULTS / f"benchmark_s{session}.json"
    if target.exists():
        raise FileExistsError(target)
    assert read(RESULTS / "fixture_parity.json")["exact_passed"]
    assert read(RESULTS / "regression.json")["quality_passed"]
    assert read(RESULTS / "continuous_state.json")["exact_passed"]
    smoke, feeds = prepare()
    times = {v: {component: [] for component in feeds} for v in ["Z0", "Z1"]}
    pipeline = {v: [] for v in times}
    started, orders = now(), []
    for cycle in range(5):
        order = ["Z0", "Z1"] if (cycle + session) % 2 else ["Z1", "Z0"]
        orders.append(order)
        for variant in order:
            runners = create(variant)
            for component, inputs in feeds.items():
                runner = runners[component]
                for i in range(16):
                    runner.run(inputs[i % len(inputs)])
                runner.times.clear()
                for i in range(32):
                    runner.run(inputs[i % len(inputs)])
                times[variant][component].extend(runner.times)
            # Also warm frontend and complete decoding before counted pipeline runs.
            transcribe(runners, DAY20 / "artifacts/source/test_wavs" / smoke[0]["name"])
            seconds, audio = 0.0, 0.0
            records = []
            for record in smoke:
                value, _ = transcribe(
                    runners, DAY20 / "artifacts/source/test_wavs" / record["name"]
                )
                assert value["tokens"] == record["tokens"]
                seconds += value["pipeline_seconds"]
                audio += value["audio_seconds"]
                records.append(
                    {
                        "name": record["name"],
                        "pipeline_seconds": value["pipeline_seconds"],
                        "audio_seconds": value["audio_seconds"],
                    }
                )
            pipeline[variant].append(
                {
                    "RTF": seconds / audio,
                    "pipeline_seconds": seconds,
                    "audio_seconds": audio,
                    "records": records,
                }
            )
            del runners
            gc.collect()
        print("WRAPPER BENCHMARK", session, cycle + 1, "/5", flush=True)
    write(
        target,
        {
            "started_at_utc": started,
            "finished_at_utc": now(),
            "session": session,
            "orders": orders,
            "models": {
                v: {
                    "components_ms": {c: stats(x) for c, x in components.items()},
                    "pipeline_RTF": stats([r["RTF"] for r in pipeline[v]]),
                    "pipeline_rounds": pipeline[v],
                }
                for v, components in times.items()
            },
            "protocol": "Linux LiteRT2.1.2 CPU4threads,new process per session;one pipeline at a time,no competing experiment.5alternating rounds16warmups+32timings/component=160samples. All components and one unmeasured complete smoke decode warmed before each counted3audio pipeline. SameFC/decoder/joiner graph,frontend/context/state/greedy and fixed shapes. Load/metadata excluded;frontend/read/greedy included. Warm policy differs from day21;only use matched day22 controls.",
            "source_code": info(__file__),
        },
    )
    print(
        "WRAPPER TIMING",
        session,
        {
            v: {
                "encoder_ms": stats(times[v]["encoder"])["median"],
                "RTF": float(np.median([r["RTF"] for r in pipeline[v]])),
            }
            for v in times
        },
        flush=True,
    )


def paired(a, b, block, seed):
    a, b = np.asarray(a).reshape(5, block), np.asarray(b).reshape(5, block)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 5, size=(2000, 5))
    gain = [100 * (1 - np.median(b[i]) / np.median(a[i])) for i in indices]
    return {
        "median_reduction_percent": float(100 * (1 - np.median(b) / np.median(a))),
        "paired_round_bootstrap95_percent": np.percentile(gain, [2.5, 97.5]).tolist(),
        "faster_rounds": int(np.sum(np.median(b, axis=1) < np.median(a, axis=1))),
        "rounds": 5,
        "resamples": 2000,
        "seed": seed,
    }


def lock():
    target = RESULTS / "selection.json"
    if target.exists():
        raise FileExistsError(target)
    rows = {}
    for session in [1, 2]:
        record = read(RESULTS / f"benchmark_s{session}.json")["models"]
        a, b = record["Z0"], record["Z1"]
        rows[f"session{session}"] = {
            "encoder": paired(
                a["components_ms"]["encoder"]["samples"],
                b["components_ms"]["encoder"]["samples"],
                32,
                20261022 + session,
            ),
            "pipeline": paired(
                a["pipeline_RTF"]["samples"], b["pipeline_RTF"]["samples"], 1, 20261032 + session
            ),
        }
    selected = (
        "Z1"
        if all(
            r["pipeline"]["median_reduction_percent"] > 0
            and r["pipeline"]["paired_round_bootstrap95_percent"][0] > 0
            for r in rows.values()
        )
        else "Z0"
    )
    write(
        target,
        {
            "locked_at_utc": now(),
            "selected": selected,
            "comparisons": rows,
            "rule": "Exact fixtures/173tokens/state/600second parity;pipeline median lower with positive within-session paired5round95lower bound in both sessions. Samehost/day only;not deployment certification.",
            "source_code": info(__file__),
        },
    )
    print("WRAPPER SELECTED", selected, rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["benchmark", "lock"])
    parser.add_argument("--session", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    benchmark(args.session) if args.stage == "benchmark" else lock()
