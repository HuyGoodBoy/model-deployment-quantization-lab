"""Timing uncertainty from paired measurement rounds, not pooled independent draws."""

import numpy as np
from support21 import RESULTS, info, now, read, write


def paired(a, b, seed):
    a, b = np.asarray(a).reshape(5, 16), np.asarray(b).reshape(5, 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 5, size=(2000, 5))
    gain = [100 * (1 - np.median(b[ix]) / np.median(a[ix])) for ix in indices]
    return {
        "median_latency_reduction_percent": float(100 * (1 - np.median(b) / np.median(a))),
        "paired_round_block_bootstrap_95_interval_percent": np.percentile(
            gain, [2.5, 97.5]
        ).tolist(),
        "rounds_candidate_faster_by_round_median": int(
            np.sum(np.median(b, axis=1) < np.median(a, axis=1))
        ),
        "rounds": 5,
        "resamples": 2000,
        "seed": seed,
        "scope": "Approximate uncertainty from5paired round clusters;not independent80samples or proof across devices/sessions.",
    }


def main():
    a = read(RESULTS / "benchmark.json")["models"]
    b = read(RESULTS / "benchmark8.json")["models"]
    write(
        RESULTS / "timing_analysis.json",
        {
            "finished_at_utc": now(),
            "head_float_vs_float_threads4": paired(
                a["float"]["samples_ms"], a["head_float"]["samples_ms"], 20261021
            ),
            "head_float_vs_float_threads8": paired(
                b["float"]["samples_ms"], b["head_float"]["samples_ms"], 20261022
            ),
            "source_code": info(__file__),
        },
    )
    print(read(RESULTS / "timing_analysis.json"), flush=True)


if __name__ == "__main__":
    main()
