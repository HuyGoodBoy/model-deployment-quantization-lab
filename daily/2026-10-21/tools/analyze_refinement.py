"""Recompute paired round uncertainty without pooling independent-process sessions."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT / "daily/2026-10-21"
sys.path.insert(0, str(DAY / "experiments/mlsd/src"))

from support21 import info, now, read, write


def paired(reference, candidate, block, seed):
    a = np.asarray(reference).reshape(5, block)
    b = np.asarray(candidate).reshape(5, block)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 5, size=(2000, 5))
    reductions = [100 * (1 - np.median(b[ix]) / np.median(a[ix])) for ix in indices]
    return {
        "median_latency_reduction_percent": float(100 * (1 - np.median(b) / np.median(a))),
        "paired_round_bootstrap95_percent": np.percentile(reductions, [2.5, 97.5]).tolist(),
        "rounds_candidate_faster": int(np.sum(np.median(b, axis=1) < np.median(a, axis=1))),
        "rounds": 5,
        "block_size": block,
        "resamples": 2000,
        "seed": seed,
        "scope": "Approximate within-session uncertainty from only5paired round clusters,not80/160independent samples or cross-device proof.",
    }


def main():
    target = DAY / "REFINEMENT_ANALYSIS.json"
    if target.exists():
        raise FileExistsError(target)
    mlsd, sources = {}, []
    for threads in [4, 8]:
        for session in [1, 2]:
            path = (
                DAY
                / f"experiments/mlsd/results/refinement_contiguous_benchmark_t{threads}_s{session}.json"
            )
            rows = read(path)["models"]
            mlsd[f"threads{threads}_session{session}"] = {
                name + "_vs_float": paired(
                    rows["float"]["samples_ms"],
                    rows[name]["samples_ms"],
                    16,
                    20261021 + threads * 10 + session,
                )
                for name in rows
                if name != "float"
            }
            sources.append(info(path))
    path = DAY / "experiments/zipformer/results/refinement_benchmark.json"
    rows = read(path)["models"]
    asr = {
        "fc_vs_baseline_encoder": paired(
            rows["baseline"]["encoder_ms"]["samples"],
            rows["fc"]["encoder_ms"]["samples"],
            32,
            20261021,
        ),
        "fc_bias_vs_fc_encoder": paired(
            rows["fc"]["encoder_ms"]["samples"],
            rows["fc_bias"]["encoder_ms"]["samples"],
            32,
            20261022,
        ),
        "fc_vs_baseline_pipeline_reduction_percent": 100
        * (1 - rows["fc"]["pipeline_RTF"]["median"] / rows["baseline"]["pipeline_RTF"]["median"]),
        "fc_bias_vs_fc_pipeline_reduction_percent": 100
        * (1 - rows["fc_bias"]["pipeline_RTF"]["median"] / rows["fc"]["pipeline_RTF"]["median"]),
        "decision": "Retain FC as default;bias-fused encoder has no demonstrated end-to-end advantage in this measured session;retain patch as experimental evidence.",
    }
    sources.append(info(path))
    write(
        target,
        {
            "finished_at_utc": now(),
            "mlsd": mlsd,
            "zipformer": asr,
            "sources": sources,
            "source_code": info(__file__),
        },
    )
    print("REFINEMENT ANALYSIS", {k: v["C1_vs_float"] for k, v in mlsd.items()}, asr, flush=True)


if __name__ == "__main__":
    main()
