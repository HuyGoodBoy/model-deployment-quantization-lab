"""Compare the same streaming/frontend/decoder pipeline across component runtimes."""

import argparse
import gc
import importlib.metadata

import numpy as np
from streaming import errors, transcribe
from support import (
    ART,
    RESULTS,
    ROOT,
    SOURCE,
    LiteRunner,
    OnnxRunner,
    info,
    now,
    numeric,
    read,
    source,
    write,
    zeros_for_model,
)


def paths_for(variant):
    if variant == "onnx":
        return {component: source(component) for component in ["encoder", "decoder", "joiner"]}
    if variant == "fixed":
        return {
            "encoder": ART / "conversion/static/encoder/output/input_float32.tflite",
            "decoder": ART / "conversion/conv2d/decoder/output/input_float32.tflite",
            "joiner": ART / "conversion/static/joiner/output/input_float32.tflite",
        }
    registry = read(RESULTS / f"registry_{variant}.json")
    return {name: ROOT / item["path"] for name, item in registry["models"].items()}


def create(variant):
    runner = OnnxRunner if variant == "onnx" else LiteRunner
    return {name: runner(path) for name, path in paths_for(variant).items()}


def main(args):
    variant = args.variant
    target = RESULTS / f"evaluation_{variant}.json"
    if target.exists():
        raise FileExistsError(target)
    manifest_path = RESULTS / "evaluation_manifest.json"
    manifest = read(manifest_path)
    runners = create(variant)
    # Warm each loaded model; encoder states are reset for each later utterance.
    for name, runner in runners.items():
        feed = zeros_for_model(source(name))
        for _ in range(16):
            runner.run(feed)
        runner.times.clear()
    smoke = read(RESULTS / "onnx_baseline.json")["smoke_audio"]
    smoke_rows = []
    for row in smoke:
        result, encoded = transcribe(runners, SOURCE / "test_wavs" / row["name"])
        original = np.load(ROOT / row["encoded_output"]["path"])
        result.update(
            name=row["name"],
            reference=row["reference"],
            word_errors=errors(row["reference"], result["text"]),
            original_token_agreement=result["tokens"] == row["tokens"],
            encoder_parity=numeric(original, encoded),
        )
        smoke_rows.append(result)
        print(
            "SMOKE",
            variant,
            row["name"],
            result["text"],
            "token agreement",
            result["original_token_agreement"],
            flush=True,
        )
    records = []
    started_at = now()
    for index, row in enumerate(manifest["records"]):
        path = ROOT / row["path"]
        result, _encoded = transcribe(runners, path)
        result.update(
            id=row["id"], reference=row["text"], word_errors=errors(row["text"], result["text"])
        )
        records.append(result)
        if (index + 1) % 10 == 0:
            print("EVALUATION", variant, index + 1, "/", len(manifest["records"]), flush=True)
    total = {
        name: sum(r["word_errors"][name] for r in records)
        for name in ["S", "D", "I", "N", "errors"]
    }
    total["WER_percent"] = total["errors"] / total["N"] * 100
    audio_s = sum(r["audio_seconds"] for r in records)
    pipeline_s = sum(r["pipeline_seconds"] for r in records)
    timing = {
        name: {
            "calls": len(runner.times),
            "median_ms": float(np.median(runner.times)),
            "p95_ms": float(np.percentile(runner.times, 95)),
        }
        for name, runner in runners.items()
    }
    counter_widenings = {
        name: getattr(runner, "counter_widenings", 0) for name, runner in runners.items()
    }
    write(
        target,
        {
            "started_at_utc": started_at,
            "finished_at_utc": now(),
            "variant": variant,
            "models": {name: info(path) for name, path in paths_for(variant).items()},
            "manifest": info(manifest_path),
            "utterances": len(records),
            "reference_words": total["N"],
            "audio_seconds": audio_s,
            "pipeline_seconds": pipeline_s,
            "RTF": pipeline_s / audio_s,
            "word_errors": total,
            "records": records,
            "smoke": smoke_rows,
            "observed_component_timing": timing,
            "checked_INT32_to_INT64_counter_widenings": counter_widenings,
            "environment": {
                p: importlib.metadata.version(p)
                for p in ["onnxruntime", "ai-edge-litert", "kaldi-native-fbank"]
            },
            "scope": "WER on all73 public dummy utterances (one speaker), not full LibriSpeech benchmark. RTF includes shared Python frontend, file reading, state reset and greedy search; preloaded/warmed models, no model loading. Dedicated matched benchmark is separate.",
            "source_code": info(__file__),
        },
    )
    print("RESULT", variant, "WER", total, "RTF", pipeline_s / audio_s, flush=True)
    del runners
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["onnx", "fixed", "optimized"], required=True)
    main(parser.parse_args())
