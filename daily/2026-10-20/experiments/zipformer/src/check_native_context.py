"""Investigate native Sherpa disagreement without changing the locked pipeline."""

from streaming import transcribe
from support import RESULTS, SOURCE, OnnxRunner, info, now, read, source, write


def main():
    target = RESULTS / "native_context_diagnosis.json"
    if target.exists():
        raise FileExistsError(target)
    baseline = read(RESULTS / "onnx_baseline.json")
    runners = {c: OnnxRunner(source(c)) for c in ["encoder", "decoder", "joiner"]}
    rows = []
    for row in baseline["smoke_audio"]:
        result, _ = transcribe(runners, SOURCE / "test_wavs" / row["name"], initial_context=[-1, 0])
        rows.append(
            {
                "name": row["name"],
                "icefall_zero_context_text": row["text"],
                "negative_context_text": result["text"],
                "native_text": row["sherpa_native_text"],
                "negative_context_matches_native": result["text"].upper()
                == row["sherpa_native_text"].upper(),
            }
        )
        print("NATIVE CONTEXT", rows[-1], flush=True)
    write(
        target,
        {
            "finished_at_utc": now(),
            "records": rows,
            "scope": "Diagnostic only; main ONNX/fixed/lookup WER and benchmarks all retain[0,0]initial context following Icefall ONNX example. Native Sherpa initializes[-1,0], and also treats unk as blank. Do not call the two pipelines equivalent based only on conversion parity.",
            "sources": [
                "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/onnx_pretrained.py",
                "https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc",
            ],
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
