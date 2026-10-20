"""Audit and verify the streaming ONNX baseline before LiteRT comparison."""

import importlib.metadata
from collections import Counter

import numpy as np
import onnx
import sherpa_onnx
import soundfile as sf
from streaming import errors, transcribe
from support import ART, RESULTS, SOURCE, OnnxRunner, info, now, signature, source, write


def main():
    target = RESULTS / "onnx_baseline.json"
    if target.exists():
        raise FileExistsError(target)
    ART.mkdir(parents=True, exist_ok=True)
    graph = {}
    for component in ["encoder", "decoder", "joiner"]:
        model = onnx.load(source(component))
        graph[component] = {
            "model": info(source(component)),
            "opset": [{"domain": x.domain, "version": x.version} for x in model.opset_import],
            "inputs": [signature(x) for x in model.graph.input],
            "outputs": [signature(x) for x in model.graph.output],
            "metadata": {x.key: x.value for x in model.metadata_props},
            "operators": dict(Counter(n.op_type for n in model.graph.node)),
        }
    references = {}
    for line in (SOURCE / "test_wavs/trans.txt").read_text().splitlines():
        name, text = line.split(" ", 1)
        references[name] = text
    runners = {c: OnnxRunner(source(c)) for c in graph}
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(SOURCE / "tokens.txt"),
        encoder=str(source("encoder")),
        decoder=str(source("decoder")),
        joiner=str(source("joiner")),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )
    records = []
    for name, reference in references.items():
        path = SOURCE / "test_wavs" / name
        result, encoded = transcribe(runners, path, ART / ("capture_" + name.replace(".", "_")))
        result["captures"] = [info(p) for p in result["captures"]]
        output_path = ART / ("onnx_encoded_" + name + ".npy")
        np.save(output_path, encoded, allow_pickle=False)
        result["encoded_output"] = info(output_path)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        native = recognizer.create_stream()
        native.accept_waveform(rate, audio[:, 0])
        native.accept_waveform(rate, np.zeros(int(rate * 0.3), np.float32))
        native.input_finished()
        while recognizer.is_ready(native):
            recognizer.decode_stream(native)
        official = recognizer.get_result(native)
        result.update(
            name=name,
            reference=reference,
            word_errors=errors(reference, result["text"]),
            sherpa_native_text=official,
            native_text_agreement=result["text"].upper() == official.upper(),
        )
        records.append(result)
        print("BASELINE", name, result["text"], "native", official, flush=True)
    write(
        target,
        {
            "recorded_at_utc": now(),
            "graphs": graph,
            "smoke_audio": records,
            "environment": {
                p: importlib.metadata.version(p)
                for p in [
                    "onnx",
                    "onnxruntime",
                    "ai-edge-litert",
                    "onnx2tf",
                    "kaldi-native-fbank",
                    "sherpa-onnx",
                ]
            },
            "frontend": "KNF80,16kHz normalized float waveform,dither0,snip_edgesFalse,high_freq-400,tail0.3s;8k resample_poly;chunk39 frames/hop32;greedy max1symbol/frame;blank0,context2;state reset per utterance",
            "scope": "Three bundled audio clips are smoke/frontend checks, not independent quality benchmark; native sherpa is an additional reference, not the timing baseline",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
