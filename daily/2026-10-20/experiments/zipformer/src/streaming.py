"""Shared frontend/stateful greedy decoding; only component runtime changes."""

import re
import time

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from support import SOURCE, source, zeros_for_model


def words(text):
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def errors(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    previous = [(i, 0, 0, i) for i in range(len(hyp) + 1)]
    for i, expected in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, actual in enumerate(hyp, 1):
            if expected == actual:
                current.append(previous[j - 1])
                continue
            cost, s, d, ins = previous[j - 1]
            sub = (cost + 1, s + 1, d, ins)
            cost, s, d, ins = previous[j]
            delete = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = current[j - 1]
            insert = (cost + 1, s, d, ins + 1)
            current.append(min(sub, delete, insert))
        previous = current
    cost, sub, delete, insert = previous[-1]
    return {
        "S": sub,
        "D": delete,
        "I": insert,
        "N": len(ref),
        "errors": cost,
        "WER_percent": cost / max(len(ref), 1) * 100,
    }


def features(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = audio[:, 0]
    duration = len(waveform) / rate
    if rate != 16000:
        from math import gcd

        common = gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // common, rate // common).astype(np.float32)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fbank = knf.OnlineFbank(opts)
    padded = np.concatenate([waveform, np.zeros(4800, np.float32)])
    fbank.accept_waveform(16000, padded.tolist())
    # Match the upstream streaming example: 0.3 s tail, no extra reflected finish.
    matrix = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return matrix, duration, rate


def transcribe(runners, path, capture_prefix=None, initial_context=None):
    started = time.perf_counter()
    matrix, duration, sample_rate = features(path)
    state = zeros_for_model(source("encoder"))
    del state["x"]
    token_map = {}
    for line in (SOURCE / "tokens.txt").read_text(encoding="utf-8").splitlines():
        symbol, index = line.rsplit(" ", 1)
        token_map[int(index)] = symbol
    hypothesis = [0, 0] if initial_context is None else list(initial_context)
    assert len(hypothesis) == 2
    decoder = runners["decoder"].run({"y": np.array([hypothesis], np.int64)})["decoder_out"]
    captures = []
    encoded = []
    count = 0
    positions = list(range(0, max(0, len(matrix) - 39 + 1), 32))
    for offset in positions:
        feed = {"x": matrix[offset : offset + 39][None], **state}
        output = runners["encoder"].run(feed)
        tensor = output["encoder_out"]
        if capture_prefix is not None and (count < 4 or offset == positions[-1]):
            captured = capture_prefix.parent / f"{capture_prefix.name}_chunk{count:03d}.npz"
            np.savez_compressed(captured, **feed)
            captures.append(str(captured))
        encoded.append(tensor.copy())
        state = {name: output["new_" + name] for name in state}
        for frame in tensor[0]:
            logits = runners["joiner"].run({"encoder_out": frame[None], "decoder_out": decoder})[
                "logit"
            ]
            token = int(np.argmax(logits))
            if token != 0:
                hypothesis.append(token)
                decoder = runners["decoder"].run({"y": np.array([hypothesis[-2:]], np.int64)})[
                    "decoder_out"
                ]
        count += 1
    text = "".join(token_map[token] for token in hypothesis[2:]).replace("\u2581", " ").strip()
    elapsed = time.perf_counter() - started
    return {
        "text": text,
        "tokens": hypothesis[2:],
        "audio_seconds": duration,
        "original_sample_rate": sample_rate,
        "chunks": count,
        "pipeline_seconds": elapsed,
        "RTF": elapsed / max(duration, 1e-30),
        "captures": captures,
        "final_cached_len": {
            name: value.tolist() for name, value in state.items() if name.startswith("cached_len")
        },
    }, np.concatenate(encoded, axis=1)
