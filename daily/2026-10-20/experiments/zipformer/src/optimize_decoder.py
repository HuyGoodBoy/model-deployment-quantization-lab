"""Codex experiment: fold fixed token embeddings into grouped-conv lookup tables."""

import subprocess
from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper
from support import ART, RESULTS, LiteRunner, OnnxRunner, info, now, numeric, source, write


def main():
    target = RESULTS / "decoder_lookup_optimization.json"
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(ART / "conversion/static/decoder/input.onnx")
    values = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    embedding = values["decoder.embedding.weight"].reshape(500, 128, 4)
    weight = values["decoder.conv.weight"].reshape(128, 4, 4, 2)
    # Linear embedding+Conv, computed once for each token and context position.
    tables = [
        np.einsum("vgc,goc->vgo", embedding, weight[..., pos]).reshape(500, 512) for pos in range(2)
    ]
    nodes, initializers = [], []
    for pos, table in enumerate(tables):
        initializers.extend(
            [
                numpy_helper.from_array(table, f"lookup_{pos}"),
                numpy_helper.from_array(np.array([pos], np.int64), f"pos_{pos}"),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Gather", ["y", f"pos_{pos}"], [f"token_{pos}"], axis=1),
                helper.make_node(
                    "Gather", [f"lookup_{pos}", f"token_{pos}"], [f"part_{pos}"], axis=0
                ),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Add", ["part_0", "part_1"], ["conv_folded"]),
            helper.make_node("Relu", ["conv_folded"], ["/decoder/Relu_output_0"]),
            *[n for n in model.graph.node if n.op_type in ["Squeeze", "Gemm"]],
        ]
    )
    initializers.extend(
        x
        for x in model.graph.initializer
        if x.name in ["/Constant_output_0", "decoder_proj.weight", "decoder_proj.bias"]
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    folder = ART / "conversion/lookup/decoder"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "input.onnx"
    onnx.save(model, path)
    original, optimized = OnnxRunner(source("decoder")), OnnxRunner(path)
    rng = np.random.default_rng(20261020)
    contexts = [np.array([[a, b]], np.int64) for a, b in [(0, 0), (0, 499), (499, 0), (499, 499)]]
    contexts += [rng.integers(0, 500, (1, 2), dtype=np.int64) for _ in range(60)]
    parity = [
        numeric(original.run({"y": y})["decoder_out"], optimized.run({"y": y})["decoder_out"])
        for y in contexts
    ]
    assert all(p["allclose_rtol1e_4_atol1e_4"] for p in parity)
    cmd = [
        "onnx2tf",
        "-i",
        str(path),
        "-o",
        str(folder / "output"),
        "-tb",
        "flatbuffer_direct",
        "-kat",
        "y",
        "-v",
        "error",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    (folder / "conversion.log").write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
    assert run.returncode == 0, run.stderr[-2000:]
    lite_path = folder / "output/input_float32.tflite"
    lite = LiteRunner(lite_path)
    lite_parity = [
        numeric(original.run({"y": y})["decoder_out"], lite.run({"y": y})["decoder_out"])
        for y in contexts
    ]
    assert all(p["allclose_rtol1e_4_atol1e_4"] for p in lite_parity)
    ops = Counter(o["op_name"] for o in lite.runtime._get_ops_details())
    assert "CONV_2D" not in ops and not any(o.startswith("ONNX_") for o in ops)
    write(
        target,
        {
            "finished_at_utc": now(),
            "author": "Codex, not Claude; Claude auth unavailable",
            "change": "Precompute two 500x512 grouped-conv contributions from embedding weights; Gather+Add replaces token embedding+128-group Conv. Float32 rounding can change; parity and streaming WER checked separately.",
            "contexts": [y.tolist() for y in contexts],
            "onnx_parity": parity,
            "lite_parity": lite_parity,
            "operators": dict(ops),
            "onnx": info(path),
            "tflite": info(lite_path),
            "command": cmd,
            "source_code": info(__file__),
        },
    )
    write(
        RESULTS / "registry_optimized.json",
        {
            "models": {
                "encoder": info(ART / "conversion/static/encoder/output/input_float32.tflite"),
                "decoder": info(lite_path),
                "joiner": info(ART / "conversion/static/joiner/output/input_float32.tflite"),
            }
        },
    )
    print("LOOKUP", dict(ops), "MAXABS", max(p["max_abs"] for p in lite_parity), flush=True)


if __name__ == "__main__":
    main()
