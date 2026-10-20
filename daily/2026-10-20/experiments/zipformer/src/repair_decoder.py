"""Embed the exact grouped Conv1D in a rank-4 Conv without changing arithmetic."""

import subprocess
import time
from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper
from support import ART, RESULTS, LiteRunner, OnnxRunner, info, now, numeric, source, write


def main():
    target = RESULTS / "decoder_conv2d_repair.json"
    if target.exists():
        raise FileExistsError(target)
    original = ART / "conversion/static/decoder/input.onnx"
    model = onnx.load(original)
    initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    additions, nodes, changes = [], [], []
    for node in model.graph.node:
        if node.op_type != "Conv" or initializers[node.input[1]].ndim != 3:
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = initializers[node.input[1]]
        assert attrs.get("auto_pad", b"NOTSET") in [b"NOTSET", b"VALID"]
        pads = attrs.get("pads", [0, 0])
        assert len(pads) == 2
        expanded_weight = node.input[1] + "_rank4"
        axis = node.name + "_axis2"
        additions.extend(
            [
                numpy_helper.from_array(weight[:, :, None, :], expanded_weight),
                numpy_helper.from_array(np.array([2], np.int64), axis),
            ]
        )
        x4, y4 = node.input[0] + "_rank4", node.output[0] + "_rank4"
        stride, dilation = attrs.get("strides", [1]), attrs.get("dilations", [1])
        nodes.append(
            helper.make_node("Unsqueeze", [node.input[0], axis], [x4], name=node.name + "_expand")
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [x4, expanded_weight, *node.input[2:]],
                [y4],
                name=node.name + "_rank4",
                kernel_shape=[1, weight.shape[-1]],
                pads=[0, pads[0], 0, pads[1]],
                strides=[1, stride[0]],
                dilations=[1, dilation[0]],
                group=attrs.get("group", 1),
            )
        )
        nodes.append(
            helper.make_node("Squeeze", [y4, axis], list(node.output), name=node.name + "_squeeze")
        )
        changes.append(
            {
                "node": node.name,
                "group": attrs.get("group", 1),
                "old_weight_shape": list(weight.shape),
                "new_weight_shape": list(weight[:, :, None, :].shape),
                "embedding": "NCW -> NC1W; same groups/weights/bias/stride/dilation/padding",
            }
        )
    assert len(changes) == 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(additions)
    # Rank information for the newly inserted tensors must be inferred.
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    folder = ART / "conversion/conv2d/decoder"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "input.onnx"
    onnx.save(model, path)
    original_runtime, fixed_runtime = OnnxRunner(source("decoder")), OnnxRunner(path)
    rng = np.random.default_rng(20261020)
    contexts = [
        np.zeros((1, 2), np.int64),
        np.array([[1, 499]], np.int64),
        *[rng.integers(0, 500, (1, 2), dtype=np.int64) for _ in range(16)],
    ]
    parity = [
        numeric(
            original_runtime.run({"y": y})["decoder_out"],
            fixed_runtime.run({"y": y})["decoder_out"],
        )
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
    started = time.perf_counter()
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    log = folder / "conversion.log"
    log.write_text(run.stdout + "\n" + run.stderr, encoding="utf-8", newline="\n")
    assert run.returncode == 0, run.stderr[-1000:]
    tflite = folder / "output/input_float32.tflite"
    runtime = LiteRunner(tflite)
    lite_parity = [
        numeric(original_runtime.run({"y": y})["decoder_out"], runtime.run({"y": y})["decoder_out"])
        for y in contexts
    ]
    assert all(p["allclose_rtol1e_4_atol1e_4"] for p in lite_parity)
    ops = Counter(x["op_name"] for x in runtime.runtime._get_ops_details())
    assert not any(x.startswith("ONNX_") for x in ops)
    write(
        target,
        {
            "finished_at_utc": now(),
            "author": "Codex; baseline compatibility repair, not a Claude execution",
            "original": info(original),
            "fixed_onnx": info(path),
            "tflite": info(tflite),
            "changes": changes,
            "command": cmd,
            "conversion_seconds": time.perf_counter() - started,
            "log": info(log),
            "test_contexts": [x.tolist() for x in contexts],
            "onnx_parity": parity,
            "lite_parity": lite_parity,
            "runtime_operators": dict(ops),
            "source_code": info(__file__),
        },
    )
    print(
        "REPAIRED DECODER",
        dict(ops),
        "worst maxabs",
        max(x["max_abs"] for x in lite_parity),
        flush=True,
    )


if __name__ == "__main__":
    main()
