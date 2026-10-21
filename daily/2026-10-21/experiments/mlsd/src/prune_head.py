"""Prune unused final filters into one five-channel Conv BEFORE PTQ."""

import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper
from support21 import ART, RESULTS, ROOT, checked, info, now, read, sha, write


def main():
    target = RESULTS / "refinement_prune_conversion.json"
    if target.exists():
        raise FileExistsError(target)
    registry = read(ROOT / "daily/2026-10-19/experiments/selective/results/exports.json")
    original = checked(registry["models"]["qat_weights_fp32"]["model"])
    original_sha = sha(original)
    model = onnx.load(original)
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    final = [n for n in model.graph.node if n.op_type == "Conv" and "/block16/conv3/" in n.name]
    assert len(final) == 1
    final = final[0]
    resize = next(n for n in model.graph.node if n.name == "/model/model/Resize")
    assert weights[final.input[1]].shape == (16, 64, 1, 1)
    assert weights[final.input[2]].shape == (16,)
    additions = [
        numpy_helper.from_array(weights[final.input[1]][7:12].copy(), "pruned_weight"),
        numpy_helper.from_array(weights[final.input[2]][7:12].copy(), "pruned_bias"),
        numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "pruned_resize_scales"),
    ]
    conv = helper.make_node(
        "Conv",
        [final.input[0], "pruned_weight", "pruned_bias"],
        ["pruned_lowres"],
        name="pruned_active_conv",
    )
    conv.attribute.extend(final.attribute)
    up = helper.make_node(
        "Resize",
        ["pruned_lowres", "", "pruned_resize_scales"],
        ["output"],
        name="pruned_output_resize",
    )
    up.attribute.extend(resize.attribute)
    nodes = [*list(model.graph.node)[: list(model.graph.node).index(final)], conv, up]
    needed, kept = {"output"}, []
    for node in reversed(nodes):
        if any(o in needed for o in node.output):
            kept.append(node)
            needed.update(i for i in node.input if i)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [*model.graph.initializer, *additions]
    del model.graph.initializer[:]
    model.graph.initializer.extend(i for i in initializers if i.name in needed)
    del model.graph.value_info[:]
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 5, 256, 256])
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    folder = ART / "refinement/prune_conversion"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "pruned.onnx"
    onnx.save(model, path)
    before = info(path)
    command = [
        "onnx2tf",
        "-i",
        str(path),
        "-o",
        str(folder / "output"),
        "-tb",
        "flatbuffer_direct",
        "-kat",
        "input",
        "-v",
        "error",
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    log = folder / "conversion.log"
    log.write_text(run.stdout + "\n" + run.stderr, encoding="utf-8", newline="\n")
    assert run.returncode == 0, run.stderr[-3000:]
    assert sha(original) == original_sha
    converted = folder / "output/pruned_float32.tflite"
    assert converted.exists()
    write(
        target,
        {
            "finished_at_utc": now(),
            "original": info(original),
            "rewritten_before_simplification": before,
            "converted_onnx": info(path),
            "tflite": info(converted),
            "command": command,
            "log": info(log),
            "rewrite": "Single final Conv with original filters7:12; unchanged upstream; same align_corners2xresize; one active5channel output; no split/concat output adapter.",
            "source_unchanged": True,
            "source_code": info(__file__),
        },
    )
    print("PRUNED CONVERSION", info(converted), flush=True)


if __name__ == "__main__":
    main()
