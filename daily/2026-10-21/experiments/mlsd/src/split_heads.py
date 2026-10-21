"""Separate active Conv filters BEFORE quantization, with two native outputs."""

import importlib.metadata
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper
from support21 import ART, RESULTS, ROOT, checked, info, now, read, sha, write


def main():
    target = RESULTS / "split_conversion.json"
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
    assert (
        helper.get_attribute_value(
            next(a for a in resize.attribute if a.name == "coordinate_transformation_mode")
        )
        == b"align_corners"
    )
    cut = list(model.graph.node).index(final)
    nodes = list(model.graph.node)[:cut]
    additions = [numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "split_resize_scales")]
    for name, start, end in [("center", 7, 8), ("displacement", 8, 12)]:
        wname, bname = name + "_weight", name + "_bias"
        additions.extend(
            [
                numpy_helper.from_array(weights[final.input[1]][start:end].copy(), wname),
                numpy_helper.from_array(weights[final.input[2]][start:end].copy(), bname),
            ]
        )
        conv = helper.make_node(
            "Conv",
            [final.input[0], wname, bname],
            [name + "_lowres"],
            name="split_" + name + "_conv",
        )
        conv.attribute.extend(final.attribute)
        up = helper.make_node(
            "Resize",
            [name + "_lowres", "", "split_resize_scales"],
            [name],
            name="split_" + name + "_resize",
        )
        up.attribute.extend(resize.attribute)
        nodes.extend([conv, up])
    # Remove all channels not consumed by ActiveMlsd, retaining active order7..11.
    needed = {"center", "displacement"}
    kept = []
    for node in reversed(nodes):
        if any(o in needed for o in node.output):
            kept.append(node)
            needed.update(i for i in node.input if i)
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    initializers = [*model.graph.initializer, *additions]
    del model.graph.initializer[:]
    model.graph.initializer.extend(i for i in initializers if i.name in needed)
    del model.graph.value_info[:]
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("center", onnx.TensorProto.FLOAT, [1, 1, 256, 256]),
            helper.make_tensor_value_info("displacement", onnx.TensorProto.FLOAT, [1, 4, 256, 256]),
        ]
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    folder = ART / "split_conversion"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "split.onnx"
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
    assert run.returncode == 0, run.stderr[-2500:]
    assert sha(original) == original_sha
    converted = folder / "output/split_float32.tflite"
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
            "rewrite": "Keep original upstream graph; split final16-filter1x1Conv into original filter7(center) and8:12(displacement); independent align_corners2xresize; discard unused filters; two native outputs, application merges only after dequantization. Numeric/API parity is checked separately before PTQ.",
            "source_unchanged": True,
            "environment": {
                p: importlib.metadata.version(p)
                for p in ["onnx2tf", "ai-edge-litert", "onnxruntime"]
            },
            "source_code": info(__file__),
        },
    )
    print("SPLIT CONVERTED", info(converted), flush=True)


if __name__ == "__main__":
    main()
