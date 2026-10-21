"""Preserve public graph/state contracts; lower only constant rank2 RHS BMM to FC."""

from collections import Counter

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from asr21 import ART, DAY20, RESULTS, ROOT, LiteRunner, info, now, numeric, paths, read, sha, write


def main():
    target = RESULTS / "encoder_fc_rewrite.json"
    if target.exists():
        raise FileExistsError(target)
    original = paths("baseline")["encoder"]
    before = sha(original)
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(original.read_bytes(), 0))
    assert len(model.subgraphs) == 1
    graph = model.subgraphs[0]
    fc_code = s.OperatorCodeT()
    fc_code.builtinCode = s.BuiltinOperator.FULLY_CONNECTED
    fc_code.deprecatedBuiltinCode = s.BuiltinOperator.FULLY_CONNECTED
    fc_code.version = 5  # keep_num_dims; three input slots, optional bias=-1.
    opcode = len(model.operatorCodes)
    model.operatorCodes.append(fc_code)
    uses = Counter(int(i) for op in graph.operators for i in op.inputs if i >= 0)
    changed = []
    for op in graph.operators:
        if model.operatorCodes[op.opcodeIndex].builtinCode != s.BuiltinOperator.BATCH_MATMUL:
            continue
        if op.builtinOptions.adjX or op.builtinOptions.adjY:
            continue
        left, right = (graph.tensors[int(k)] for k in op.inputs)
        if len(right.shape) != 2 or right.type != s.TensorType.FLOAT32:
            continue
        data = model.buffers[right.buffer].data
        if data is None or len(data) == 0:
            continue
        assert uses[int(op.inputs[1])] == 1, "Shared RHS needs a cloned tensor"
        k, n = map(int, right.shape)
        assert int(left.shape[-1]) == k and left.type == s.TensorType.FLOAT32
        weight = np.frombuffer(data, dtype=np.float32).reshape(k, n)
        transposed = np.ascontiguousarray(weight.T)
        buffer = s.BufferT()
        buffer.data = np.frombuffer(transposed.tobytes(), dtype=np.uint8)
        right.buffer = len(model.buffers)
        model.buffers.append(buffer)
        right.shape = np.array([n, k], dtype=np.int32)
        if right.shapeSignature is not None and len(right.shapeSignature):
            right.shapeSignature = right.shape.copy()
        options = s.FullyConnectedOptionsT()
        options.keepNumDims = True
        options.fusedActivationFunction = s.ActivationFunctionType.NONE
        op.opcodeIndex = opcode
        op.inputs = np.array([int(op.inputs[0]), int(op.inputs[1]), -1], dtype=np.int32)
        op.builtinOptionsType = s.BuiltinOptions.FullyConnectedOptions
        op.builtinOptions = options
        changed.append(
            {
                "weight": right.name.decode(),
                "old_shape": [k, n],
                "new_shape": [n, k],
                "output_tensor": int(op.outputs[0]),
                "bias_fused": False,
            }
        )
    assert len(changed) == 112, len(changed)
    # Discard superseded weight buffers so the rewrite does not duplicate weights.
    used = {0}
    used.update(t.buffer for g in model.subgraphs for t in g.tensors)
    used.update(m.buffer for m in (model.metadata or []))
    if model.metadataBuffer is not None:
        used.update(int(i) for i in model.metadataBuffer)
    remap = {old: new for new, old in enumerate(sorted(used))}
    model.buffers = [model.buffers[i] for i in sorted(used)]
    for g in model.subgraphs:
        for tensor in g.tensors:
            tensor.buffer = remap[tensor.buffer]
    for metadata in model.metadata or []:
        metadata.buffer = remap[metadata.buffer]
    if model.metadataBuffer is not None:
        model.metadataBuffer = np.array(
            [remap[int(i)] for i in model.metadataBuffer], dtype=np.int32
        )
    ART.mkdir(parents=True, exist_ok=True)
    output = ART / "encoder_fc.tflite"
    assert not output.exists()
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    output.write_bytes(bytes(builder.Output()))
    assert sha(original) == before
    baseline, candidate = LiteRunner(original), LiteRunner(output)
    baseline_smoke = read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
    captures = [p for row in baseline_smoke for p in row["captures"]]
    parity = []
    for captured in captures:
        with np.load(ROOT / captured["path"]) as data:
            feed = {k: v.copy() for k, v in data.items()}
        a, b = baseline.run(feed), candidate.run(feed)
        row = {name: numeric(a[name], b[name]) for name in a}
        assert all(p["allclose_rtol1e_4_atol1e_4"] for p in row.values())
        assert all(p["exact"] for name, p in row.items() if name.startswith("new_cached_len_"))
        parity.append(row)
    counts = Counter(op["op_name"] for op in candidate.runtime._get_ops_details())
    assert counts["FULLY_CONNECTED"] == 112 and counts["BATCH_MATMUL"] == 40
    write(
        target,
        {
            "finished_at_utc": now(),
            "author": "Codex, replacement for Claude explicitly requested by user on21/10",
            "original": info(original),
            "model": info(output),
            "changes": changed,
            "runtime_operator_counts": dict(counts),
            "parity_on15_state_fixtures": parity,
            "reference_versioning_source": "https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/tools/versioning/op_version.cc",
            "scope": "112 BMMs with constant FLOAT32 rank2 RHS and no adjoints become FCv5 keepNumDims; transpose exact weight values, no bias fusion/reassociation/quantization;40 dynamic attention BMMs retained; same signatures and35state. No timing claim before matched benchmark.",
            "source_code": info(__file__),
        },
    )
    print("REWRITE PASS", dict(counts), "size", output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
