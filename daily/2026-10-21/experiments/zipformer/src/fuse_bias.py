"""Fuse eligible constant ADD biases into FC; keep public state/API unchanged."""

from collections import Counter

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from asr21 import ART, DAY20, RESULTS, ROOT, LiteRunner, info, now, numeric, paths, read, write


def counts(model):
    names = {getattr(s.BuiltinOperator, n): n for n in dir(s.BuiltinOperator) if n.isupper()}
    return dict(
        Counter(
            names[model.operatorCodes[op.opcodeIndex].builtinCode]
            for op in model.subgraphs[0].operators
        )
    )


def storage(model, raw):
    payload = sum(len(b.data) for b in model.buffers if b.data is not None)
    return {
        "file_bytes": len(raw),
        "buffer_payload_bytes": payload,
        "non_payload_bytes": len(raw) - payload,
        "buffers": len(model.buffers),
        "tensors": sum(len(g.tensors) for g in model.subgraphs),
    }


def main():
    target = RESULTS / "refinement_bias_rewrite.json"
    if target.exists():
        raise FileExistsError(target)
    original = paths("fc")["encoder"]
    original_record = info(original)
    raw = original.read_bytes()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(raw, 0))
    graph = model.subgraphs[0]
    before_counts = counts(model)
    users = {}
    for index, op in enumerate(graph.operators):
        for tensor in op.inputs:
            if tensor >= 0:
                users.setdefault(int(tensor), []).append(index)
    deleted, changes, skips = set(), [], Counter()
    for index, op in enumerate(graph.operators):
        if model.operatorCodes[op.opcodeIndex].builtinCode != s.BuiltinOperator.FULLY_CONNECTED:
            continue
        output = int(op.outputs[0])
        consumers = users.get(output, [])
        if len(consumers) != 1 or output in graph.outputs:
            skips["multiple_consumers_or_public_output"] += 1
            continue
        add_index = consumers[0]
        add = graph.operators[add_index]
        if model.operatorCodes[add.opcodeIndex].builtinCode != s.BuiltinOperator.ADD:
            skips["consumer_not_ADD"] += 1
            continue
        if len(op.inputs) != 3 or op.inputs[2] != -1 or len(add.outputs) != 1:
            skips["existing_bias_or_multi_output"] += 1
            continue
        if (
            op.builtinOptions.fusedActivationFunction != s.ActivationFunctionType.NONE
            or add.builtinOptions.fusedActivationFunction != s.ActivationFunctionType.NONE
        ):
            skips["non_identity_activation"] += 1
            continue
        other = [int(t) for t in add.inputs if int(t) != output]
        if len(other) != 1:
            skips["not_one_other_input"] += 1
            continue
        bias_id = other[0]
        bias = graph.tensors[bias_id]
        weight = graph.tensors[int(op.inputs[1])]
        old_out = graph.tensors[output]
        new_out = graph.tensors[int(add.outputs[0])]
        data = model.buffers[bias.buffer].data
        if (
            bias.type != s.TensorType.FLOAT32
            or len(bias.shape) != 1
            or bias.shape[0] != weight.shape[0]
            or data is None
            or len(data) != int(bias.shape[0]) * 4
            or bias.isVariable
        ):
            skips["not_constant_float32_1d_output_bias"] += 1
            continue
        if (
            old_out.type != s.TensorType.FLOAT32
            or new_out.type != s.TensorType.FLOAT32
            or not np.array_equal(old_out.shape, new_out.shape)
        ):
            skips["output_shape_or_dtype_mismatch"] += 1
            continue
        assert index < add_index
        op.inputs = op.inputs.copy()
        op.inputs[2] = bias_id
        op.outputs = add.outputs.copy()
        deleted.add(add_index)
        changes.append(
            {
                "fc_index": index,
                "removed_ADD_index": add_index,
                "bias_tensor": bias_id,
                "bias_name": bias.name.decode(),
                "old_fc_output": output,
                "new_fc_output": int(op.outputs[0]),
            }
        )
    assert len(changes) == 92, (len(changes), skips)
    graph.operators = [op for index, op in enumerate(graph.operators) if index not in deleted]
    assert before_counts["ADD"] - counts(model)["ADD"] == len(changes)
    folder = ART / "refinement"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "encoder_fc_bias.tflite"
    assert not output.exists()
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    output.write_bytes(bytes(builder.Output()))
    assert info(original) == original_record
    baseline, candidate = LiteRunner(original), LiteRunner(output)
    captures = [
        p
        for row in read(DAY20 / "results/onnx_baseline.json")["smoke_audio"]
        for p in row["captures"]
    ]
    parity = []
    for capture in captures:
        with np.load(ROOT / capture["path"]) as data:
            feeds = {k: v.copy() for k, v in data.items()}
        a, b = baseline.run(feeds), candidate.run(feeds)
        row = {name: numeric(a[name], b[name]) for name in a}
        parity.append(row)
    passed = all(v["allclose_rtol1e_4_atol1e_4"] for row in parity for v in row.values())
    counters_exact = all(
        v["exact"] for row in parity for n, v in row.items() if n.startswith("new_cached_len_")
    )
    old_raw = paths("baseline")["encoder"].read_bytes()
    old_model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(old_raw, 0))
    write(
        target,
        {
            "finished_at_utc": now(),
            "original": original_record,
            "model": info(output),
            "changes": changes,
            "fused_ADD_count": len(changes),
            "skips": dict(skips),
            "before_op_counts": before_counts,
            "after_op_counts": counts(model),
            "parity_on15_state_fixtures": parity,
            "tensor_gate_passed": passed,
            "counters_exact": counters_exact,
            "serialization": {
                "baseline": storage(old_model, old_raw),
                "FC": storage(s.ModelT.InitFromObj(s.Model.GetRootAsModel(raw, 0)), raw),
                "FC_bias": storage(model, output.read_bytes()),
            },
            "scope": "Algebraic FC+constant ADD bias fusion;rounding may differ;no promise of fewer runtime kernels until measured. No weight change or public state/API change.",
            "source_code": info(__file__),
        },
    )
    print(
        "BIAS REWRITE",
        len(changes),
        "tensor_gate",
        passed,
        "counters_exact",
        counters_exact,
        "skips",
        dict(skips),
        flush=True,
    )


if __name__ == "__main__":
    main()
