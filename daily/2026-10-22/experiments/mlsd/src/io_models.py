"""Add native float input quantization and optional native NCHW output to C1."""

import argparse

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from mlsd22 import ART, RESULTS, checked, info, now, read, references, write


def serialize(model):
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def materialize(model, raw):
    count, total = 0, 0
    for buffer in model.buffers:
        if not buffer.offset and not buffer.size:
            continue
        assert buffer.offset > 0 and buffer.size > 0
        assert buffer.data is None or len(buffer.data) == 0
        start, size = int(buffer.offset), int(buffer.size)
        assert start + size <= len(raw)
        buffer.data = np.frombuffer(raw[start : start + size], np.uint8).copy()
        buffer.offset = buffer.size = 0
        count += 1
        total += size
    for graph in model.subgraphs:
        assert all(
            not o.largeCustomOptionsOffset and not o.largeCustomOptionsSize for o in graph.operators
        )
    return {"external_buffers": count, "payload_bytes_materialized": total}


def payloads(model, raw):
    return [
        raw[int(b.offset) : int(b.offset + b.size)]
        if b.offset
        else b""
        if b.data is None
        else bytes(b.data)
        for b in model.buffers
    ]


def tensor(name, shape, dtype, buffer=0):
    value = s.TensorT()
    value.name = name.encode()
    value.shape = np.array(shape, dtype=np.int32)
    value.shapeSignature = value.shape.copy()
    value.type = dtype
    value.buffer = buffer
    value.hasRank = True
    return value


def opcode(model, builtin, version):
    value = s.OperatorCodeT()
    value.builtinCode = builtin
    value.deprecatedBuiltinCode = builtin
    value.version = version
    model.operatorCodes.append(value)
    return len(model.operatorCodes) - 1


def quantize_op(code, input_id, output_id):
    value = s.OperatorT()
    value.opcodeIndex = code
    value.inputs = np.array([input_id], dtype=np.int32)
    value.outputs = np.array([output_id], dtype=np.int32)
    value.builtinOptionsType = s.BuiltinOptions.QuantizeOptions
    value.builtinOptions = s.QuantizeOptionsT()
    return value


def miniature(scale, zero, length):
    model = s.ModelT()
    model.version = 3
    model.description = b"Single QUANTIZE probe with unchanged C1 input parameters"
    model.buffers = [s.BufferT()]
    model.operatorCodes = []
    code = opcode(model, s.BuiltinOperator.QUANTIZE, 2)
    graph = s.SubGraphT()
    graph.name = b"quantize_probe"
    graph.inputs = np.array([0], np.int32)
    graph.outputs = np.array([1], np.int32)
    output = tensor("quantized", [length], s.TensorType.INT8)
    output.quantization = s.QuantizationParametersT()
    output.quantization.scale = np.array([scale], np.float32)
    output.quantization.zeroPoint = np.array([zero], np.int64)
    graph.tensors = [tensor("float_input", [length], s.TensorType.FLOAT32), output]
    graph.operators = [quantize_op(code, 0, 1)]
    model.subgraphs = [graph]
    return serialize(model)


def build(name):
    target = RESULTS / ("model_" + name + ".json")
    if target.exists():
        raise FileExistsError(target)
    assert name in ["C1_roundtrip", "M1", "M2"]
    if name == "M2":
        assert read(RESULTS / "quality_M1.json")["quality_passed"]
        source_record = read(RESULTS / "model_M1.json")["model"]
    elif name == "M1":
        assert read(RESULTS / "roundtrip_control.json")["output_exact"]
        source_record = read(RESULTS / "model_C1_roundtrip.json")["model"]
    else:
        source_record = references()["C1"]
    path = checked(source_record)
    raw = path.read_bytes()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(raw, 0))
    before_payloads = payloads(model, raw)
    materialization = materialize(model, raw)
    assert len(model.subgraphs) == len(model.signatureDefs) == 1
    graph = model.subgraphs[0]
    signature = model.signatureDefs[0]
    assert len(graph.inputs) == len(graph.outputs) == 1
    assert signature.subgraphIndex == 0
    original_buffers = len(model.buffers)
    if name == "C1_roundtrip":
        change = {
            "added": "Identity control:materialize external buffers and serialize,without adding operators or changing I/O"
        }
    elif name == "M1":
        old_input_id = int(graph.inputs[0])
        old_input = graph.tensors[old_input_id]
        assert old_input.type == s.TensorType.INT8
        assert old_input.shape.tolist() == [1, 4, 512, 512]
        assert len(old_input.quantization.scale) == len(old_input.quantization.zeroPoint) == 1
        new_input_id = len(graph.tensors)
        graph.tensors.append(tensor("input_float", old_input.shape, s.TensorType.FLOAT32))
        code = opcode(model, s.BuiltinOperator.QUANTIZE, 2)
        graph.operators.insert(0, quantize_op(code, new_input_id, old_input_id))
        graph.inputs = np.array([new_input_id], np.int32)
        assert signature.inputs[0].tensorIndex == old_input_id
        signature.inputs[0].tensorIndex = new_input_id
        change = {
            "old_INT8_input_tensor": old_input_id,
            "new_float_input_tensor": new_input_id,
            "scale": float(old_input.quantization.scale[0]),
            "zero_point": int(old_input.quantization.zeroPoint[0]),
            "added": "One float input tensor and QUANTIZEv2;original internal tensors/weights/scales unchanged",
        }
    else:
        old_output_id = int(graph.outputs[0])
        old_output = graph.tensors[old_output_id]
        assert old_output.type == s.TensorType.FLOAT32
        assert old_output.shape.tolist() == [1, 256, 256, 5]
        buffer = s.BufferT()
        buffer.data = np.array([0, 3, 1, 2], np.int32).view(np.uint8)
        model.buffers.append(buffer)
        perm_id = len(graph.tensors)
        graph.tensors.append(
            tensor("output_NCHW_permutation", [4], s.TensorType.INT32, original_buffers)
        )
        new_output_id = len(graph.tensors)
        graph.tensors.append(tensor("output_NCHW", [1, 5, 256, 256], s.TensorType.FLOAT32))
        value = s.OperatorT()
        value.opcodeIndex = opcode(model, s.BuiltinOperator.TRANSPOSE, 1)
        value.inputs = np.array([old_output_id, perm_id], np.int32)
        value.outputs = np.array([new_output_id], np.int32)
        value.builtinOptionsType = s.BuiltinOptions.TransposeOptions
        value.builtinOptions = s.TransposeOptionsT()
        graph.operators.append(value)
        graph.outputs = np.array([new_output_id], np.int32)
        assert signature.outputs[0].tensorIndex == old_output_id
        signature.outputs[0].tensorIndex = new_output_id
        change = {
            "old_NHWC_output_tensor": old_output_id,
            "new_NCHW_output_tensor": new_output_id,
            "permutation": [0, 3, 1, 2],
            "added": "One TRANSPOSEv1 with16byte permutation;original graph tensors/weight buffers unchanged",
        }
    folder = ART / "models"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (name + ".tflite")
    assert not output.exists()
    output.write_bytes(serialize(model))
    after_raw = output.read_bytes()
    after = s.ModelT.InitFromObj(s.Model.GetRootAsModel(after_raw, 0))
    assert payloads(after, after_raw)[: len(before_payloads)] == before_payloads
    write(
        target,
        {
            "finished_at_utc": now(),
            "name": name,
            "model": info(output),
            "source_model": source_record,
            "change": change,
            "materialization": materialization,
            "original_buffer_payloads_exact": True,
            "source_code": info(__file__),
            "recalibrated": False,
            "precision_changed": False,
        },
    )
    print("BUILT", name, info(output), change, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["C1_roundtrip", "M1", "M2"])
    build(parser.parse_args().name)
