"""Export cung model goc sang ONNX va TFLite, giu FP32."""
import hashlib

import numpy as np

from .common import ARTIFACTS, INPUT_SHAPE, tensorflow, write_json


def main():
    tf = tensorflow()
    import onnx
    import tf2onnx

    original = ARTIFACTS / "mobilenetv2.keras"
    model = tf.keras.models.load_model(original, compile=False)
    signature = [tf.TensorSpec(INPUT_SHAPE, tf.float32, name="images")]

    @tf.function(input_signature=signature, autograph=False)
    def serving(x):
        return tf.identity(model(x, training=False), name="probabilities")

    onnx_path = ARTIFACTS / "mobilenetv2.onnx"
    print("Export ONNX (opset=13, FP32)...", flush=True)
    graph, _ = tf2onnx.convert.from_function(
        serving, input_signature=signature, opset=13, output_path=str(onnx_path))
    onnx.checker.check_model(graph, full_check=True)
    if graph.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX input khong phai FP32")
    if graph.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX output khong phai FP32")
    for tensor in graph.graph.initializer:
        if tensor.data_type in {onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}:
            raise ValueError(f"ONNX co tensor khong phai FP32: {tensor.name}")

    print("Export TFLite (FP32, optimizations=[] se khong quantize)...", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_path = ARTIFACTS / "mobilenetv2.tflite"
    tflite_path.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    if any(d["dtype"] in (np.float16, np.float64) for d in details):
        raise ValueError("TFLite co floating tensor khong phai FP32")
    if any(len(d["quantization_parameters"]["scales"]) for d in details):
        raise ValueError("Phat hien tensor TFLite da quantize")
    for d in interpreter.get_input_details() + interpreter.get_output_details():
        if d["dtype"] != np.float32:
            raise ValueError("TFLite I/O khong phai FP32")
    write_json(ARTIFACTS / "conversion_info.json", {
        "tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__,
        "onnx": onnx.__version__, "opset": 13, "precision": "FP32",
        "quantized": False, "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "files": {p.name: {"bytes": p.stat().st_size,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in (onnx_path, tflite_path)},
    })
    print("Converted and checked both formats.", flush=True)


if __name__ == "__main__":
    main()
