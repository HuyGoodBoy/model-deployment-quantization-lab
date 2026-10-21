"""One float application contract for single-output and split-head LiteRT models."""

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def quantize(value, details):
    if details["dtype"] == np.float32:
        return value
    scale, zero = details["quantization"]
    assert scale > 0
    limits = np.iinfo(details["dtype"])
    return np.clip(np.rint(value / scale) + zero, limits.min, limits.max).astype(details["dtype"])


def dequantize(value, details):
    if np.issubdtype(value.dtype, np.floating):
        return value
    scale, zero = details["quantization"]
    assert scale > 0
    return (value.astype(np.float32) - zero) * scale


class Runner:
    def __init__(self, path, threads=4):
        self.runtime = Interpreter(model_path=str(path), num_threads=threads)
        self.runtime.allocate_tensors()
        signatures = self.runtime.get_signature_list()
        assert len(signatures) == 1
        self.signature = self.runtime.get_signature_runner(next(iter(signatures)))
        self.inputs = self.signature.get_input_details()
        self.outputs = self.signature.get_output_details()
        assert list(self.inputs) == ["input"]
        assert set(self.outputs) in [{"output"}, {"center", "displacement"}]

    def run(self, image):
        assert image.shape == (4, 512, 512) and image.dtype == np.float32
        output = self.signature(input=quantize(image[None], self.inputs["input"]))
        output = {k: dequantize(v, self.outputs[k]) for k, v in output.items()}
        if "output" in output:
            value = output["output"]
        else:
            for name, channels in [("center", 1), ("displacement", 4)]:
                if output[name].shape == (1, 256, 256, channels):
                    output[name] = output[name].transpose(0, 3, 1, 2)
            assert output["center"].shape == (1, 1, 256, 256)
            assert output["displacement"].shape == (1, 4, 256, 256)
            value = np.concatenate([output["center"], output["displacement"]], axis=1)
        assert value.shape == (1, 5, 256, 256)
        return value


def infer(runner, inputs):
    values = np.concatenate([runner.run(x) for x in inputs]).transpose(0, 2, 3, 1)
    assert np.isfinite(values).all()
    return values


def io_details(runner):
    return {
        kind: {
            name: {
                "dtype": str(row["dtype"]),
                "shape": row["shape"].tolist(),
                "scale": row["quantization"][0],
                "zero_point": row["quantization"][1],
            }
            for name, row in values.items()
        }
        for kind, values in [("inputs", runner.inputs), ("outputs", runner.outputs)]
    }
