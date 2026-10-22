"""Fixed-shape public direct calls, with the same model and input adaptation."""

from asr22 import LiteRunner, paths


class DirectCall:
    def __init__(self, runtime, inputs, outputs):
        self.runtime = runtime
        self.inputs = [(name, int(d["index"]), tuple(d["shape"])) for name, d in inputs.items()]
        self.outputs = [(name, int(d["index"])) for name, d in outputs.items()]

    def __call__(self, **values):
        if len(values) != len(self.inputs):
            raise ValueError("Fixed-shape direct call received wrong number of inputs")
        for name, index, shape in self.inputs:
            value = values[name]
            if value.shape != shape:
                raise ValueError(f"Fixed-shape direct call: {name}: {value.shape} != {shape}")
            self.runtime.set_tensor(index, value)
        self.runtime.invoke()
        return {name: self.runtime.get_tensor(index) for name, index in self.outputs}


class DirectRunner(LiteRunner):
    def __init__(self, path):
        super().__init__(path)
        assert self.runtime.num_subgraphs() == 1
        outputs = self.function.get_output_details()
        # The SignatureRunner holds an extra reference to the native wrapper.
        # Releasing it keeps Interpreter.invoke()'s normal safety guard valid.
        del self.function
        self.function = DirectCall(self.runtime, self.input_details, outputs)


def create(variant):
    assert variant in ["Z0", "Z1"]
    runner = LiteRunner if variant == "Z0" else DirectRunner
    return {name: runner(path) for name, path in paths("fc").items()}
