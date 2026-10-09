"""Record default delegate boundaries after measurements; no timing claims."""

import argparse
import collections
import importlib.metadata
import os
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
sys.path.insert(0, str(ROOT / "quantization"))
from qlab.common import sha256, write_json


def main(runtime):
    os.environ.update(TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="-1")
    if runtime == "new":
        from ai_edge_litert.interpreter import Interpreter

        version = importlib.metadata.version("ai-edge-litert")
    else:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        version = tf.__version__
    path = ROOT / "quantization/artifacts/cv/tflite_dynamic.tflite"
    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()

    def describe(index):
        d = details.get(int(index), {})
        return {
            "index": int(index),
            "name": d.get("name"),
            "dtype": np.dtype(d["dtype"]).name if "dtype" in d else None,
        }

    boundaries = [
        {
            "index": int(op["index"]),
            "inputs": [describe(i) for i in op["inputs"]],
            "outputs": [describe(i) for i in op["outputs"]],
        }
        for op in ops
        if op["op_name"] == "DELEGATE"
    ]
    write_json(
        TASK / f"results/run-01/delegate_audit_{runtime}.json",
        {
            "runtime": version,
            "model_sha256": sha256(path),
            "threads": 4,
            "operator_counts": dict(collections.Counter(op["op_name"] for op in ops)),
            "delegate_boundaries": boundaries,
            "scope": "private Interpreter introspection, original nodes plus delegate nodes; not kernel timing or verified execution-plan coverage",
        },
    )
    print(runtime, version, "delegate partitions", len(boundaries), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", choices=("old", "new"), required=True)
    main(p.parse_args().runtime)
