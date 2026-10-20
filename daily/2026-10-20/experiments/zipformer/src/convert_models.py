"""Log raw conversion first, then optional shape specialization/graph optimization."""

import argparse
import importlib.metadata
import shutil
import subprocess
import time
from collections import Counter

import onnx
from support import (
    ART,
    RESULTS,
    OnnxRunner,
    info,
    now,
    source,
    write,
    zeros_for_model,
)


def specialized(component, optimize=False):
    import onnxoptimizer
    import onnxsim

    model = onnx.load(source(component))
    runtime = OnnxRunner(source(component))
    outputs = runtime.run(zeros_for_model(source(component)))
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name in outputs:
            dims = outputs[value.name].shape
        else:
            dims = zeros_for_model(source(component))[value.name].shape
        del value.type.tensor_type.shape.dim[:]
        for size in dims:
            value.type.tensor_type.shape.dim.add().dim_value = size
    before = dict(Counter(n.op_type for n in model.graph.node))
    passes = []
    if optimize:
        model, checked = onnxsim.simplify(model, check_n=3)
        if not checked:
            raise ValueError("onnxsim verification failed")
        available = set(onnxoptimizer.get_available_passes())
        passes = [
            p
            for p in [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "fuse_consecutive_transposes",
                "eliminate_deadend",
                "extract_constant_to_initializer",
                "eliminate_unused_initializer",
            ]
            if p in available
        ]
        model = onnxoptimizer.optimize(model, passes)
    onnx.checker.check_model(model, full_check=True)
    return model, before, passes


def main(args):
    variant = args.variant
    for component in args.components:
        record_path = RESULTS / f"conversion_{variant}_{component}.json"
        if record_path.exists():
            print("PRESERVED", record_path.name, flush=True)
            continue
        folder = ART / "conversion" / variant / component
        folder.mkdir(parents=True, exist_ok=True)
        input_path = folder / "input.onnx"
        before_sha = info(source(component))
        changes = {"author": "Codex", "claude_execution": False}
        if variant == "raw":
            shutil.copy2(source(component), input_path)
        else:
            model, original_ops, passes = specialized(component, variant == "optimized")
            onnx.save(model, input_path)
            changes.update(
                original_ops=original_ops,
                specialized_ops=dict(Counter(n.op_type for n in model.graph.node)),
                passes=passes,
                shape_policy="Replace dynamic N by1 in its actual axis; output shapes obtained from original ONNX zero-state execution; state INT64 retained",
            )
        cmd = [
            "onnx2tf",
            "-i",
            str(input_path),
            "-o",
            str(folder / "output"),
            "-v",
            "error",
            "-tb",
            "flatbuffer_direct",
        ]
        if variant != "raw":
            model = onnx.load(input_path)
            cmd.extend(["-kat", *[x.name for x in model.graph.input]])
        started = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
            output, returncode = result.stdout + "\n" + result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            output, returncode = str(error), -1
        elapsed = time.perf_counter() - started
        log = folder / "conversion.log"
        log.write_text(output, encoding="utf-8", newline="\n")
        models = [info(p) for p in folder.rglob("*.tflite")]
        assert info(source(component)) == before_sha, "Original source modified"
        record = {
            "finished_at_utc": now(),
            "variant": variant,
            "component": component,
            "backend": "onnx2tf flatbuffer_direct; no TensorFlow conversion",
            "onnx2tf": importlib.metadata.version("onnx2tf"),
            "litert": importlib.metadata.version("ai-edge-litert"),
            "command": cmd,
            "returncode": returncode,
            "elapsed_s": elapsed,
            "input": info(input_path),
            "original": before_sha,
            "changes": changes,
            "models": models,
            "log": info(log),
            "last_log_lines": output.splitlines()[-30:],
            "source_code": info(__file__),
        }
        write(record_path, record)
        print(
            "CONVERT",
            variant,
            component,
            "code",
            returncode,
            "seconds",
            round(elapsed, 2),
            "models",
            [x["path"] for x in models],
            flush=True,
        )
        if returncode:
            print("\n".join(output.splitlines()[-10:]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["raw", "static", "optimized"], required=True)
    parser.add_argument("--components", nargs="+", default=["decoder", "joiner", "encoder"])
    main(parser.parse_args())
