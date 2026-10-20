"""Always convert a disposable copy: onnx2tf may simplify/overwrite its input."""

import importlib.metadata
import shutil
import subprocess

from mlsd_support import ART, RESULTS, info, model_paths, now, sha, write


def main():
    target = RESULTS / "float_conversion_reproduction.json"
    if target.exists():
        raise FileExistsError(target)
    source = model_paths()["qat_weights_fp32"]
    before = sha(source)
    folder = ART / "conversion_reproduced"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "qat_weights_fp32.onnx"
    shutil.copy2(source, path)
    original_copy = info(path)
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
    assert run.returncode == 0, run.stderr[-2000:]
    assert sha(source) == before, "Read-only source was modified"
    output = folder / "output/qat_weights_fp32_float32.tflite"
    earlier = ART / "float_conversion/qat_weights_fp32_float32.tflite"
    reference_existed = earlier.exists()
    if not reference_existed:
        earlier.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, earlier)
    assert sha(output) == sha(earlier), "Reproduced model differs"
    write(
        target,
        {
            "finished_at_utc": now(),
            "command": command,
            "source_unchanged": info(source),
            "input_copy_before": original_copy,
            "input_copy_after": info(path),
            "model": info(output),
            "byte_identical_to_evaluated_model": info(earlier),
            "reference_existed_before_conversion": reference_existed,
            "log": info(log),
            "environment": {
                p: importlib.metadata.version(p)
                for p in ["onnx2tf", "ai-edge-litert", "onnxruntime"]
            },
            "incident": "The first direct conversion simplified the previous-day ONNX file in place. The original was re-exported with the original code/checkpoint and restored only after SHA256 exactly matched the pre-existing exports.json (ce7b26475cb5eee7e9f5b34d873b51e7401c3cc9d9584e415b5a186e0f49dd1e). The altered copy is preserved in today's artifacts. This reproduction uses a disposable copy and verifies the source remains byte-identical.",
            "source_code": info(__file__),
        },
    )
    print("REPRODUCED", sha(output), flush=True)


if __name__ == "__main__":
    main()
