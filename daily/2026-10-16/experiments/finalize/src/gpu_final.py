"""Isolated CPU/CUDA comparison using the same ORT GPU build and frozen data."""

import argparse
import gc
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from common import ART, DAY15, RESULTS, ROOT, checked, info, read, write

GPU_ART = ART / "gpu"
GPU_RESULTS = RESULTS / "gpu"


def runtime_setup():
    sys.path.insert(0, str(ROOT / "quantization"))
    from qlab.gpu_runtime import configure_runtime

    return configure_runtime()


def model_paths(task):
    if task == "mlsd":
        base = read(DAY15 / "results/research/deployment.json")
        selected = read(RESULTS / "mlsd_selection.json")
        return {
            "fp32": checked(base["artifacts"]["original_fp32_model"]),
            "fp16": GPU_ART / "mlsd_fp16.onnx",
            "qat_41_integer": checked(base["artifacts"]["integer_model"]),
            "qat_qdq": checked(selected["candidates"][selected["selected"]]["model"]),
        }
    base = read(RESULTS / "text_export.json")
    selected = read(RESULTS / "text_selection.json")
    return {
        "fp32": checked(base["models"]["teacher_fp32"]["model"]),
        "fp16": GPU_ART / "text_fp16.onnx",
        "qat_static": checked(selected["candidates"][selected["selected"]]["model"]),
    }


def prepare():
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    GPU_ART.mkdir(parents=True, exist_ok=True)
    GPU_RESULTS.mkdir(parents=True, exist_ok=True)
    dlls = runtime_setup()
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    write(
        GPU_RESULTS / "environment.json",
        {
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "ort": ort.__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
            "available_providers": ort.get_available_providers(),
            "nvidia_smi": smi.stdout.strip(),
            "preloaded_dlls": dlls,
            "scope": "Same ORT 1.20.2 GPU wheel used for CPU and CUDA controls; separate from earlier ORT 1.30 CPU run",
        },
    )
    for task in ["mlsd", "text"]:
        paths = model_paths(task)
        if paths["fp16"].exists():
            raise FileExistsError("Preserve previously converted FP16 artifact")
        source = onnx.load(paths["fp32"])
        model = convert_float_to_float16(
            source, keep_io_types=True, min_positive_val=1e-7, max_finite_val=1e4
        )
        OnnxModel(model).topological_sort(is_deterministic=True)
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, paths["fp16"])
        write(
            GPU_RESULTS / f"{task}_fp16_conversion.json",
            {
                "source": info(paths["fp32"]),
                "model": info(paths["fp16"]),
                "settings": {
                    "keep_io_types": True,
                    "min_positive_val": 1e-7,
                    "max_finite_val": 1e4,
                    "default_blocked_ops": True,
                },
                "initializer_types": dict(
                    Counter(
                        onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer
                    )
                ),
                "scope": "Floating precision conversion, not QAT or INT8; default blocked ops remain FP32",
                "source_code": info(__file__),
            },
        )
        print("CONVERTED", task, flush=True)


def create_session(path, device, profile=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.log_severity_level = 3
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    providers = (
        ["CPUExecutionProvider"]
        if device == "cpu"
        else [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "use_tf32": "0",
                    "do_copy_in_default_stream": "1",
                    "gpu_mem_limit": str(2 * 1024**3),
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
    )
    result = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    result.disable_fallback()
    if device == "cuda" and "CUDAExecutionProvider" not in result.get_providers():
        raise RuntimeError("CUDA initialization failed; refusing a CPU-only result labelled GPU")
    return result


def feeds(task):
    manifest = read(RESULTS / f"{task}_manifest.json")
    loaded = np.load(checked(manifest["inputs"]))
    if task == "mlsd":
        return [{"input": x[None]} for x in loaded], manifest
    return [
        {
            "input_ids": loaded["input_ids"][i : i + 1],
            "attention_mask": loaded["attention_mask"][i : i + 1],
        }
        for i in range(len(loaded["labels"]))
    ], manifest


def worker(task, variant, device):
    import metrics
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "quantization"))
    from qlab.gpu_runtime import summarize_profile

    target = GPU_RESULTS / f"{task}_{variant}_{device}.json"
    if target.exists():
        raise FileExistsError("GPU comparison result already recorded")
    record = {
        "task": task,
        "variant": variant,
        "device": device,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "ort": ort.__version__,
        "status": "failed",
        "source_code": info(__file__),
    }
    try:
        record["preloaded_dlls"] = runtime_setup()
        path = model_paths(task)[variant]
        record["model"] = info(path)
        examples, manifest = feeds(task)
        record["manifest"] = info(RESULTS / f"{task}_manifest.json")
        indices = np.linspace(0, len(examples) - 1, 16, dtype=int)
        benchmark_inputs = [examples[int(i)] for i in indices]
        live = create_session(path, device)
        record["providers"] = live.get_providers()
        record["provider_options"] = live.get_provider_options()
        for feed in benchmark_inputs:
            live.run(None, feed)
        timings = []
        for _ in range(5):
            for feed in benchmark_inputs:
                start = time.perf_counter_ns()
                live.run(None, feed)
                timings.append((time.perf_counter_ns() - start) / 1e6)
        output = []
        for index, feed in enumerate(examples):
            result = live.run(None, feed)[0]
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite runtime output")
            output.append(result)
            if (index + 1) % 200 == 0:
                print("EVALUATED", task, variant, device, index + 1, flush=True)
        output = np.concatenate(output)
        if task == "mlsd":
            output = output.transpose(0, 2, 3, 1)
        del live
        gc.collect()
        output_path = GPU_ART / f"{task}_{variant}_{device}.npy"
        np.save(output_path, output, allow_pickle=False)
        base = read(RESULTS / f"{task}_final.json")
        baseline_name = "original_fp32" if task == "mlsd" else "teacher_fp32"
        reference = np.load(checked(base["models"][baseline_name]["output"]))
        record["parity_vs_ort130_cpu_fp32"] = metrics.numerical(reference, output)
        if task == "mlsd":
            record["quality"] = metrics.line_metrics(
                output, read(checked(manifest["ground_truth"]))
            )
        else:
            labels = np.load(checked(manifest["inputs"]))["labels"]
            correct = int(np.count_nonzero(output.argmax(-1) == labels))
            record["quality"] = {
                "correct": correct,
                "samples": len(labels),
                "accuracy_percent": correct / len(labels) * 100,
                "label_agreement_vs_ort130_cpu_fp32": float(
                    np.mean(output.argmax(-1) == reference.argmax(-1))
                ),
            }
        profiler = create_session(path, device, GPU_ART / f"profile_{task}_{variant}_{device}")
        profiler.run(None, examples[0])
        profile_path = Path(profiler.end_profiling())
        profile = summarize_profile(read(profile_path))
        del profiler
        record.update(
            {
                "status": "ok"
                if device == "cpu" or profile["cuda_compute_nodes"] > 0
                else "cpu_only_partition",
                "output": info(output_path),
                "profile": {"file": info(profile_path), **profile},
                "benchmark": {
                    "samples_ms": timings,
                    "runs": len(timings),
                    "median_ms": float(np.median(timings)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "input_indices": indices.tolist(),
                    "protocol": "One isolated process/session per variant/device; ORT 1.20.2, 4 intra/1 inter threads, sequential ALL, spinning disabled; 16 warmups then 5x16 timing samples; batch1; profile in separate session after timing/evaluation",
                    "scope": "Synchronous session.run, CPU NumPy IO includes H2D/D2H and API overhead; excludes load/preprocess/decode; CUDA TF32 disabled; 2GiB arena limit and heuristic cuDNN workspace policy",
                },
                "scope": "Same frozen 54-image/772-sentence set used for runtime comparison; no new tuning or independent test-set claim",
            }
        )
        print(
            "RESULT",
            task,
            variant,
            device,
            record["status"],
            record["benchmark"]["median_ms"],
            "CUDA nodes",
            profile["cuda_compute_nodes"],
            "CPU nodes",
            len(profile["cpu_nodes"]),
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 -- isolate runtime failures per experiment
        record["error"] = str(error)
        record["error_type"] = type(error).__name__
        record.pop("benchmark", None)
        print("FAILED", task, variant, device, type(error).__name__, str(error)[:400], flush=True)
    write(target, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--task", choices=["mlsd", "text"])
    parser.add_argument(
        "--variant", choices=["fp32", "fp16", "qat_41_integer", "qat_qdq", "qat_static"]
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.task and args.variant and args.device:
        worker(args.task, args.variant, args.device)
    else:
        parser.error("Use --prepare or provide --task, --variant and --device")
