"""Run Duke's pinned scripts from a read-only checkout, writing only our artifacts.

Supports full pipeline in a compatible environment and an explicit ONNX-only
fallback on Windows. Original source files are never edited. Evaluation uses
the original q4 functions, but isolates failures so one unsupported model does
not hide results for the others. All adaptations are recorded in evidence.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
REVISION = "536b9ae1f9e1b3829ac4776a98fb05677df9b6aa"
SOURCE = ROOT / ".cache/duke-readonly/4-quantization"
PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "litert-torch",
    "ai-edge-litert",
    "ai-edge-quantizer",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def main(args):
    if not SOURCE.is_dir():
        raise RuntimeError("Clone the pinned peer repo into .cache/duke-readonly first.")
    artifacts = TASK / "artifacts" / args.run_id
    results = TASK / "results" / args.run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(ROOT / ".cache/torch-peer"),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    inventory = {}
    for name in PACKAGES:
        try:
            inventory[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            inventory[name] = None
    source_hashes = {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()}
    lock = json.loads((TASK / "peer-sources.lock.json").read_text(encoding="utf-8"))
    normalized_hashes = {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in SOURCE.iterdir()
        if p.is_file()
    }
    if lock["revision"] != REVISION or normalized_hashes != lock["files_lf"]:
        raise ValueError("Peer checkout differs from the pinned source hashes.")
    evidence_path = results / "peer_reproduction.json"
    metadata = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.exists()
        else {
            "peer_revision": REVISION,
            "source_sha256": source_hashes,
            "source_edit_policy": "read only; execution cwd is our artifacts directory",
            "python": sys.version,
            "platform": platform.platform(),
            "packages": inventory,
            "run_id": args.run_id,
            "mode": args.mode,
            "hardware_match_required": False,
            "protocol": {"threads": 4, "warmup": 10, "runs": 100, "batch": 1},
            "adaptations": [],
            "stages": [],
            "results": [],
        }
    )
    if metadata["source_sha256"] != source_hashes or metadata["mode"] != args.mode:
        raise ValueError("Source or mode changed; use a new run ID.")
    metadata["hardware"] = {
        "architecture": platform.machine(),
        "visible_logical_cpus": os.cpu_count(),
    }
    if Path("/proc/cpuinfo").exists():
        metadata["hardware"]["cpu_model"] = next(
            (
                l.split(":", 1)[1].strip()
                for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")
            ),
            None,
        )
    for name in ["cpu.max", "memory.max"]:
        p = Path("/sys/fs/cgroup") / name
        if p.exists():
            metadata["hardware"][name] = p.read_text().strip()

    def checkpoint():
        write(results / "peer_reproduction.json", metadata)

    def run_stage(name, command):
        log = results / (name + ".log")
        print(f"Running {name}; log={log}", flush=True)
        started = time.perf_counter()
        with log.open("w", encoding="utf-8") as f:
            process = subprocess.run(
                command, cwd=artifacts, env=env, stdout=f, stderr=subprocess.STDOUT, check=False
            )
        entry = {
            "name": name,
            "command": [str(x) for x in command],
            "returncode": process.returncode,
            "seconds": time.perf_counter() - started,
        }
        metadata["stages"].append(entry)
        checkpoint()
        print(f"{name}: exit {process.returncode}", flush=True)
        return process.returncode == 0

    if args.stage in ("all", "prepare"):
        print("Selecting the original 100 calibration / 300 evaluation JPEGs...", flush=True)
        archive = ROOT / "quantization/.cache/imagenette2-160.tgz"
        if not archive.is_file():
            raise FileNotFoundError(
                "Prepare the Imagenette archive using the baseline download stage first."
            )
        # Use the same lexicographically first JPEGs as original load_images.
        tree = ast.parse((SOURCE / "q1_prepare.py").read_text(encoding="utf-8-sig"))
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CLASSES" for t in n.targets)
        )
        classes = ast.literal_eval(assignment.value)
        chosen = []
        with tarfile.open(archive, "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            for split, count in [("train", 10), ("val", 30)]:
                for folder in classes:
                    prefix = f"imagenette2-160/{split}/{folder}/"
                    names = sorted(
                        n for n in members if n.startswith(prefix) and n.endswith(".JPEG")
                    )[:count]
                    if len(names) != count:
                        raise RuntimeError(f"Incomplete source dataset: {split}/{folder}")
                    chosen.extend(names)
            # Read gzip members in physical order to avoid repeated decompression.
            # Selected names and original load_images ordering remain unchanged.
            for name in sorted(chosen, key=lambda n: members[n].offset_data):
                dest = (artifacts / name).resolve()
                if not dest.is_relative_to(artifacts.resolve()):
                    raise ValueError("Unsafe archive path")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(members[name]).read())
        metadata["data"] = {
            "archive_sha256": sha(archive),
            "selection": "sorted first 10 train / 30 val per class",
            "files": [{"path": n, "sha256": sha(artifacts / n)} for n in chosen],
        }
        if args.mode == "full":
            ready = run_stage("q1_prepare", [sys.executable, str(SOURCE / "q1_prepare.py")])
        else:
            # Keep the peer source untouched; explicitly remove just unsupported
            # import/conversion statements in a separately saved AST adaptation.
            removed = []
            filtered = []
            for node in tree.body:
                snippet = ast.get_source_segment(
                    (SOURCE / "q1_prepare.py").read_text(encoding="utf-8-sig"), node
                )
                skip = isinstance(node, ast.Import) and any(
                    a.name == "litert_torch" for a in node.names
                )
                skip = skip or (
                    isinstance(node, ast.Expr)
                    and snippet
                    and snippet.startswith("litert_torch.convert(")
                )
                if skip:
                    removed.append(snippet)
                else:
                    filtered.append(node)
            adapted = artifacts / "q1_onnx_only_adapted.py"
            adapted.write_text(
                ast.unparse(ast.Module(body=filtered, type_ignores=[])) + "\n", encoding="utf-8"
            )
            metadata["adaptations"].append(
                {
                    "stage": "q1",
                    "reason": "Windows converter unavailable",
                    "removed_statements": removed,
                    "adapted_sha256": sha(adapted),
                }
            )
            ready = run_stage("q1_prepare_onnx_only", [sys.executable, str(adapted)])
        if not ready:
            metadata["status"] = "prepare_failed"
            checkpoint()
            return 1
    if args.stage in ("all", "onnx"):
        run_stage("q2_quant_onnx", [sys.executable, str(SOURCE / "q2_quant_onnx.py")])
    if args.stage in ("all", "tflite") and args.mode == "full":
        run_stage("q3_quant_tflite", [sys.executable, str(SOURCE / "q3_quant_tflite.py")])
    if args.stage in ("all", "evaluate"):
        metadata["results"] = []
        metadata["evaluation_passes_per_variant"] = 1
        metadata["harness_sha256"] = sha(__file__)
        os.environ.update(
            {
                k: env[k]
                for k in [
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                ]
            }
        )
        os.chdir(artifacts)
        # Load imports, assignments and function definitions; don't execute q4's
        # all-or-nothing loop. Original functions and measurements stay intact.
        tree = ast.parse((SOURCE / "q4_evaluate.py").read_text(encoding="utf-8-sig"))
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef))
        ]
        namespace = {"__name__": "peer_evaluation_definitions"}
        exec(compile(tree, str(SOURCE / "q4_evaluate.py"), "exec"), namespace)
        import numpy as np

        class CaptureNumpy:
            samples = None
            outputs = None

            def __getattr__(self, name):
                return getattr(np, name)

            def median(self, values, *a, **kw):
                self.samples = [float(x) for x in values]
                return np.median(values, *a, **kw)

            def concatenate(self, *a, **kw):
                self.outputs = np.concatenate(*a, **kw)
                return self.outputs

        capture = CaptureNumpy()
        namespace["np"] = capture
        metadata["adaptations"].append(
            {
                "stage": "q4",
                "change": "execute unchanged runner/evaluate/benchmark functions separately per variant; retain evaluate concatenate output and collect np.median argument after timer",
                "reason": "retain successful results and raw latency if an operator fails in another variant",
            }
        )
        metadata["reference_accuracy_pct"] = float(
            np.mean(namespace["ref"].argmax(1) == namespace["val_y"]) * 100
        )
        metadata["tensor_hashes"] = {
            name: sha(artifacts / name)
            for name in ["calib_x.npy", "val_x.npy", "val_y.npy", "ref_output.npy"]
        }
        for name, filename, factory in namespace["MODELS"]:
            record = {"name": name, "file": filename}
            try:
                if not Path(filename).exists():
                    record.update(
                        status="not_generated",
                        reason="Converter stage omitted or failed; not benchmarked",
                    )
                else:
                    run = factory(filename)
                    acc, agreement, snr = namespace["evaluate"](run)
                    median = namespace["benchmark"](run)
                    outputs = capture.outputs
                    output_file = results / (name.lower().replace(" ", "_") + "_outputs.npy")
                    np.save(output_file, outputs)
                    ref = namespace["ref"].astype(np.float64)
                    error = ref - outputs.astype(np.float64)
                    record.update(
                        status="ok",
                        accuracy_pct=float(acc),
                        agreement_pct=float(agreement),
                        snr_db_source_float32=float(snr),
                        snr_db_float64=float(10 * np.log10(np.sum(ref**2) / np.sum(error**2))),
                        mae=float(np.mean(np.abs(error))),
                        max_abs_error=float(np.max(np.abs(error))),
                        median_ms=float(median),
                        p95_ms=float(np.percentile(capture.samples, 95)),
                        samples_ms=capture.samples,
                        model_sha256=sha(filename),
                        size_mb=Path(filename).stat().st_size / 1e6,
                        output_file=output_file.name,
                        output_sha256=sha(output_file),
                    )
                    record["top1_tie_samples"] = int(
                        np.sum(np.sum(outputs == outputs.max(axis=1, keepdims=True), axis=1) > 1)
                    )
                    record["reference_logits_range"] = [float(ref.min()), float(ref.max())]
                    record["output_logits_range"] = [float(outputs.min()), float(outputs.max())]
                    if factory.__name__ == "make_tflite_runner":
                        closure = {
                            name: cell.cell_contents
                            for name, cell in zip(run.__code__.co_freevars, run.__closure__)
                        }
                        record["io"] = {}
                        for key in ["inp", "out"]:
                            detail = closure[key]
                            scale, zp = detail["quantization"]
                            record["io"][key] = {
                                "dtype": np.dtype(detail["dtype"]).name,
                                "shape": [int(x) for x in detail["shape"]],
                                "scale": float(scale),
                                "zero_point": int(zp),
                            }
                        out = record["io"]["out"]
                        if out["dtype"] == "int8":
                            lo = out["scale"] * (-128 - out["zero_point"])
                            hi = out["scale"] * (127 - out["zero_point"])
                            record["output_representable_range"] = [lo, hi]
                            record["reference_values_outside_output_range"] = int(
                                np.sum((ref < lo) | (ref > hi))
                            )
            except Exception as e:
                record.update(status="failed", error=str(e), exception=traceback.format_exc())
            metadata["results"].append(record)
            print(
                name
                + ": "
                + record["status"]
                + (
                    f" {record['median_ms']:.3f} ms / accuracy {record['accuracy_pct']:.2f}%"
                    if record["status"] == "ok"
                    else ""
                ),
                flush=True,
            )
            checkpoint()
        metadata["status"] = (
            "full_pipeline_measured"
            if all(r["status"] == "ok" for r in metadata["results"])
            else "partial_pipeline_measured"
        )
    final_hashes = {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()}
    if source_hashes != final_hashes:
        raise AssertionError("Peer source changed during reproduction")
    metadata["source_unchanged"] = True
    checkpoint()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "onnx-only"], default="full")
    p.add_argument(
        "--stage", choices=["all", "prepare", "onnx", "tflite", "evaluate"], default="all"
    )
    p.add_argument("--run-id", default="run-03-peer")
    args = p.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        p.error("run-id must contain only letters, numbers, - and _")
    sys.exit(main(args))
