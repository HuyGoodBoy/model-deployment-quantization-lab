"""Recompute GPU evidence and summarize matched ORT CPU/CUDA comparisons."""

import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np
from common import RESULTS, ROOT, checked, info, read, sha, write
from gpu_final import GPU_RESULTS, model_paths
from metrics import line_metrics, numerical

sys.path.insert(0, str(ROOT / "quantization"))
from qlab.gpu_runtime import package_versions, summarize_profile


def compare(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            compare(value, actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            compare(left, right)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(expected, actual, rtol=1e-10, atol=1e-10)
    else:
        assert expected == actual


def main():
    rows, comparisons, evidence = {}, {}, {}
    environment = read(GPU_RESULTS / "environment.json")
    assert environment["ort"] == "1.20.2"
    for task in ["mlsd", "text"]:
        conversion = read(GPU_RESULTS / f"{task}_fp16_conversion.json")
        assert checked(conversion["source"]) == model_paths(task)["fp32"]
        assert checked(conversion["model"]) == model_paths(task)["fp16"]
        assert conversion["initializer_types"]["FLOAT16"] > 0
        manifest = read(RESULTS / f"{task}_manifest.json")
        baseline = read(RESULTS / f"{task}_final.json")
        baseline_name = "original_fp32" if task == "mlsd" else "teacher_fp32"
        reference = np.load(checked(baseline["models"][baseline_name]["output"]))
        truth = (
            read(checked(manifest["ground_truth"]))
            if task == "mlsd"
            else np.load(checked(manifest["inputs"]))["labels"]
        )
        outputs = {}
        for variant, model in model_paths(task).items():
            for device in ["cpu", "cuda"]:
                key = f"{task}_{variant}_{device}"
                path = GPU_RESULTS / f"{key}.json"
                record = read(path)
                evidence[key] = info(path)
                assert record["ort"] == environment["ort"]
                assert checked(record["model"]) == model
                checked(record["source_code"])
                checked(record["manifest"])
                row = {"status": record["status"], "model": record["model"]}
                rows[key] = row
                if record["status"] == "failed":
                    assert record["error"] and "benchmark" not in record
                    row["error"] = record["error"]
                    continue
                output = np.load(checked(record["output"]))
                assert output.shape == reference.shape and np.isfinite(output).all()
                outputs[(variant, device)] = output
                compare(record["parity_vs_ort130_cpu_fp32"], numerical(reference, output))
                if task == "mlsd":
                    quality = line_metrics(output, truth)
                    score = quality["sAP10_subset_percent"]
                else:
                    correct = int(np.count_nonzero(output.argmax(-1) == truth))
                    quality = {
                        "correct": correct,
                        "samples": len(truth),
                        "accuracy_percent": correct / len(truth) * 100,
                        "label_agreement_vs_ort130_cpu_fp32": float(
                            np.mean(output.argmax(-1) == reference.argmax(-1))
                        ),
                    }
                    score = quality["accuracy_percent"]
                compare(record["quality"], quality)
                timing = record["benchmark"]
                samples = np.asarray(timing["samples_ms"])
                assert timing["runs"] == len(samples) == 80 and (samples > 0).all()
                compare(timing["median_ms"], float(np.median(samples)))
                compare(timing["p95_ms"], float(np.percentile(samples, 95)))
                assert (
                    timing["input_indices"]
                    == np.linspace(0, len(truth) - 1, 16, dtype=int).tolist()
                )
                events = read(checked(record["profile"]["file"]))
                profile = summarize_profile(events)
                typed_nodes = {}
                for event in events:
                    args = event.get("args", {})
                    shapes = args.get("input_type_shape", [])
                    if (
                        event.get("cat") == "Node"
                        and event.get("name", "").endswith("_kernel_time")
                        and args.get("provider")
                        and shapes
                    ):
                        typed_nodes[(args["provider"], event["name"])] = (
                            args["provider"],
                            args["op_name"],
                            ",".join(shapes[0]),
                        )
                arithmetic = Counter("/".join(value) for value in typed_nodes.values())
                compare({k: v for k, v in record["profile"].items() if k != "file"}, profile)
                assert (profile["cuda_compute_nodes"] > 0) == (
                    device == "cuda" and record["status"] == "ok"
                )
                if device == "cuda":
                    assert record["provider_options"]["CUDAExecutionProvider"]["use_tf32"] == "0"
                row.update(
                    quality_percent=score,
                    median_ms=timing["median_ms"],
                    p95_ms=timing["p95_ms"],
                    cuda_compute_nodes=profile["cuda_compute_nodes"],
                    cpu_nodes=len(profile["cpu_nodes"]),
                    operator_counts_by_provider=profile["operator_counts_by_provider"],
                    scheduled_operator_first_input_type_counts=dict(arithmetic),
                )
        for variant in model_paths(task):
            if (variant, "cpu") not in outputs or (variant, "cuda") not in outputs:
                continue
            cpu, cuda = outputs[(variant, "cpu")], outputs[(variant, "cuda")]
            left, right = rows[f"{task}_{variant}_cpu"], rows[f"{task}_{variant}_cuda"]
            comparisons[f"{task}_{variant}"] = {
                "cpu_to_cuda_parity": numerical(cpu, cuda),
                "quality_delta_cuda_minus_cpu_pp": right["quality_percent"]
                - left["quality_percent"],
                "cpu_over_cuda_median_ratio": left["median_ms"] / right["median_ms"],
            }
            if task == "text":
                comparisons[f"{task}_{variant}"]["label_agreement"] = float(
                    np.mean(cpu.argmax(-1) == cuda.argmax(-1))
                )
    assert (
        rows["mlsd_qat_41_integer_cuda"]["operator_counts_by_provider"]["CPUExecutionProvider"][
            "ConvInteger"
        ]
        == 41
    )
    assert (
        rows["text_qat_static_cuda"]["operator_counts_by_provider"]["CPUExecutionProvider"][
            "MatMulInteger"
        ]
        == 38
    )
    assert (
        rows["mlsd_qat_qdq_cuda"]["scheduled_operator_first_input_type_counts"][
            "CUDAExecutionProvider/Conv/float"
        ]
        == 41
    )
    write(
        GPU_RESULTS / "verification.json",
        {
            "status": "passed",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "scope": "Evidence consistency for 14 attempted cases; failed cases remain failures; existing QAT release gates unchanged",
            "environment": info(GPU_RESULTS / "environment.json"),
            "verification_package_versions": package_versions(),
            "rows": rows,
            "matched_runtime_comparisons": comparisons,
            "checked_results": evidence,
            "source_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                for p in sorted((RESULTS.parent / "src").glob("*.py"))
            },
        },
    )
    print("PASS GPU metrics, timings, hashes and actual provider partitioning", flush=True)
    for key, row in rows.items():
        print(key, row["status"], row.get("median_ms"), row.get("quality_percent"), flush=True)


if __name__ == "__main__":
    main()
