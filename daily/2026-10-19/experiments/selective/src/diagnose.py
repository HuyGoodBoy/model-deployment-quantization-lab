"""Warm profiles and same-input local sensitivity, using development data only."""

import gc
import platform
from collections import Counter, defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from support import (
    ART,
    DAY15,
    DAY16,
    RESULTS,
    ROOT,
    baselines,
    block_name,
    checked,
    info,
    now,
    pilot,
    profile_model,
    read,
    write,
)

# isort: split
from models import QatOperation
from research_mlsd import load_model


def graph_rows(original_path, optimized_path, profile):
    original, optimized = onnx.load(original_path), onnx.load(optimized_path)
    producers = {out: n for n in original.graph.node for out in n.output}
    consumers = defaultdict(list)
    for n in original.graph.node:
        for arg in n.input:
            consumers[arg].append(n)
    rows = []
    for node in original.graph.node:
        if node.op_type != "Conv":
            continue
        weight_dq = producers[node.input[1]]
        layer = weight_dq.input[0].removesuffix(".weight")
        runtime_nodes = [n for n in profile["nodes"] if n["name"].startswith(node.name)]
        if len(runtime_nodes) != 1:
            raise ValueError((node.name, runtime_nodes))
        after, frontier = [], list(node.output)
        for _ in range(4):
            next_nodes = {n.name: n for value in frontier for n in consumers[value]}
            after.append([n.op_type for n in next_nodes.values()])
            frontier = [out for n in next_nodes.values() for out in n.output]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        rows.append(
            {
                "layer": layer,
                "block": block_name(layer),
                "node": node.name,
                "group": attrs.get("group", 1),
                "consumers_first_4_steps": after,
                "runtime": runtime_nodes[0],
            }
        )
    return {
        "convolutions": rows,
        "original_operators": dict(Counter(n.op_type for n in original.graph.node)),
        "optimized_operators": dict(Counter(n.op_type for n in optimized.graph.node)),
    }


def local_sensitivity(model, inputs):
    sums = defaultdict(lambda: {"squared_error": 0.0, "squared_reference": 0.0, "values": 0})
    handles = []

    def hook(name, module, args, output):
        layer = module.layer
        ref = torch.nn.functional.conv2d(
            args[0],
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )
        item = sums[name]
        item["squared_error"] += float((output - ref).square().sum(dtype=torch.float64))
        item["squared_reference"] += float(ref.square().sum(dtype=torch.float64))
        item["values"] += ref.numel()

    for name, module in model.named_modules():
        if isinstance(module, QatOperation):
            handles.append(
                module.register_forward_hook(
                    lambda mod, args, out, name=name: hook(name, mod, args, out)
                )
            )
    with torch.no_grad():
        for index in [0, 5, 10, 15]:
            model(inputs[index : index + 1])
    for handle in handles:
        handle.remove()
    return [
        {
            "layer": name,
            "block": block_name(name),
            **item,
            "relative_l2": float(
                np.sqrt(item["squared_error"] / max(item["squared_reference"], 1e-30))
            ),
        }
        for name, item in sums.items()
    ]


def main():
    if (RESULTS / "diagnosis.json").exists():
        raise FileExistsError("Diagnosis already recorded")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    ART.mkdir(parents=True, exist_ok=True)
    data = pilot.load_datasets("mlsd")
    selection = read(DAY16 / "results/mlsd_selection.json")
    manifest = read(DAY16 / "results/mlsd_manifest.json")
    frozen_files = [
        ROOT / "daily/2026-10-19/PROTOCOL.md",
        DAY16 / "results/mlsd_selection.json",
        DAY16 / "results/mlsd_manifest.json",
        DAY16 / "results/mlsd_final.json",
        DAY16 / "results/mlsd_benchmark.json",
        DAY15 / "results/research/selection.json",
    ]
    frozen_files.extend(checked(item["model"]) for item in baselines().values())
    frozen_files.extend(
        [
            checked(selection["checkpoint"]),
            checked(manifest["inputs"]),
            checked(manifest["ground_truth"]),
        ]
    )
    environment = {
        "recorded_at_utc": now(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "intra_threads": 4,
        "inter_threads": 1,
        "spinning": False,
        "batch": 1,
        "input_shape": [1, 4, 512, 512],
    }
    write(RESULTS / "environment.json", environment)
    write(
        RESULTS / "frozen_sources.json",
        {
            "locked_at_utc": now(),
            "files": [info(p) for p in dict.fromkeys(frozen_files)],
            "scope": "Only checking hashes of existing regression artifacts; no new regression inference before selection",
        },
    )
    profiles = {}
    for name in ["qdq_all", "custom_41_integer", "qdq_optimizer_off"]:
        profiles[name] = profile_model(name, baselines()[name], data["evaluation"][0].numpy())
        print("WARM PROFILE", name, profiles[name]["unique_operators"], flush=True)
    graph = graph_rows(
        checked(baselines()["qdq_all"]["model"]),
        checked(profiles["qdq_all"]["optimized_model"]),
        profiles["qdq_all"],
    )
    model, teacher, checkpoint, _ = load_model(checked(selection["checkpoint"]))
    sensitivity = local_sensitivity(model, data["evaluation"][0])
    lookup = {r["layer"]: r for r in graph["convolutions"]}
    blocks = {}
    for row in sensitivity:
        assert row["layer"] in lookup, row["layer"]
        group = blocks.setdefault(
            row["block"],
            {"layers": [], "conv_profile_ms": 0, "squared_error": 0, "squared_reference": 0},
        )
        group["layers"].append(row["layer"])
        group["conv_profile_ms"] += lookup[row["layer"]]["runtime"]["mean_ms"]
        group["squared_error"] += row["squared_error"]
        group["squared_reference"] += row["squared_reference"]
    for item in blocks.values():
        item["local_relative_l2"] = float(
            np.sqrt(item["squared_error"] / max(item["squared_reference"], 1e-30))
        )
        item["selection_score"] = item["conv_profile_ms"] * item["local_relative_l2"]
    anchor = max(blocks, key=lambda key: blocks[key]["selection_score"])
    family = [key for key in blocks if ("backbone" in key) == ("backbone" in anchor)]
    index = family.index(anchor)
    expanded = family[max(0, index - 1) : index + 2]
    float_a = blocks[anchor]["layers"]
    float_b = [name for group in expanded for name in blocks[group]["layers"]]
    write(
        RESULTS / "diagnosis.json",
        {
            "finished_at_utc": now(),
            "profiles": profiles,
            "graph": graph,
            "local_sensitivity": sensitivity,
            "sensitivity_indices": [0, 5, 10, 15],
            "blocks": blocks,
            "anchor": anchor,
            "expanded_blocks": expanded,
            "candidate_float_layers": {"selective_a": float_a, "selective_b": float_b},
            "selection_rule": "Max warm Conv-time times same-input local quantization relative L2; deterministic adjacent-block expansion; development only",
            "limitations": "Local error is not global causal attribution; topology patterns are observations, not proof of unsupported fusion; profiler overhead is excluded from benchmark",
            "source_code": info(__file__),
        },
    )
    print("ANCHOR", anchor, "EXPANDED", expanded, flush=True)
    del model, teacher, checkpoint
    gc.collect()


if __name__ == "__main__":
    main()
