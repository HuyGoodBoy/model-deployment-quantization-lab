"""Two selective-precision exports; all retained QAT scales are frozen."""

import copy
import gc

import onnx
import torch
from support import (
    ART,
    DAY16,
    RESULTS,
    baselines,
    checked,
    infer,
    info,
    now,
    pilot,
    profile_model,
    read,
    save_output,
    write,
)
from torch.ao.quantization import disable_fake_quant

# isort: split
from finalize_mlsd import QdqConv
from models import QatOperation
from research_mlsd import load_model


def replace(model, float_names, qdq=True):
    changed = []

    def walk(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full = prefix + name
            if isinstance(child, QatOperation):
                if full in float_names:
                    parent._modules[name] = copy.deepcopy(child.layer)
                    changed.append(full)
                elif qdq:
                    parent._modules[name] = QdqConv(child)
            else:
                walk(child, full + ".")

    walk(model)
    assert set(changed) == set(float_names)
    return changed


def export(model, path, sample):
    if path.exists():
        raise FileExistsError(path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample,),
            str(path),
            opset_version=17,
            dynamo=False,
            input_names=["input"],
            output_names=["output"],
        )
    onnx.checker.check_model(str(path), full_check=True)


def main():
    if (RESULTS / "exports.json").exists():
        raise FileExistsError("Exports already recorded")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    diagnosis = read(RESULTS / "diagnosis.json")
    selected = read(DAY16 / "results/mlsd_selection.json")
    model, _teacher, _checkpoint, _ = load_model(checked(selected["checkpoint"]))
    data = pilot.load_datasets("mlsd")
    registry = baselines()
    all_fake = pilot.forward_all(model, data["evaluation"], "mlsd")
    reference_path = save_output("qat_fake_reference", all_fake)
    outputs = {}
    for name, float_names in diagnosis["candidate_float_layers"].items():
        policy = copy.deepcopy(model)
        replace(policy, float_names, qdq=False)
        policy_ref = pilot.forward_all(policy, data["evaluation"], "mlsd")
        lowering = copy.deepcopy(model)
        replace(lowering, float_names)
        lowering_ref = pilot.forward_all(lowering, data["evaluation"], "mlsd")
        lowering_parity = pilot.numerical(policy_ref, lowering_ref)
        if lowering_parity["relative_l2"] > 1e-5:
            raise ValueError("QDQ lowering changed selective policy reference")
        # Surviving FakeQuantize modules must have identical qparams to checkpoint.
        source_modules = dict(model.named_modules())
        retained = {}
        for key, module in policy.named_modules():
            if hasattr(module, "scale") and hasattr(module, "zero_point"):
                before = source_modules[key]
                assert torch.equal(module.scale, before.scale)
                assert torch.equal(module.zero_point, before.zero_point)
                retained[key] = {
                    "scale": module.scale.tolist(),
                    "zero_point": module.zero_point.tolist(),
                }
        path = ART / f"{name}.onnx"
        export(lowering, path, data["evaluation"][0][:1])
        registry[name] = {
            "model": info(path),
            "optimization": "all",
            "float_layers": float_names,
            "retained_quantizers": retained,
        }
        actual = infer(path, data["evaluation"][0].numpy())
        parity = pilot.numerical(policy_ref, actual)
        ref_ap = pilot.line_metrics(policy_ref, data["ground_truth"])["sAP10_subset_percent"]
        actual_ap = pilot.line_metrics(actual, data["ground_truth"])["sAP10_subset_percent"]
        outputs[name] = {
            "policy_reference": save_output(name + "_policy_reference", policy_ref),
            "development_output": save_output(name + "_development", actual),
            "lowering_parity": lowering_parity,
            "export_parity": parity,
            "export_sAP10_delta_pp": actual_ap - ref_ap,
            "export_gate_passed": parity["relative_l2"] <= 0.02 and abs(actual_ap - ref_ap) <= 0.5,
            "policy_vs_original_fake": pilot.numerical(all_fake, policy_ref),
            "source_weight_unchanged": True,
            "profile": profile_model(name, registry[name], data["evaluation"][0].numpy()),
        }
        print(
            "EXPORTED",
            name,
            "float",
            len(float_names),
            "AP",
            actual_ap,
            "export L2",
            parity["relative_l2"],
            flush=True,
        )
        del policy, lowering, policy_ref, lowering_ref, actual
        gc.collect()
    floating = copy.deepcopy(model)
    floating.apply(disable_fake_quant)
    path = ART / "qat_weights_fp32.onnx"
    export(floating, path, data["evaluation"][0][:1])
    registry["qat_weights_fp32"] = {"model": info(path), "optimization": "all"}
    ref = pilot.forward_all(floating, data["evaluation"], "mlsd")
    actual = infer(path, data["evaluation"][0].numpy())
    fp32_parity = pilot.numerical(ref, actual)
    outputs["qat_weights_fp32"] = {
        "policy_reference": save_output("qat_weights_fp32_reference", ref),
        "development_output": save_output("qat_weights_fp32_development", actual),
        "export_parity": fp32_parity,
    }
    for item in registry.values():
        checked(item["model"])
    write(
        RESULTS / "exports.json",
        {
            "finished_at_utc": now(),
            "models": registry,
            "candidates": outputs,
            "fake_quant_reference": reference_path,
            "checkpoint": selected["checkpoint"],
            "diagnosis": info(RESULTS / "diagnosis.json"),
            "scope": "A/B deliberately remove input and weight quantization inside chosen blocks; remaining and output scales unchanged; no training/recalibration",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
