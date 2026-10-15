"""Controlled M-LSD diagnostics, train-only foreground calibration and loss ablations."""

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, PerChannelMinMaxObserver, disable_observer
from torch.nn import functional as F
from train_continue import PILOT, TASK, pilot, relative, training_loss

sys.path.insert(0, str(PILOT / "src"))
from models import QatOperation, prepare_qat

ART = TASK / "artifacts/research"
RESULTS = TASK / "results/research"
SOURCE = TASK / "artifacts/mlsd_step0032.pt"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(path=SOURCE):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    teacher = pilot.mlsd_teacher()
    model, config = prepare_qat(teacher, "mlsd")
    policy = checkpoint.get("research_policy", {})
    if policy:
        configure_range(model, policy)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.apply(disable_observer)
    model.eval()
    return model, teacher, checkpoint, config


def configure_range(model, policy):
    if not policy:
        return
    if policy["per_channel"]:
        model.regression_quant = FakeQuantize(
            observer=PerChannelMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_affine,
            ch_axis=1,
            quant_min=-128,
            quant_max=127,
        )
    quantizer = model.regression_quant
    observer = quantizer.activation_post_process
    observer.min_val.resize_(len(policy["minimum"]) if policy["per_channel"] else ())
    observer.max_val.resize_(len(policy["maximum"]) if policy["per_channel"] else ())
    observer.min_val.copy_(torch.tensor(policy["minimum"]).reshape(observer.min_val.shape))
    observer.max_val.copy_(torch.tensor(policy["maximum"]).reshape(observer.max_val.shape))
    scale, zero = observer.calculate_qparams()
    quantizer.scale.resize_(scale.shape).copy_(scale)
    quantizer.zero_point.resize_(zero.shape).copy_(zero)
    quantizer.disable_observer()
    if "center_minimum" in policy:
        center = model.center_quant
        observer = center.activation_post_process
        observer.min_val.copy_(torch.tensor(policy["center_minimum"]))
        observer.max_val.copy_(torch.tensor(policy["center_maximum"]))
        scale, zero = observer.calculate_qparams()
        center.scale.copy_(scale)
        center.zero_point.copy_(zero)
        center.disable_observer()


def save_outputs(name, output, data):
    path = ART / (name + "_output.npy")
    np.save(path, output, allow_pickle=False)
    return {
        "name": name,
        "metrics": pilot.evaluate(output, data, "mlsd"),
        "output": {"path": relative(path), "sha256": pilot.sha(path)},
    }


def stats(values):
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "percentiles": dict(
            zip(
                ["0.1", "1", "50", "99", "99.9"],
                np.percentile(values, [0.1, 1, 50, 99, 99.9]).tolist(),
            )
        ),
        "values": int(values.size),
    }


def balanced_loss(predicted, target):
    probability = target[:, :1].sigmoid()
    center = (
        F.binary_cross_entropy_with_logits(predicted[:, :1], probability, reduction="none")
        * (1 + 4 * probability)
    ).mean()
    mask = (probability > 0.1).expand_as(target[:, 1:5])
    raw = F.smooth_l1_loss(predicted[:, 1:5], target[:, 1:5], beta=1, reduction="none")
    regression = raw[mask].mean() / 128 if mask.any() else raw.mean() * 0
    return center + regression, {
        "soft_center_bce": float(center.detach()),
        "positive_displacement_smooth_l1": float(regression.detach()),
        "positive_regression_values": int(mask.sum()),
    }


def scan():
    data = pilot.load_datasets("mlsd")
    model, teacher, _saved, _config = load_model()
    teacher_train = pilot.forward_all(teacher, data["train"], "mlsd")
    raw_train = pilot.forward_all(model.model, data["train"], "mlsd")
    probability = 1 / (1 + np.exp(-np.clip(teacher_train[..., 0], -80, 80)))
    foreground = probability > 0.1
    fg = raw_train[..., 1:5][foreground]
    margin = 0.02
    minimum, maximum = fg.min(axis=0), fg.max(axis=0)
    padding = (maximum - minimum) * margin
    channel_policy = {
        "per_channel": True,
        "minimum": (minimum - padding).tolist(),
        "maximum": (maximum + padding).tolist(),
        "threshold": 0.1,
        "margin_fraction": margin,
        "fit_split": "32 Wireframe train images only",
        "rule": "QAT pre-output displacement extrema at teacher confidence > 0.1, 2% margin/channel",
    }
    tensor_policy = {
        **channel_policy,
        "per_channel": False,
        "minimum": [float((minimum - padding).min())],
        "maximum": [float((maximum + padding).max())],
    }
    policies = {"foreground_tensor": tensor_policy, "foreground_channel": channel_policy}
    pilot.write_json(RESULTS / "range_policies.json", policies)
    rows = []
    variants = [
        "baseline",
        "regression_output_float",
        "center_output_float",
        "both_outputs_float",
        "weight_only",
        "activation_only",
        "foreground_tensor",
        "foreground_channel",
    ]
    for name in variants:
        candidate = copy.deepcopy(model)
        if name in ["regression_output_float", "both_outputs_float"]:
            candidate.regression_quant.disable_fake_quant()
        if name in ["center_output_float", "both_outputs_float"]:
            candidate.center_quant.disable_fake_quant()
        for module in candidate.modules():
            if isinstance(module, QatOperation):
                if name == "weight_only":
                    module.activation_fake_quant.disable_fake_quant()
                if name == "activation_only":
                    module.weight_fake_quant.disable_fake_quant()
        if name == "weight_only":
            candidate.center_quant.disable_fake_quant()
            candidate.regression_quant.disable_fake_quant()
        if name in policies:
            configure_range(candidate, policies[name])
        output = pilot.forward_all(candidate, data["evaluation"], "mlsd")
        entry = save_outputs("scan_" + name, output, data)
        entry["diagnostic_only"] = name.endswith("float") or name in [
            "weight_only",
            "activation_only",
        ]
        entry["regression_scale"] = candidate.regression_quant.scale.tolist()
        rows.append(entry)
        print(name, entry["metrics"]["line_quality"]["sAP10_subset_percent"], flush=True)

    gradient_rows = []
    model.eval()
    last_weight = [m.layer.weight for m in model.modules() if isinstance(m, QatOperation)][-1]
    for index in [0, 1, 2, 3]:
        item = (data["train"][0][index : index + 1],)
        with torch.no_grad():
            target = teacher(*item)
        predicted = model(*item)
        probability = target[:, :1].sigmoid()
        center = (
            F.binary_cross_entropy_with_logits(predicted[:, :1], probability, reduction="none")
            * (1 + 4 * probability)
        ).mean()
        mask = (probability > 0.1).expand_as(target[:, 1:5])
        old_reg = F.smooth_l1_loss(predicted[:, 1:5] / 128, target[:, 1:5] / 128, reduction="none")[
            mask
        ].mean()
        new_reg = (
            F.smooth_l1_loss(predicted[:, 1:5], target[:, 1:5], reduction="none")[mask].mean() / 128
        )
        norms = {}
        for label, loss in [
            ("center", center),
            ("old_displacement", old_reg),
            ("new_displacement", new_reg),
        ]:
            gradient = torch.autograd.grad(loss, last_weight, retain_graph=True)[0]
            norms[label] = float(gradient.norm())
        gradient_rows.append(
            {
                "train_index": index,
                "loss": {
                    "center": float(center.detach()),
                    "old_displacement": float(old_reg.detach()),
                    "new_displacement": float(new_reg.detach()),
                },
                "last_conv_weight_gradient_l2": norms,
            }
        )
    record = {
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": {"path": relative(SOURCE), "sha256": pilot.sha(SOURCE)},
        "development_images": 16,
        "range_fit_images": 32,
        "protocol": "All scan variants use identical step32 weights, same decoder/16-image subset; no training or score-threshold changes",
        "teacher_foreground_fraction_train": float(foreground.mean()),
        "all_displacement_train": stats(raw_train[..., 1:5]),
        "foreground_displacement_train": [stats(fg[:, c]) for c in range(4)],
        "original_regression_scale": model.regression_quant.scale.tolist(),
        "policies": policies,
        "ablations": rows,
        "gradient_diagnostics": gradient_rows,
        "note": "Loss magnitudes alone do not establish gradient influence; gradient norms are for the final Conv weight on four train images",
        "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
    }
    pilot.write_json(RESULTS / "diagnostics.json", record)


def train(args):
    data = pilot.load_datasets("mlsd")
    model, teacher, saved, config = load_model()
    policy = (
        read(RESULTS / "selected_policy.json")["policy"]
        if args.variant in ["range_only", "range_and_loss"]
        else {}
    )
    configure_range(model, policy)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng_state"])
    before = pilot.forward_all(model, data["evaluation"], "mlsd")
    start = saved["step"]
    del saved
    order = np.random.default_rng(20261014).permutation(32)
    histories = []
    entries = []
    for target_step in [64, 128]:
        model.train()
        history = []
        for step in range(start, target_step):
            index = int(order[step % 32])
            item = (data["train"][0][index : index + 1],)
            with torch.no_grad():
                target = teacher(*item)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(*item)
            if args.variant == "range_only":
                loss, components = training_loss(predicted, target, data, [index], "mlsd")
            else:
                loss, components = balanced_loss(predicted, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            if any(
                not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise RuntimeError("Non-finite gradient")
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "train_index": index,
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                    **components,
                }
            )
            if (step + 1) % 16 == 0:
                print(args.variant, step + 1, float(loss.detach()), flush=True)
        name = f"{args.variant}_step{target_step:04d}"
        output = pilot.forward_all(model, data["evaluation"], "mlsd")
        entry = save_outputs(name, output, data)
        checkpoint = ART / (name + ".pt")
        torch.save(
            {
                "task": "mlsd",
                "step": target_step,
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "research_policy": policy,
            },
            checkpoint,
        )
        entry["checkpoint"] = {"path": relative(checkpoint), "sha256": pilot.sha(checkpoint)}
        entry["start_step"] = start
        entry["history"] = history
        histories.extend(history)
        entries.append(entry)
        print("SAVED", name, entry["metrics"]["line_quality"]["sAP10_subset_percent"], flush=True)
        start = target_step
    pilot.write_json(
        RESULTS / (args.variant + ".json"),
        {
            "variant": args.variant,
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "source_checkpoint": {"path": relative(SOURCE), "sha256": pilot.sha(SOURCE)},
            "before": save_outputs(args.variant + "_before", before, data),
            "range_policy": policy,
            "loss_changed": args.variant != "range_only",
            "loss": "WBCE center + mean positive SmoothL1(raw displacement, beta=1) /128"
            if args.variant != "range_only"
            else "Unchanged day14 loss; normalize displacement before SmoothL1(beta=1)",
            "parameters_held_fixed": "Same step32 model/AdamW state, seed/RNG, sample order, lr, 32 train images, teacher, decoder and evaluation",
            "train_examples": 32,
            "new_steps": len(histories),
            "evaluation_examples": 16,
            "milestones": entries,
            "source_sha256": {relative(Path(__file__)): pilot.sha(__file__)},
        },
    )


def center_scan():
    data = pilot.load_datasets("mlsd")
    model, _teacher, _saved, _config = load_model()
    raw_train = pilot.forward_all(model.model, data["train"], "mlsd")
    # Below the fixed score>=.01 decoder gate; clipped background cannot become a candidate.
    center_minimum = float(np.log(0.005 / (1 - 0.005)))
    center_maximum = float(raw_train[..., 0].max())
    center_maximum += (center_maximum - center_minimum) * 0.02
    policies = read(RESULTS / "range_policies.json")
    base_reg = model.regression_quant.activation_post_process
    policies["center_only"] = {
        "per_channel": False,
        "minimum": [float(base_reg.min_val)],
        "maximum": [float(base_reg.max_val)],
        "fit_split": "Original day14 regression range",
    }
    rows = []
    for name, reg_policy in policies.items():
        policy = {
            **reg_policy,
            "center_minimum": center_minimum,
            "center_maximum": center_maximum,
            "center_rule": "Clip below sigmoid .005, below unchanged decoder score .01; max from train + 2% margin",
        }
        candidate = copy.deepcopy(model)
        configure_range(candidate, policy)
        output = pilot.forward_all(candidate, data["evaluation"], "mlsd")
        entry = save_outputs("center_scan_" + name, output, data)
        entry["policy"] = policy
        entry["center_scale"] = candidate.center_quant.scale.tolist()
        entry["regression_scale"] = candidate.regression_quant.scale.tolist()
        rows.append(entry)
        print(name, entry["metrics"]["line_quality"]["sAP10_subset_percent"], flush=True)
    best = max(rows, key=lambda r: r["metrics"]["line_quality"]["sAP10_subset_percent"])
    pilot.write_json(
        RESULTS / "center_calibration.json",
        {
            "checkpoint": {"path": relative(SOURCE), "sha256": pilot.sha(SOURCE)},
            "protocol": "Same step32 weights; center/range clipping fit on 32 train images, fixed decoder, same 16 development images",
            "ablations": rows,
        },
    )
    pilot.write_json(
        RESULTS / "selected_policy.json",
        {
            "policy": best["policy"],
            "development_variant": best["name"],
            "development_sAP10": best["metrics"]["line_quality"]["sAP10_subset_percent"],
            "criterion": "Highest development sAP10 among predeclared three center/range policies, before any holdout inference",
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "holdout_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["scan", "center-scan", "train"], required=True)
    parser.add_argument("--variant", choices=["range_only", "loss_only", "range_and_loss"])
    args = parser.parse_args()
    torch.manual_seed(20261014)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    ART.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "scan":
        scan()
    elif args.stage == "center-scan":
        center_scan()
    else:
        if not args.variant:
            parser.error("--variant is required for training")
        train(args)
