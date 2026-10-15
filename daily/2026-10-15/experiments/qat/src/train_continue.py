"""Resume the day-14 QAT protocol and preserve a checkpoint at every requested step."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import disable_observer
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
PILOT = ROOT / "daily/2026-10-14/experiments/qat"
sys.path.insert(0, str(PILOT / "src"))
import train as pilot
from models import load_text, prepare_qat


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def training_loss(predicted, target, data, indices, task):
    if task == "mlsd":
        probability = target[:, :1].sigmoid()
        classification = (
            F.binary_cross_entropy_with_logits(predicted[:, :1], probability, reduction="none")
            * (1 + 4 * probability)
        ).mean()
        mask = (probability > 0.1).expand_as(target[:, 1:5])
        regression = F.smooth_l1_loss(
            predicted[:, 1:5] / 128, target[:, 1:5] / 128, reduction="none"
        )
        regression = regression[mask].mean() if mask.any() else regression.mean() * 0
        return classification + regression, {
            "soft_center_bce": float(classification.detach()),
            "positive_displacement_smooth_l1": float(regression.detach()),
            "positive_regression_values": int(mask.sum()),
        }
    supervised = F.cross_entropy(predicted, data["train_labels"][indices])
    kd = (
        F.kl_div(
            F.log_softmax(predicted / 2, dim=-1),
            F.softmax(target / 2, dim=-1),
            reduction="batchmean",
        )
        * 4
    )
    return 0.5 * supervised + 0.5 * kd, {
        "cross_entropy": float(supervised.detach()),
        "distillation_kl_T2": float(kd.detach()),
    }


def main(args):
    torch.manual_seed(20261014)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    artifacts = TASK / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source = args.resume_from or PILOT / "artifacts" / (args.task + "_qat_checkpoint.pt")
    source = source.resolve()
    source_sha = pilot.sha(source)
    previous = torch.load(source, map_location="cpu", weights_only=False)
    if previous["task"] != args.task:
        raise ValueError("Checkpoint task mismatch")
    start = previous["step"]
    if args.targets != sorted(set(args.targets)) or any(t <= start for t in args.targets):
        raise ValueError("Targets must be increasing, distinct and above the resumed step")
    for target in args.targets:
        stem = f"{args.task}_step{target:04d}"
        if (TASK / "results" / (stem + ".json")).exists() or (artifacts / (stem + ".pt")).exists():
            raise FileExistsError(f"Already recorded {stem}; choose a later resume/target")

    data = pilot.load_datasets(args.task)
    teacher = (
        pilot.mlsd_teacher()
        if args.task == "mlsd"
        else load_text(PILOT / "artifacts/distilbert_weights.npz")
    )
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher_output = pilot.forward_all(teacher, data["evaluation"], args.task)
    gate = pilot.numerical(data["reference"], teacher_output)
    if gate["relative_l2"] > 1e-4 or (args.task == "text" and gate["max_abs"] > 1e-4):
        raise RuntimeError("FP32 teacher fails same-checkpoint equivalence gate")
    qat, config = prepare_qat(teacher, args.task)
    for name, parameter in qat.named_parameters():
        parameter.requires_grad = not (args.task == "text" and "distilbert.embeddings." in name)
    qat.load_state_dict(previous["model"], strict=True)
    qat.apply(disable_observer)
    optimizer = torch.optim.AdamW(
        [p for p in qat.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01
    )
    optimizer.load_state_dict(previous["optimizer"])
    torch.set_rng_state(previous["torch_rng_state"])
    del previous

    before = pilot.forward_all(qat, data["evaluation"], args.task)
    if start == 8 and source.parent == PILOT / "artifacts":
        old_output = np.load(PILOT / "artifacts" / (args.task + "_after.npy"))
        if not np.array_equal(before, old_output):
            raise RuntimeError("Resumed output differs from saved day-14 output")
    count = len(data["train"][0])
    batch = 1 if args.task == "mlsd" else 2
    order = np.random.default_rng(20261014).permutation(count)
    teacher_metrics = pilot.evaluate(teacher_output, data, args.task)
    print(f"{args.task}: resumed step {start}; teacher and resume gates passed", flush=True)

    for target in args.targets:
        stem = f"{args.task}_step{target:04d}"
        before_path = artifacts / (stem + "_before.npy")
        after_path = artifacts / (stem + "_after.npy")
        np.save(before_path, before, allow_pickle=False)
        before_hash = pilot.weight_hash(qat)
        history = []
        qat.train()
        for step in range(start, target):
            indices = order[(np.arange(batch) + step * batch) % count]
            item = tuple(x[indices] for x in data["train"])
            with torch.no_grad():
                teacher_target = teacher(*item)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            predicted = qat(*item)
            loss, components = training_loss(predicted, teacher_target, data, indices, args.task)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite QAT loss")
            loss.backward()
            if any(
                not torch.isfinite(p.grad).all() for p in qat.parameters() if p.grad is not None
            ):
                raise RuntimeError("Non-finite gradient")
            norm = torch.nn.utils.clip_grad_norm_(qat.parameters(), 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "train_indices": indices.tolist(),
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm),
                    "seconds": time.perf_counter() - started,
                    **components,
                }
            )
            if (step + 1) % 8 == 0 or step + 1 == target:
                print(f"{args.task} step {step + 1}: loss={float(loss.detach()):.6f}", flush=True)
        after = pilot.forward_all(qat, data["evaluation"], args.task)
        if not np.isfinite(after).all():
            raise RuntimeError("Non-finite evaluation output")
        np.save(after_path, after, allow_pickle=False)
        after_hash = pilot.weight_hash(qat)
        if before_hash == after_hash:
            raise RuntimeError("Training did not update weights")
        checkpoint = artifacts / (stem + ".pt")
        torch.save(
            {
                "model": qat.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": target,
                "torch_rng_state": torch.get_rng_state(),
                "task": args.task,
                "config": config,
            },
            checkpoint,
        )
        record = {
            "task_date": "2026-10-15",
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "status": "in_progress",
            "task": args.task,
            "framework": torch.__version__,
            "device": "CPU",
            "cuda_available": torch.cuda.is_available(),
            "threads": 4,
            "start_step": start,
            "steps_this_run": target - start,
            "total_steps": target,
            "batch_size": batch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer": "AdamW, weight_decay=.01, gradient clipping 1.0",
            "observer_schedule": "Restored frozen day-14 ranges; fake quant stays enabled",
            "sampling": "Continue fixed seed-20261014 permutation, cycle without reshuffle",
            "data": {
                "train_examples_available": count,
                "evaluation_examples": len(data["evaluation"][0]),
                "train_samples_this_run": (target - start) * batch,
                "train_samples_total": target * batch,
                "unique_train_examples_total": min(target * batch, count),
                "equivalent_train_epochs": target * batch / count,
                "evaluation_used_in_training": False,
                "labels": "FP32 teacher pseudo-labels on Wireframe train"
                if args.task == "mlsd"
                else "SST-2 train labels + FP32 teacher distillation",
                "manifest_sha256": pilot.sha(
                    pilot.DAY13 / "results/data_manifest.json"
                    if args.task == "mlsd"
                    else PILOT / "results/data_manifest.json"
                ),
            },
            "config": config,
            "original_teacher_gate": gate,
            "original_teacher": teacher_metrics,
            "fake_quant_before": pilot.evaluate(before, data, args.task),
            "fake_quant_after": pilot.evaluate(after, data, args.task),
            "history": history,
            "weights_before_sha256": before_hash,
            "weights_after_sha256": after_hash,
            "resumed_from": {"path": relative(source), "sha256": source_sha, "step": start},
            "checkpoint": {"path": relative(checkpoint), "sha256": pilot.sha(checkpoint)},
            "outputs": {
                "before": {"path": relative(before_path), "sha256": pilot.sha(before_path)},
                "after": {"path": relative(after_path), "sha256": pilot.sha(after_path)},
            },
            "source_sha256": {
                relative(p): pilot.sha(p)
                for p in [Path(__file__), PILOT / "src/train.py", PILOT / "src/models.py"]
            },
            "limitations": [
                "Small fixed evaluation subset, repeatedly monitored; not a fresh final test set",
                "Only fake-quant training metrics; deployment INT8 must be evaluated separately",
                "M-LSD uses teacher pseudo-labels, not ground-truth supervised training",
            ],
        }
        pilot.write_json(TASK / "results" / (stem + ".json"), record)
        metric = record["fake_quant_after"]
        value = (
            metric["line_quality"]["sAP10_subset_percent"]
            if args.task == "mlsd"
            else metric["accuracy"]
        )
        print(f"SAVED {stem}: quality={value:.6f}", flush=True)
        source, source_sha, start, before = checkpoint, pilot.sha(checkpoint), target, after
    print(json.dumps({"task": args.task, "total_steps": start, "status": "completed_today"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mlsd", "text"], required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--resume-from", type=Path)
    main(parser.parse_args())
