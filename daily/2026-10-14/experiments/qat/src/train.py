"""Short QAT pilot only. Train with fake quantization; INT8 export is a later milestone."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from models import ActiveMlsd, load_text, prepare_qat
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
)
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
DAY13 = ROOT / "daily/2026-10-13/experiments/mlsd-ptq"
sys.path.insert(0, str(DAY13 / "src"))
from metrics import line_metrics, numerical

from data import load as line_data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def weight_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def mlsd_teacher():
    path = ROOT / ".cache/mlsd/mbv2_mlsd_tiny.py"
    spec = importlib.util.spec_from_file_location("mlsd_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MobileV2_MLSD_Tiny().eval()
    model.load_state_dict(
        torch.load(
            ROOT / ".cache/mlsd/mlsd_tiny_512_fp32.pth", map_location="cpu", weights_only=True
        )
    )
    return ActiveMlsd(model).eval()


def load_datasets(task):
    if task == "mlsd":
        inputs, gt, manifest = line_data()

        def tensor(x):
            return torch.from_numpy(((x / 127.5 - 1).transpose(0, 3, 1, 2)).copy())

        return {
            "train": (tensor(inputs["calibration"]),),
            "calibration": (tensor(inputs["calibration"]),),
            "evaluation": (tensor(inputs["evaluation"]),),
            "ground_truth": gt,
            "labels": None,
            "manifest": manifest,
            "reference": np.load(DAY13 / "artifacts/reference_evaluation.npy", allow_pickle=False),
        }
    train = np.load(TASK / "artifacts/text_train.npz", allow_pickle=False)
    calibration = np.load(ROOT / "quantization/data/text/calibration.npz", allow_pickle=False)
    evaluation = np.load(ROOT / "quantization/data/text/evaluation.npz", allow_pickle=False)

    def tensors(data):
        return tuple(
            torch.from_numpy(data[name].astype(np.int64))
            for name in ["input_ids", "attention_mask"]
        )

    with np.load(ROOT / "quantization/results/text/outputs.npz", allow_pickle=False) as output:
        reference = output["tensorflow_fp32"].copy()
    return {
        "train": tensors(train),
        "calibration": tensors(calibration),
        "evaluation": tensors(evaluation),
        "train_labels": torch.from_numpy(train["labels"]),
        "labels": evaluation["labels"],
        "reference": reference,
        "manifest": json.loads((TASK / "results/data_manifest.json").read_text(encoding="utf-8")),
    }


def forward_all(model, inputs, task):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(len(inputs[0])):
            outputs.append(model(*(x[i : i + 1] for x in inputs)).cpu().numpy())
    output = np.concatenate(outputs)
    return output.transpose(0, 2, 3, 1) if task == "mlsd" else output


def evaluate(output, data, task):
    metrics = numerical(data["reference"], output)
    if task == "mlsd":
        return {
            "active": metrics,
            "center": numerical(data["reference"][..., :1], output[..., :1]),
            "displacement": numerical(data["reference"][..., 1:5], output[..., 1:5]),
            "line_quality": line_metrics(output, data["ground_truth"]),
        }
    return {
        "logits": metrics,
        "accuracy": float(np.mean(np.argmax(output, axis=-1) == data["labels"])),
        "agreement_with_original": float(
            np.mean(np.argmax(output, axis=-1) == np.argmax(data["reference"], axis=-1))
        ),
    }


def main(args):
    torch.manual_seed(20261014)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    ART = TASK / "artifacts"
    ART.mkdir(exist_ok=True)
    data = load_datasets(args.task)
    teacher = mlsd_teacher() if args.task == "mlsd" else load_text(ART / "distilbert_weights.npz")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_output = forward_all(teacher, data["evaluation"], args.task)
    gate = numerical(data["reference"], teacher_output)
    if gate["relative_l2"] > 1e-4 or (args.task == "text" and gate["max_abs"] > 1e-4):
        write_json(TASK / "results" / (args.task + "_gate_failed.json"), gate)
        raise RuntimeError("FP32 teacher fails same-checkpoint equivalence gate")
    qat, config = prepare_qat(teacher, args.task)
    # deepcopy preserves teacher requires_grad=False: restore trainability except frozen embeddings.
    for name, p in qat.named_parameters():
        p.requires_grad = not (args.task == "text" and "distilbert.embeddings." in name)
    qat.apply(disable_fake_quant)
    qat.apply(disable_observer)
    fused_output = forward_all(qat, tuple(x[:1] for x in data["evaluation"]), args.task)
    fused_gate = numerical(teacher_output[:1], fused_output)
    if fused_gate["relative_l2"] > 1e-4:
        raise RuntimeError("BN folding changed FP32 output too much")
    qat.apply(enable_fake_quant)
    qat.apply(enable_observer)
    qat.eval()
    with torch.no_grad():
        for i in range(len(data["calibration"][0])):
            qat(*(x[i : i + 1] for x in data["calibration"]))
    qat.apply(disable_observer)
    optimizer = torch.optim.AdamW(
        [p for p in qat.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    start_step = 0
    if args.resume:
        previous = torch.load(
            ART / (args.task + "_qat_checkpoint.pt"), map_location="cpu", weights_only=False
        )
        qat.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start_step = previous["step"]
        torch.set_rng_state(previous["torch_rng_state"])
    before = forward_all(qat, data["evaluation"], args.task)
    np.save(ART / (args.task + "_before.npy"), before, allow_pickle=False)
    before_hash = weight_hash(qat)
    train_count = len(data["train"][0])
    batch = 1 if args.task == "mlsd" else 2
    order = np.random.default_rng(20261014).permutation(train_count)
    history = []
    qat.train()
    for offset in range(args.steps):
        step = start_step + offset
        indices = order[(np.arange(batch) + step * batch) % train_count]
        item = tuple(x[indices] for x in data["train"])
        with torch.no_grad():
            target = teacher(*item)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        predicted = qat(*item)
        if args.task == "mlsd":
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
            loss = classification + regression
            components = {
                "soft_center_bce": float(classification.detach()),
                "positive_displacement_smooth_l1": float(regression.detach()),
                "positive_regression_values": int(mask.sum()),
            }
        else:
            labels = data["train_labels"][indices]
            supervised = F.cross_entropy(predicted, labels)
            kd = (
                F.kl_div(
                    F.log_softmax(predicted / 2, dim=-1),
                    F.softmax(target / 2, dim=-1),
                    reduction="batchmean",
                )
                * 4
            )
            loss = 0.5 * supervised + 0.5 * kd
            components = {
                "cross_entropy": float(supervised.detach()),
                "distillation_kl_T2": float(kd.detach()),
            }
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite QAT loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in qat.parameters() if p.grad is not None):
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
        print(f"{args.task} QAT step {step + 1}: loss={float(loss.detach()):.6f}", flush=True)
    after = forward_all(qat, data["evaluation"], args.task)
    np.save(ART / (args.task + "_after.npy"), after, allow_pickle=False)
    after_hash = weight_hash(qat)
    if before_hash == after_hash:
        raise RuntimeError("Training did not update weights")
    checkpoint = ART / (args.task + "_qat_checkpoint.pt")
    torch.save(
        {
            "model": qat.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": start_step + args.steps,
            "torch_rng_state": torch.get_rng_state(),
            "task": args.task,
            "config": config,
        },
        checkpoint,
    )
    record = {
        "task_date": "2026-10-14",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "status": "in_progress",
        "milestone": "fake-quant QAT pilot ran; full weekly training/evaluation and deployable INT8 export remain",
        "task": args.task,
        "framework": torch.__version__,
        "device": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "threads": 4,
        "steps_this_run": args.steps,
        "total_steps": start_step + args.steps,
        "batch_size": batch,
        "learning_rate": args.lr,
        "optimizer": "AdamW, weight_decay=.01, gradient clipping 1.0",
        "observer_schedule": "MinMax warm-up on train/calibration only, then frozen; fake quant remains enabled during training/evaluation",
        "data": {
            "train_examples_available": train_count,
            "observer_examples": len(data["calibration"][0]),
            "evaluation_examples": len(data["evaluation"][0]),
            "train_samples_this_run": args.steps * batch,
            "evaluation_used_in_training": False,
            "labels": "FP32 teacher pseudo-labels on Wireframe train (not ground truth training)"
            if args.task == "mlsd"
            else "SST-2 train labels + frozen same-checkpoint FP32 teacher logits",
        },
        "config": config,
        "original_teacher_gate": gate,
        "folding_gate": fused_gate,
        "original_teacher": evaluate(teacher_output, data, args.task),
        "fake_quant_before": evaluate(before, data, args.task),
        "fake_quant_after": evaluate(after, data, args.task),
        "history": history,
        "trainable_parameters": sum(p.numel() for p in qat.parameters() if p.requires_grad),
        "weights_before_sha256": before_hash,
        "weights_after_sha256": after_hash,
        "checkpoint_sha256": sha(checkpoint),
        "output_hashes": {
            "before": sha(ART / (args.task + "_before.npy")),
            "after": sha(ART / (args.task + "_after.npy")),
        },
        "not_yet_done": [
            "longer training and independent validation",
            "export fake quant graph to deployed INT8 and audit actual operator coverage",
            "measure deployed INT8 accuracy and latency",
        ],
        "limitations": [
            "CPU-only installed PyTorch build, GPU not used for this pilot",
            "some operators deliberately remain float; fake quant is not integer execution",
            "8-step pilot is not evidence of convergence or final quality",
        ],
    }
    write_json(TASK / "results" / (args.task + "_progress.json"), record)
    print(args.task, "pilot complete; weekly task still in progress", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mlsd", "text"], required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("Positive steps required")
    main(args)
