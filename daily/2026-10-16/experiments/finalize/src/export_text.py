"""Export frozen DistilBERT QAT Linear layers to static 8-bit MatMulInteger."""

import gc
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from common import ART, DAY15, RESULTS, checked, info, profile_summary, read, session, write
from models import QatOperation, load_text, prepare_qat
from torch import nn
from torch.ao.quantization import disable_fake_quant, disable_observer, enable_fake_quant
from torch.onnx import symbolic_helper
from train_continue import PILOT, pilot


class IntegerLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, xs, xz, ws, bias):
        x = torch.fake_quantize_per_tensor_affine(x, xs, xz.to(torch.int32) - 128, -128, 127)
        w = (weight.float() - 128) * ws.reshape(1, -1)
        return torch.nn.functional.linear(x, w.T.contiguous(), bias)

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v", "v", "v")
    def symbolic(g, x, weight, xs, xz, ws, bias):
        q = g.op("QuantizeLinear", x, xs, xz)
        wz = g.op("Constant", value_t=torch.tensor(128, dtype=torch.uint8))
        acc = g.op("MatMulInteger", q, weight, xz, wz)
        scaled = g.op("Mul", g.op("Cast", acc, to_i=onnx.TensorProto.FLOAT), g.op("Mul", xs, ws))
        return g.op("Add", scaled, bias)


class ExportLinear(nn.Module):
    def __init__(self, operation):
        super().__init__()
        aq, wq, layer = (
            operation.activation_fake_quant,
            operation.weight_fake_quant,
            operation.layer,
        )
        assert aq.quant_min == -128 and aq.quant_max == 127 and torch.all(wq.zero_point == 0)
        self.register_buffer("xs", aq.scale.detach().reshape(()).clone())
        self.register_buffer("xz", (aq.zero_point.detach() + 128).to(torch.uint8).reshape(()))
        self.register_buffer("ws", wq.scale.detach().clone())
        # The fake-quant operator is the oracle; do not introduce a different weight rounding formula.
        fake_weight = wq(layer.weight.detach())
        codes = torch.clamp(torch.round(fake_weight / self.ws[:, None]), -127, 127)
        self.register_buffer("weight", (codes.T.contiguous() + 128).to(torch.uint8))
        self.register_buffer(
            "bias",
            layer.bias.detach().clone()
            if layer.bias is not None
            else torch.zeros(layer.out_features),
        )

    def forward(self, x):
        return IntegerLinear.apply(x, self.weight, self.xs, self.xz, self.ws, self.bias)


def replace(parent):
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, QatOperation):
            assert isinstance(child.layer, nn.Linear)
            parent._modules[name] = ExportLinear(child)
            count += 1
        else:
            count += replace(child)
    return count


def export(model, sample, path):
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(path),
            dynamo=False,
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
        )
    onnx.checker.check_model(str(path), full_check=True)
    print("EXPORTED", path.name, flush=True)


def infer(runtime, inputs):
    outputs = []
    for index in range(len(inputs[0])):
        outputs.append(
            runtime.run(
                None,
                {
                    "input_ids": inputs[0][index : index + 1],
                    "attention_mask": inputs[1][index : index + 1],
                },
            )[0]
        )
    return np.concatenate(outputs)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if (RESULTS / "text_export.json").exists():
        raise FileExistsError("Preserve the existing text deployment record")
    ART.mkdir(parents=True, exist_ok=True)
    # Declare task-specific export criteria before inspecting deployment outputs.
    gate = {
        "maximum_relative_l2": 0.02,
        "minimum_label_agreement": 0.99,
        "maximum_accuracy_delta_pp": 1.0,
        "reason": "A 2% logit parity tolerance with at least 99/100 unchanged labels and at most one accuracy point on development; separate numerical and decision checks; not a population quality guarantee",
    }
    write(
        RESULTS / "text_export_protocol.json",
        {
            "declared_at_utc": datetime.now(UTC).isoformat(),
            "gate": gate,
            "scope": "Static trained W8A8 Linear; no dynamic calibration; embedding, attention matmuls, normalization, activation and boundaries remain float",
        },
    )
    old = read(DAY15 / "results/text_step0128.json")
    checkpoint = checked(old["checkpoint"])
    teacher = load_text(PILOT / "artifacts/distilbert_weights.npz")
    data = pilot.load_datasets("text")
    sample = tuple(x[:1] for x in data["evaluation"])
    teacher_path = ART / "text_teacher_fp32.onnx"
    export(teacher, sample, teacher_path)
    model, config = prepare_qat(teacher, "text")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(saved["model"], strict=True)
    del saved, teacher
    model.apply(disable_observer)
    model.eval()
    reference = np.load(checked(old["outputs"]["after"]))
    current = pilot.forward_all(model, tuple(x[:4] for x in data["evaluation"]), "text")
    assert np.array_equal(current, reference[:4])
    model.apply(disable_fake_quant)
    float_path = ART / "text_qat_weights_fp32.onnx"
    float_reference = pilot.forward_all(model, tuple(x[:4] for x in data["evaluation"]), "text")
    export(model, sample, float_path)
    model.apply(enable_fake_quant)
    count = replace(model)
    lowered = pilot.forward_all(model, tuple(x[:4] for x in data["evaluation"]), "text")
    lowering_parity = pilot.numerical(reference[:4], lowered)
    assert lowering_parity["relative_l2"] < 1e-5
    integer_path = ART / "text_qat_static_integer.onnx"
    export(model, sample, integer_path)
    del model
    gc.collect()
    optimized = ART / "text_qat_static_integer_optimized.onnx"
    runtime = session(integer_path, optimized, ART / "text_integer_profile")
    runtime.run(None, {"input_ids": sample[0].numpy(), "attention_mask": sample[1].numpy()})
    profile = profile_summary(Path(runtime.end_profiling()))
    del runtime
    operators = profile["unique_executed_operators"]
    executed = sum(n for op, n in operators.items() if "MatMulInteger" in op)
    assert executed == count == 38, (operators, count)
    arrays = tuple(x.numpy() for x in data["evaluation"])
    records = {}
    for name, path in {
        "teacher_fp32": teacher_path,
        "qat_weights_fp32": float_path,
        "qat_static_integer": integer_path,
    }.items():
        runtime = session(path)
        outputs = infer(runtime, arrays)
        del runtime
        target = ART / f"text_{name}_development.npy"
        np.save(target, outputs, allow_pickle=False)
        records[name] = {
            "model": info(path),
            "output": info(target),
            "metrics": pilot.evaluate(outputs, data, "text"),
        }
        if name == "qat_weights_fp32":
            assert pilot.numerical(float_reference, outputs[:4])["relative_l2"] < 1e-4
        print("DEVELOPMENT", name, records[name]["metrics"]["accuracy"], flush=True)
    actual = np.load(checked(records["qat_static_integer"]["output"]))
    parity = pilot.numerical(reference, actual)
    agreement = float(np.mean(actual.argmax(-1) == reference.argmax(-1)))
    delta = (
        np.mean(actual.argmax(-1) == data["labels"])
        - np.mean(reference.argmax(-1) == data["labels"])
    ) * 100
    gate["passed"] = (
        parity["relative_l2"] <= 0.02 and agreement >= 0.99 and abs(delta) <= 1.0 + 1e-10
    )
    graph = onnx.load(integer_path)
    assert sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 38
    write(
        RESULTS / "text_export.json",
        {
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": old["checkpoint"],
            "models": records,
            "fake_quant_reference": old["outputs"]["after"],
            "float_lowering_parity_first_four": lowering_parity,
            "fake_quant_to_integer_parity": parity,
            "label_agreement": agreement,
            "accuracy_delta_pp": float(delta),
            "gate": gate,
            "integer_linear_layers": count,
            "graph_operators": dict(Counter(n.op_type for n in graph.graph.node)),
            "runtime": profile,
            "is_full_integer_model": False,
            "format": "Static frozen QAT scales; UINT8 offset x UINT8 weight -> INT32 MatMulInteger; per-output-channel weight scales",
            "float_regions": config["float_regions"],
            "source_code": info(__file__),
        },
    )
    print("TEXT EXPORT", parity, "agreement", agreement, "gate", gate["passed"], flush=True)


if __name__ == "__main__":
    main()
