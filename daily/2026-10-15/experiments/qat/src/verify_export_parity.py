"""Check export arithmetic against PyTorch and isolate convolution backend sensitivity."""

import io
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from export_integer import ExportConv, infer_all, make_session
from export_parity import explicit_rounding, quantizer_audit
from models import QatOperation
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from research_mlsd import load_model, read
from train_continue import ROOT, TASK, pilot, relative

ART = TASK / "artifacts/export-parity"
RESULTS = TASK / "results/export-parity"


def rounding_cases():
    """Adversarial ties, adjacent FP32 values, clipping and channel broadcasting."""
    results = []
    for kind in ("signed_tensor", "unsigned_tensor", "signed_channel"):
        channel = kind.endswith("channel")
        unsigned = kind.startswith("unsigned")
        scales = np.array([0.023529412, 0.24889714, 0.49074554, 0.87240624], np.float32)
        zeros = np.array([-128, -1, 5, 112], np.int32)
        if not channel:
            scales, zeros = scales[:1], zeros[:1]
        dtype = np.uint8 if unsigned else np.int8
        zero = (zeros + (128 if unsigned else 0)).astype(dtype)
        ties = np.array(
            [-500, -129.5, -128.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, 127.5, 500], np.float32
        )
        x = np.broadcast_to(
            ties[None, None, None, :] * scales.reshape(1, -1, 1, 1), (1, 4, 1, 12)
        ).copy()
        x = np.concatenate([np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf)], axis=2)
        qtype = T.UINT8 if unsigned else T.INT8
        attrs = {"axis": 1} if channel else {}
        graph = h.make_graph(
            [
                h.make_node("QuantizeLinear", ["input", "scale", "zero"], ["q"], **attrs),
                h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["output"], **attrs),
            ],
            kind,
            [h.make_tensor_value_info("input", T.FLOAT, list(x.shape))],
            [h.make_tensor_value_info("output", T.FLOAT, list(x.shape))],
            [
                nh.from_array(scales if channel else scales.reshape(()), "scale"),
                nh.from_array(zero if channel else zero.reshape(()), "zero"),
            ],
        )
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8)
        repaired, _ = explicit_rounding(model)
        session = make_session_bytes(repaired)
        actual = session.run(None, {"input": x})[0]
        del session
        if channel:
            expected = torch.fake_quantize_per_channel_affine(
                torch.from_numpy(x), torch.from_numpy(scales), torch.from_numpy(zeros), 1, -128, 127
            ).numpy()
        else:
            expected = torch.fake_quantize_per_tensor_affine(
                torch.from_numpy(x), float(scales[0]), int(zeros[0]), -128, 127
            ).numpy()
        if not np.array_equal(expected, actual):
            raise AssertionError(f"Rounding case failed: {kind}")
        results.append(
            {
                "case": kind,
                "values": x.size,
                "exact_match": True,
                "output_code_dtype": T.DataType.Name(qtype),
            }
        )
    return results


def make_session_bytes(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def local_convolutions(qat, example):
    rows, handles = [], []

    def check(name, mod, args, output):
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                ExportConv(mod),
                (args[0],),
                buffer,
                dynamo=False,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
            )
        exported, _ = explicit_rounding(onnx.load_model_from_string(buffer.getvalue()))
        session = make_session_bytes(exported)
        actual = session.run(None, {"input": args[0].numpy()})[0]
        del session
        numerical = pilot.numerical(output.numpy(), actual)
        rows.append({"name": name, **numerical})

    for name, mod in qat.named_modules():
        if isinstance(mod, QatOperation):
            handles.append(
                mod.register_forward_hook(
                    lambda mod, args, out, name=name: check(name, mod, args, out)
                )
            )
    try:
        with torch.no_grad():
            qat(example)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    record = read(RESULTS / "rounding.json")
    for path, expected in record["protected_evidence_sha256"].items():
        assert pilot.sha(ROOT / path) == expected, path
    for item in record["artifacts"].values():
        assert pilot.sha(ROOT / item["path"]) == item["sha256"]
    selection = read(TASK / "results/research/selection.json")
    deployment = read(TASK / "results/research/deployment.json")
    candidate_path = ROOT / record["artifacts"]["model"]["path"]
    baseline_path = ROOT / deployment["artifacts"]["integer_model"]["path"]
    source, candidate = onnx.load(baseline_path), onnx.load(candidate_path)
    old_weights = {item.name: item.SerializeToString() for item in source.graph.initializer}
    new_weights = {item.name: item.SerializeToString() for item in candidate.graph.initializer}
    assert all(new_weights[name] == value for name, value in old_weights.items())
    for a, b in zip(
        [n for n in source.graph.node if n.op_type != "QuantizeLinear"],
        [n for n in candidate.graph.node if not n.name.startswith("parity_rounding_")],
        strict=True,
    ):
        assert a.SerializeToString() == b.SerializeToString()
    assert sum(n.op_type == "ConvInteger" for n in candidate.graph.node) == 41
    assert not any(n.op_type in ("Conv", "FusedConv") for n in candidate.graph.node)
    cases = rounding_cases()
    print("PASS rounding ties, adjacent values, clipping and per-channel broadcasting", flush=True)
    data = pilot.load_datasets("mlsd")
    qat, *_ = load_model(ROOT / selection["checkpoint"]["path"])
    assert quantizer_audit(qat, data["evaluation"][0][:1]) == record["same_input_quantizer_audit"]
    local = local_convolutions(qat, data["evaluation"][0][:1])
    assert len(local) == 41
    print(
        "LOCAL CONVS",
        len(local),
        "max relative L2",
        max(r["relative_l2"] for r in local),
        flush=True,
    )
    reference = np.load(ROOT / selection["development_output"]["path"], allow_pickle=False)
    assert torch.backends.mkldnn.enabled
    current_reference = pilot.forward_all(qat, data["evaluation"], "mlsd")
    assert np.array_equal(current_reference, reference)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.backends.mkldnn.flags(enabled=False):
            backend_output = pilot.forward_all(qat, data["evaluation"], "mlsd")
    backend_path = ART / "mlsd_pytorch_mkldnn_disabled.npy"
    np.save(backend_path, backend_output, allow_pickle=False)
    backend = {
        "reference_mkldnn_enabled": True,
        "candidate_mkldnn_enabled": False,
        "parity": pilot.numerical(reference, backend_output),
        "metrics": pilot.evaluate(backend_output, data, "mlsd"),
        "output": {"path": relative(backend_path), "sha256": pilot.sha(backend_path)},
        "interpretation": "Same frozen PyTorch model with a different convolution backend; diagnostic only, not a deployment candidate or reason to relax the gate",
    }
    print("PYTORCH BACKEND SENSITIVITY", backend["parity"]["relative_l2"], flush=True)
    del qat
    profiled = make_session(
        candidate_path, profile_prefix=ART / "integer_profile", disable_spinning=True
    )
    profiled.run(None, {"input": data["evaluation"][0][:1].numpy()})
    profile_path = Path(profiled.end_profiling())
    events = read(profile_path)
    kernels = [
        e["args"] for e in events if e.get("cat") == "Node" and "op_name" in e.get("args", {})
    ]
    operators = Counter(item["op_name"] for item in kernels)
    assert operators["ConvInteger"] == 41 and not operators["Conv"]
    assert all(item.get("provider") == "CPUExecutionProvider" for item in kernels)
    del profiled
    session = make_session(candidate_path, disable_spinning=True)
    fresh = infer_all(session, data["evaluation"][0].numpy())
    del session
    assert np.array_equal(
        fresh, np.load(ROOT / record["artifacts"]["output"]["path"], allow_pickle=False)
    )
    assert pilot.numerical(reference, fresh) == record["candidate"]
    assert pilot.evaluate(fresh, data, "mlsd") == record["candidate_metrics"]
    gate = record["parity_gate"]
    assert gate["passed"] == (
        record["candidate"]["relative_l2"] <= 0.02 and abs(gate["actual_sAP10_delta_pp"]) <= 0.5
    )
    for path, expected in record["protected_evidence_sha256"].items():
        assert pilot.sha(ROOT / path) == expected, path
    pilot.write_json(
        RESULTS / "verification.json",
        {
            "status": "passed",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "rounding_cases": cases,
            "frozen_constants_and_nonquantizer_nodes_unchanged": True,
            "local_convolution_audit": local,
            "pytorch_backend_sensitivity": backend,
            "runtime_profile": {
                "path": relative(profile_path),
                "sha256": pilot.sha(profile_path),
                "executed_operators": dict(operators),
            },
            "full_development_output_reproduced": True,
            "export_parity_gate_passed": gate["passed"],
            "protected_evidence_sha256": record["protected_evidence_sha256"],
            "source_sha256": {
                relative(p): pilot.sha(p)
                for p in [TASK / "src/export_parity.py", TASK / "src/verify_export_parity.py"]
            },
            "scope": "Evidence verified; does not override a failed numerical export gate",
        },
    )
    print(
        "PASS export evidence, 41 executed integer convolutions and unchanged checkpoint/holdout",
        flush=True,
    )


if __name__ == "__main__":
    main()
