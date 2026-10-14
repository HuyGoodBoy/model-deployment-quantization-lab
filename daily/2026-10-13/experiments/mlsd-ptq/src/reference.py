"""Run the unchanged, pinned PyTorch checkpoint on the new inputs (no training)."""

import importlib.util
import json

import numpy as np
import torch

from data import ROOT, TASK, load, sha, write_json


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source = ROOT / ".cache/mlsd/mbv2_mlsd_tiny.py"
    checkpoint = ROOT / ".cache/mlsd/mlsd_tiny_512_fp32.pth"
    lock = json.loads((ROOT / "daily/2026-10-09/experiments/03-mlsd/sources.lock.json").read_text())
    for path in [source, checkpoint]:
        if sha(path) != lock[path.name]["sha256"]:
            raise RuntimeError("Original source/checkpoint changed")
    spec = importlib.util.spec_from_file_location("pinned_mlsd", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MobileV2_MLSD_Tiny().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    inputs, _, manifest = load()
    folder = TASK / "artifacts"
    folder.mkdir(exist_ok=True)
    hashes = {}
    for split in ["calibration", "evaluation"]:
        outputs = []
        with torch.inference_mode():
            for index, image in enumerate(inputs[split]):
                x = torch.from_numpy((image / 127.5 - 1).transpose(2, 0, 1)[None])
                outputs.append(model(x).numpy().transpose(0, 2, 3, 1)[..., :5])
        path = folder / f"reference_{split}.npy"
        np.save(path, np.concatenate(outputs), allow_pickle=False)
        hashes[split] = sha(path)
        print(f"Torch reference {split}: {len(outputs)} images", flush=True)
    write_json(
        TASK / "results/reference.json",
        {
            "torch": torch.__version__,
            "checkpoint_sha256": sha(checkpoint),
            "source_sha256": sha(source),
            "input_sha256": manifest["inputs_sha256"],
            "outputs_sha256": hashes,
            "scope": "unchanged PyTorch model, first five active output channels; no training",
        },
    )


if __name__ == "__main__":
    main()
