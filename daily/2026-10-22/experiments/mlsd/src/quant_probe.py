"""Compare native QUANTIZE with the old adapter, including ties and clipping."""

import hashlib

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from io_models import miniature
from mlsd22 import RESULTS, info, now, pilot_data, read, write


class Probe:
    def __init__(self, scale, zero, length, mode):
        kwargs = (
            {}
            if mode == "default"
            else {
                "experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            }
        )
        self.runtime = Interpreter(
            model_content=miniature(scale, zero, length), num_threads=4, **kwargs
        )
        self.runtime.allocate_tensors()
        self.input = self.runtime.get_input_details()[0]["index"]
        self.output = self.runtime.get_output_details()[0]["index"]
        self.delegates = sum(o["op_name"] == "DELEGATE" for o in self.runtime._get_ops_details())

    def run(self, value):
        self.runtime.set_tensor(
            self.input, np.ascontiguousarray(value.reshape(-1), dtype=np.float32)
        )
        self.runtime.invoke()
        return self.runtime.get_tensor(self.output)


def clipped(value, zero):
    return np.clip(value + zero, -128, 127).astype(np.int8)


def compare(value, native, scale, zero):
    flat = value.reshape(-1)
    division = flat / np.float32(scale)
    multiplication = flat * np.float32(1 / scale)
    old = clipped(np.rint(division), zero)
    alternatives = [old]
    for scaled in [division, multiplication]:
        alternatives.append(clipped(np.rint(scaled), zero))
        away = np.sign(scaled) * np.floor(np.abs(scaled).astype(np.float64) + 0.5)
        alternatives.append(clipped(away, zero))
    diff = native.astype(np.int16) - old.astype(np.int16)
    explained = np.logical_or.reduce([native == a for a in alternatives])
    return {
        "elements": flat.size,
        "mismatches_vs_numpy": int(np.count_nonzero(diff)),
        "max_abs_LSB": int(np.max(np.abs(diff))),
        "unexplained_by_divide_multiply_even_away_rounding": int(np.count_nonzero(~explained)),
        "native_min": int(native.min()),
        "native_max": int(native.max()),
        "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        "numpy_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
    }


def main():
    target = RESULTS / "quantize_probe.json"
    if target.exists():
        raise FileExistsError(target)
    model = read(RESULTS / "model_M1.json")
    scale, zero = model["change"]["scale"], model["change"]["zero_point"]
    _, data = pilot_data()
    arrays = {
        "calibration": data["calibration"][0].numpy(),
        "development": data["evaluation"][0].numpy(),
    }
    centers = (np.arange(-130, 131, dtype=np.float32) + np.float32(0.5) - zero) * np.float32(scale)
    synthetic = np.concatenate(
        [
            centers,
            np.nextafter(centers, np.float32(-np.inf)),
            np.nextafter(centers, np.float32(np.inf)),
            np.array([-4, -2, -1, -scale, 0, scale, 1, 2, 4], np.float32),
        ]
    )
    random = np.random.default_rng(20261022).uniform(-4, 4, 65536).astype(np.float32)
    rows = {}
    for mode in ["default", "builtin"]:
        special = {}
        for name, values in [
            ("half_bins_neighbors_clipping_zero", synthetic),
            ("random_finite_minus4_to4", random),
        ]:
            probe = Probe(scale, zero, values.size, mode)
            native = probe.run(values)
            special[name] = {
                "comparison": compare(values, native, scale, zero),
                "delegate_nodes": probe.delegates,
            }
            if name.startswith("half"):
                special[name].update(input_values=values.tolist(), native_values=native.tolist())
        probe = Probe(scale, zero, arrays["development"][0].size, mode)
        real = []
        for kind, images in arrays.items():
            for index, image in enumerate(images):
                native = probe.run(image)
                real.append(
                    {"dataset": kind, "index": index, **compare(image, native, scale, zero)}
                )
        all_checks = [v["comparison"] for v in special.values()] + real
        passed = all(
            r["max_abs_LSB"] <= 1 and r["unexplained_by_divide_multiply_even_away_rounding"] == 0
            for r in all_checks
        )
        rows[mode] = {
            "special": special,
            "real_images": real,
            "gate_passed": passed,
            "real_mismatches": sum(r["mismatches_vs_numpy"] for r in real),
            "real_elements": sum(r["elements"] for r in real),
            "real_max_abs_LSB": max(r["max_abs_LSB"] for r in real),
            "real_delegate_nodes": probe.delegates,
        }
        print(
            "QUANTIZE PROBE",
            mode,
            "pass",
            passed,
            "real_mismatches",
            rows[mode]["real_mismatches"],
            flush=True,
        )
    write(
        target,
        {
            "finished_at_utc": now(),
            "scale": scale,
            "zero_point": zero,
            "modes": rows,
            "rounding_gate_passed": all(r["gate_passed"] for r in rows.values()),
            "input_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
            "scope": "Standalone QUANTIZE with C1 parameters;half-bin/neighbors/clipping/zero and bounded finite random inputs. Mismatch<=1LSB must match divide/multiply+even/away alternatives. Does not prove the in-model delegated QUANTIZE uses identical kernel;full development AP remains required.",
            "source_code": info(__file__),
        },
    )


if __name__ == "__main__":
    main()
