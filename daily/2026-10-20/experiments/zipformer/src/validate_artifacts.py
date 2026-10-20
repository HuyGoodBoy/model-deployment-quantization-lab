"""Creation is not execution: validate LiteRT allocation/contracts and strict errors."""

import traceback

import onnx
from ai_edge_litert.interpreter import Interpreter
from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from support import (
    ART,
    RESULTS,
    LiteRunner,
    OnnxRunner,
    info,
    now,
    numeric,
    source,
    write,
    zeros_for_model,
)


def main():
    for variant in ["raw", "static"]:
        target = RESULTS / f"runtime_validation_{variant}.json"
        if target.exists():
            continue
        rows = {}
        for component in ["decoder", "joiner", "encoder"]:
            path = ART / "conversion" / variant / component / "output/input_float32.tflite"
            row = {"model": info(path)}
            raw = Interpreter(model_path=str(path), num_threads=4)
            row["inputs"] = [
                {"name": x["name"], "shape": x["shape"].tolist(), "dtype": str(x["dtype"])}
                for x in raw.get_input_details()
            ]
            row["operators"] = [x["op_name"] for x in raw._get_ops_details()]
            del raw
            try:
                runtime = LiteRunner(path)
                feed = zeros_for_model(source(component))
                output = runtime.run(feed)
                original = OnnxRunner(source(component)).run(feed)
                row["parity"] = {name: numeric(original[name], output[name]) for name in original}
                row["status"] = "allocated_and_invoked"
            except Exception as error:  # noqa: BLE001 -- record any converter/runtime failure as evidence
                row.update(
                    status="runtime_failed", error_type=type(error).__name__, error=str(error)
                )
            rows[component] = row
            print("RUNTIME", variant, component, row["status"], row.get("error", ""), flush=True)
        write(target, {"evaluated_at_utc": now(), "models": rows, "source_code": info(__file__)})
    target = RESULTS / "strict_decoder_error_modelproto.json"
    if not target.exists():
        model = onnx.load(ART / "conversion/static/decoder/input.onnx")
        graph = model
        folder = ART / "strict_decoder"
        folder.mkdir(exist_ok=True)
        try:
            export_tflite_model_flatbuffer_direct(
                onnx_graph=graph,
                output_folder_path=str(folder),
                output_file_name="decoder",
                flatbuffer_direct_allow_custom_ops=False,
                keep_shape_absolutely_input_names=["y"],
            )
            record = {"status": "strict_conversion_succeeded"}
        except Exception as error:  # noqa: BLE001 -- strict negative test must preserve full traceback
            text = traceback.format_exc()
            (folder / "traceback.txt").write_text(text, encoding="utf-8", newline="\n")
            record = {
                "status": "strict_conversion_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": text,
                "log": info(folder / "traceback.txt"),
            }
            print("STRICT ERROR", text[-2400:], flush=True)
        write(target, {"recorded_at_utc": now(), **record, "source_code": info(__file__)})


if __name__ == "__main__":
    main()
