"""Inspect quantization scales from a TFLite FlatBuffer, without inference."""
import argparse
import importlib.util
from pathlib import Path

from .common import ARTIFACTS, RESULTS, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cv', 'text'), default='text')
    parser.add_argument('--variant', default='tflite_static')
    args = parser.parse_args()
    if args.variant not in ('tflite_fp32', 'tflite_dynamic', 'tflite_fp16', 'tflite_static'):
        parser.error('Expected a known TFLite variant')
    # The generated schema is pure Python; avoid initializing TensorFlow kernels.
    tf_spec = importlib.util.find_spec('tensorflow')
    schema_path = Path(tf_spec.origin).parent / 'lite/python/schema_py_generated.py'
    spec = importlib.util.spec_from_file_location('tflite_schema_inspector', schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    path = ARTIFACTS / args.task / f'{args.variant}.tflite'
    blob = path.read_bytes()
    graph = schema.Model.GetRootAsModel(blob, 0).Subgraphs(0)
    scales = []
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        q = tensor.Quantization()
        if q is None or q.ScaleLength() == 0:
            continue
        scales.append({'name': tensor.Name().decode(), 'index': index,
                       'max_scale': max(float(q.Scale(i)) for i in range(q.ScaleLength())),
                       'tensor_type_enum': tensor.Type(),
                       'zero_points': [int(q.ZeroPoint(i)) for i in range(min(4, q.ZeroPointLength()))]})
    scales.sort(key=lambda row: row['max_scale'], reverse=True)
    record = {'model_sha256': sha256(path), 'largest_scales': scales[:15],
              'scales_above_one_million': sum(row['max_scale'] > 1e6 for row in scales),
              'scope': 'FlatBuffer metadata inspection; does not prove a sole causal explanation'}
    write_json(RESULTS / args.task / f'diagnostic_{args.variant}.json', record)
    print(f"Inspected {args.task}/{args.variant}: {record['scales_above_one_million']} scales > 1e6")
    for row in scales[:6]:
        print(f"{row['max_scale']:.6g}: {row['name']}")


if __name__ == '__main__':
    main()
