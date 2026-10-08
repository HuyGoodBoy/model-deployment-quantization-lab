"""Convert FP16 or measure one ONNX graph/provider in an isolated process."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import subprocess
import time
import traceback

import numpy as np

from .common import (ARTIFACTS, RESULTS, ROOT, classification_metrics, load_inputs,
                     numerical_metrics, samples, sha256, system_info, write_json)
from .gpu_runtime import (GPU_VARIANTS, configure_runtime, create_session,
                          gpu_model_path, package_versions, summarize_profile)


def convert_fp16(task):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel
    source = gpu_model_path(task, 'onnx_fp32')
    target = gpu_model_path(task, 'onnx_fp16')
    target.parent.mkdir(parents=True, exist_ok=True)
    model = convert_float_to_float16(onnx.load(source), keep_io_types=True,
                                    min_positive_val=1e-7, max_finite_val=1e4)
    # IO-preserving conversion appends Cast nodes; put them before consumers.
    OnnxModel(model).topological_sort(is_deterministic=True)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    audit = {'initializer_types': dict(Counter(onnx.TensorProto.DataType.Name(x.data_type)
                                               for x in model.graph.initializer)),
             'operator_counts': dict(Counter(x.op_type for x in model.graph.node)),
             'io': [{'name': x.name, 'type': onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type)}
                    for x in (*model.graph.input, *model.graph.output)]}
    write_json(RESULTS / 'gpu' / task / 'conversion_onnx_fp16.json',
               {'status': 'ok', 'source_sha256': sha256(source), 'sha256': sha256(target),
                'size_bytes': target.stat().st_size, 'audit': audit,
                'settings': {'keep_io_types': True, 'min_positive_val': 1e-7,
                             'max_finite_val': 1e4, 'default_op_block_list': True,
                             'topological_sort_after_conversion': True},
                'note': 'Floating-point precision conversion, not integer PTQ. Unsupported ops may stay FP32. Constants outside finite bounds are clipped; attention mask sentinel may change.'})
    print(f'Converted {task} to ONNX FP16: {target.stat().st_size/1048576:.2f} MiB', flush=True)


def original_comparison(task, output, labels):
    with np.load(RESULTS / task / 'outputs.npz', allow_pickle=False) as archive:
        reference = archive['tensorflow_fp32']
        np.testing.assert_array_equal(labels, archive['labels'])
    metrics = {**numerical_metrics(reference, output),
               **classification_metrics(output, labels, task)}
    metrics['top1_agreement'] = float(np.mean(reference.argmax(1) == output.argmax(1)))
    metrics['allclose_fp32_tolerance'] = bool(np.allclose(output, reference,
                                                       atol=1e-5 if task == 'cv' else 1e-4, rtol=1e-4))
    if task == 'cv':
        a = np.argsort(-reference, axis=1, kind='stable')[:, :5]
        b = np.argsort(-output, axis=1, kind='stable')[:, :5]
        metrics['top5_overlap'] = float(np.mean([len(set(x)&set(y))/5 for x, y in zip(a, b)]))
    return metrics


def worker(task, variant, device, warmup, runs):
    folder = RESULTS / 'gpu' / task
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / f'{device}_{variant}.json'
    record = {'task': task, 'variant': variant, 'device': device,
              'measured_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'failed', 'warmup': warmup, 'runs': runs}
    try:
        dlls = configure_runtime()
        import onnxruntime as ort
        inputs, labels, manifest = load_inputs(task, 'evaluation')
        session = create_session(task, variant, device)
        record.update({'runtime_version': ort.__version__, 'package_versions': package_versions(),
                       'session_providers': session.get_providers(),
                       'provider_options': session.get_provider_options(), 'preloaded_dlls': dlls,
                       'threads': 1, 'inter_op_threads': 1, 'batch_size': 1,
                       'system': system_info(), 'canonical_input_sha256': manifest['tensor_hashes']['evaluation'],
                       'model_sha256': sha256(gpu_model_path(task, variant)),
                       'model_size_bytes': gpu_model_path(task, variant).stat().st_size,
                       'input_record': manifest['records']['evaluation'][0]})
        # Load/initialize/warm-up are outside the latency timer. The first input
        # is identical to the CPU lab. Output evaluation runs after timing.
        first = next(samples(inputs))
        for _ in range(warmup):
            session.run(None, first)
        latency = []
        for _ in range(runs):
            started = time.perf_counter_ns()
            output = session.run(None, first)[0]
            latency.append((time.perf_counter_ns() - started)/1e6)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite benchmark output')
        values = np.asarray(latency)
        record['benchmark'] = {'latency_samples_ms': latency, 'mean_ms': float(values.mean()),
                               'median_ms': float(np.median(values)),
                               'p95_ms': float(np.percentile(values, 95)), 'std_ms': float(values.std()),
                               'timing_scope': 'Synchronous session.run with CPU NumPy input/output, including H2D/D2H and Python/API overhead; excludes model load, preprocessing/tokenization; profiling disabled while timing'}
        output = []
        for index, canonical in enumerate(samples(inputs)):
            y = np.asarray(session.run(None, canonical)[0], dtype=np.float32)
            if y.shape != (1, 1000 if task == 'cv' else 2) or not np.isfinite(y).all():
                raise ValueError('Invalid evaluation output')
            output.append(y[0])
            if (index+1) % 25 == 0:
                print(f'{task}/{device}/{variant}: evaluated {index+1}/100', flush=True)
        output = np.stack(output)
        output_path = folder / f'output_{device}_{variant}.npy'
        np.save(output_path, output, allow_pickle=False)
        record['output_sha256'] = sha256(output_path)
        record['metrics_vs_original'] = original_comparison(task, output, labels)
        if variant == 'onnx_fp32' and not record['metrics_vs_original']['allclose_fp32_tolerance']:
            raise ValueError('FP32 output failed reference tolerance')
        del session
        gc.collect()
        # Profiling is deliberately a fresh session so it cannot inflate timing.
        profile_session = create_session(task, variant, device,
                                         profile_prefix=folder / f'profile_{device}_{variant}')
        for _ in range(2):
            profile_session.run(None, first)
        profile_path = __import__('pathlib').Path(profile_session.end_profiling())
        events = json.loads(profile_path.read_text(encoding='utf-8'))
        profile = summarize_profile(events)
        record['profile'] = {**profile, 'file': profile_path.relative_to(ROOT).as_posix(),
                             'sha256': sha256(profile_path)}
        if device == 'cuda' and profile['cuda_compute_nodes'] == 0:
            record['status'] = 'cpu_only_fallback'
            record['reason'] = 'Requested CUDA, but profiling found no CUDA compute nodes'
        else:
            record['status'] = 'ok'
        write_json(record_path, record)
        print(f"{task}/{device}/{variant}: {record['status']}, median {values.mean():.3f} mean / {np.median(values):.3f} median ms; providers {profile['unique_nodes_by_provider']}", flush=True)
    except Exception as exc:
        record.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        # Never publish partial timing as an accepted CUDA benchmark.
        record.pop('benchmark', None)
        write_json(record_path, record)
        traceback.print_exc()
        raise


def inventory():
    loaded = configure_runtime()
    import onnxruntime as ort
    command = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    record = {'nvidia_smi': result.stdout.strip(), 'runtime_version': ort.__version__,
              'available_providers': ort.get_available_providers(), 'system': system_info(),
              'package_versions': package_versions(), 'preloaded_dlls': loaded,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    write_json(RESULTS / 'gpu/environment.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('inventory', 'convert', 'run'), default='run')
    parser.add_argument('--task', choices=('cv', 'text'))
    parser.add_argument('--variant', choices=GPU_VARIANTS, default='onnx_fp32')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args()
    if args.action != 'inventory' and args.task is None:
        parser.error('--task is required')
    if min(args.warmup, args.runs) < 1:
        parser.error('Positive warmup/runs required')
    if args.action == 'inventory':
        inventory()
    elif args.action == 'convert':
        try:
            convert_fp16(args.task)
        except Exception as exc:
            write_json(RESULTS / 'gpu' / args.task / 'conversion_onnx_fp16.json',
                       {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
            raise
    else:
        worker(args.task, args.variant, args.device, args.warmup, args.runs)


if __name__ == '__main__':
    main()
