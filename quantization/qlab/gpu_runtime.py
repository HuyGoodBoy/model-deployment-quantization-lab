"""Explicit ORT providers, local NVIDIA DLLs, and evidence of node assignment."""
from collections import Counter, defaultdict
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import site

from .common import ARTIFACTS, ROOT, configure_cpu

GPU_VARIANTS = ('onnx_fp32', 'onnx_dynamic', 'onnx_static', 'onnx_fp16')
DLL_HANDLES = []


def gpu_model_path(task, variant):
    if variant not in GPU_VARIANTS:
        raise ValueError('Unknown GPU experiment variant')
    folder = ARTIFACTS / ('gpu/' + task if variant == 'onnx_fp16' else task)
    return folder / f'{variant}.onnx'


def configure_runtime():
    configure_cpu(1)
    # configure_cpu hides GPUs for the old TensorFlow lab. This separate lab
    # explicitly exposes device 0; provider choice controls CPU/GPU execution.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if os.name != 'nt':
        return []
    bins = sorted({directory for root in site.getsitepackages()
                   for directory in (Path(root) / 'nvidia').glob('*/bin')})
    for directory in bins:
        DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    os.environ['PATH'] = os.pathsep.join(map(str, bins)) + os.pathsep + os.environ['PATH']
    loaded = []
    # ORT 1.20 predates preload_dlls(). Dependencies remain local to the venv.
    for name in ('nvJitLink_120_0.dll', 'cudart64_12.dll', 'cublasLt64_12.dll',
                 'cublas64_12.dll', 'cufft64_11.dll', 'cudnn64_9.dll'):
        matches = [directory / name for directory in bins if (directory / name).is_file()]
        if not matches:
            if name.startswith('nvJitLink'):
                continue
            raise FileNotFoundError(f'Missing NVIDIA dependency: {name}')
        DLL_HANDLES.append(ctypes.WinDLL(str(matches[0])))
        loaded.append(name)
    return loaded


def create_session(task, variant, device, profile_prefix=None):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    if device == 'cuda':
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA EP is unavailable; refusing CPU-only GPU result')
        providers = [('CUDAExecutionProvider', {'device_id': '0', 'use_tf32': '0',
                                               'do_copy_in_default_stream': '1'}),
                     'CPUExecutionProvider']
    elif device == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError('Expected cpu or cuda')
    session = ort.InferenceSession(str(gpu_model_path(task, variant)),
                                   sess_options=options, providers=providers)
    # Prevent Python from retrying an inference with a different EP on failure.
    # CPU graph partitions are allowed and explicitly measured by the profile.
    session.disable_fallback()
    if device == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('CUDA initialization failed; refusing CPU-only GPU result')
    if device == 'cpu' and session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Unexpected CPU provider configuration')
    expected = {'image'} if task == 'cv' else {'input_ids', 'attention_mask'}
    if {item.name for item in session.get_inputs()} != expected:
        raise ValueError('Input name mismatch')
    return session


def summarize_profile(events):
    # One record per unique scheduled node, not a count multiplied by runs.
    unique = {}
    event_counts = Counter()
    for event in events:
        args = event.get('args', {})
        provider = args.get('provider')
        name = event.get('name', '')
        if event.get('cat') != 'Node' or not provider or not name.endswith('_kernel_time'):
            continue
        key = (provider, name)
        unique[key] = {'provider': provider, 'name': name, 'op': args.get('op_name', 'unknown'),
                       'input_type_shape': args.get('input_type_shape')}
        event_counts[provider] += 1
    providers = Counter()
    operators = defaultdict(Counter)
    for record in unique.values():
        providers[record['provider']] += 1
        operators[record['provider']][record['op']] += 1
    gpu_compute = [row for row in unique.values()
                   if row['provider'] == 'CUDAExecutionProvider'
                   and not row['op'].lower().startswith('memcpy')]
    return {'unique_nodes_by_provider': dict(providers),
            'operator_counts_by_provider': {p: dict(ops) for p, ops in operators.items()},
            'kernel_events_by_provider': dict(event_counts),
            'cuda_compute_nodes': len(gpu_compute),
            'cpu_nodes': [row for row in unique.values() if row['provider'] == 'CPUExecutionProvider'],
            'cuda_compute_operators': sorted({row['op'] for row in gpu_compute}),
            'cuda_compute_examples': gpu_compute[:12],
            'note': 'Separate profiling session after timing; counts show provider assignment, not kernel-only latency or INT8 arithmetic coverage'}


def package_versions():
    result = {}
    for name in ('onnxruntime-gpu', 'onnx', 'numpy', 'nvidia-cuda-runtime-cu12',
                 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
                 'nvidia-nvjitlink-cu12'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
