"""Verify the GPU/paired-CPU evidence without CUDA, models, or network access."""
import io
import hashlib

import numpy as np

from .common import DATA, RESULTS, read_json, sha256
from .gpu_experiment import original_comparison
from .gpu_runtime import GPU_VARIANTS, summarize_profile
from .verify import close_dict


def main():
    count = 0
    for task in ('cv', 'text'):
        folder = RESULTS / 'gpu' / task
        manifest = read_json(DATA / task / 'manifest.json')
        with np.load(RESULTS / task / 'outputs.npz', allow_pickle=False) as source:
            labels = source['labels'].copy()
        outputs = {}
        for variant in GPU_VARIANTS:
            model_hash = None
            for device in ('cpu', 'cuda'):
                path = folder / f'{device}_{variant}.json'
                if not path.is_file():
                    raise ValueError(f'Missing experiment status: {task}/{device}/{variant}')
                record = read_json(path)
                if record['status'] != 'ok':
                    if 'benchmark' in record and record['status'] == 'failed':
                        raise ValueError('Failed experiment has accepted timing')
                    continue
                if record['canonical_input_sha256'] != manifest['tensor_hashes']['evaluation']:
                    raise ValueError('Input hash mismatch')
                output_path = folder / f'output_{device}_{variant}.npy'
                if sha256(output_path) != record['output_sha256']:
                    raise ValueError('Output hash mismatch')
                output = np.load(output_path, allow_pickle=False)
                close_dict(original_comparison(task, output, labels), record['metrics_vs_original'])
                if variant == 'onnx_fp32' and not record['metrics_vs_original']['allclose_fp32_tolerance']:
                    raise ValueError('FP32 baseline tolerance failed')
                values = np.asarray(record['benchmark']['latency_samples_ms'])
                if len(values) != record['runs'] or not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError('Invalid latency samples')
                close_dict({'mean_ms': float(values.mean()), 'median_ms': float(np.median(values)),
                            'p95_ms': float(np.percentile(values, 95)), 'std_ms': float(values.std())},
                           record['benchmark'])
                profile_path = RESULTS.parent / record['profile']['file']
                if sha256(profile_path) != record['profile']['sha256']:
                    raise ValueError('Profile hash mismatch')
                profile = summarize_profile(read_json(profile_path))
                if profile != {k: v for k, v in record['profile'].items() if k not in ('file', 'sha256')}:
                    raise ValueError('Provider profiling summary mismatch')
                if device == 'cuda' and (profile['cuda_compute_nodes'] == 0 or
                                         'CUDAExecutionProvider' not in record['session_providers']):
                    raise ValueError('CUDA result has no evidence of GPU computation')
                if device == 'cpu' and record['session_providers'] != ['CPUExecutionProvider']:
                    raise ValueError('CPU baseline used a different provider')
                if model_hash is not None and model_hash != record['model_sha256']:
                    raise ValueError('CPU/CUDA model hash mismatch')
                model_hash = record['model_sha256']
                outputs[(device, variant)] = output
                count += 1
        for variant in GPU_VARIANTS:
            if ('cuda', variant) in outputs and ('cpu', variant) in outputs:
                # Both sides were checked against the same labels and reference.
                np.testing.assert_equal(outputs['cpu', variant].shape, outputs['cuda', variant].shape)
        backup = folder/'first_pass_fp32'
        if backup.exists():
            for device in ('cpu','cuda'):
                row = read_json(backup/f'{device}_onnx_fp32.json')
                current = read_json(folder/f'{device}_onnx_fp32.json')
                for key in ('model_sha256','canonical_input_sha256','runtime_version','warmup','runs'):
                    if row[key] != current[key]:
                        raise ValueError('Repeat pair changed experimental conditions')
                path = backup/f'output_{device}_onnx_fp32.npy'
                if sha256(path) != row['output_sha256']:
                    raise ValueError('Initial pass output hash mismatch')
                close_dict(original_comparison(task,np.load(path,allow_pickle=False),labels),row['metrics_vs_original'])
                values = np.asarray(row['benchmark']['latency_samples_ms'])
                if len(values)!=row['runs'] or (values<=0).any() or not np.isfinite(values).all():
                    raise ValueError('Invalid initial pass latency')
                close_dict({'mean_ms':float(values.mean()),'median_ms':float(np.median(values)),
                            'p95_ms':float(np.percentile(values,95)),'std_ms':float(values.std())},row['benchmark'])
                profile = RESULTS.parent/row['profile']['file']
                if sha256(profile)!=row['profile']['sha256']:
                    raise ValueError('Initial pass profile hash mismatch')
                if device=='cuda' and summarize_profile(read_json(profile))['cuda_compute_nodes']==0:
                    raise ValueError('Initial CUDA pass lacks compute evidence')
                count += 1
        print(f'PASS GPU evidence verification: {task}', flush=True)
    print(f'Verified {count} successful measurements offline.', flush=True)


if __name__ == '__main__':
    main()
