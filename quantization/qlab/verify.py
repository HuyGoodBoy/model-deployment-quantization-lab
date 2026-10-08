"""Recompute metrics and timing summaries from packaged raw results, offline."""
import hashlib
import io

import numpy as np

from .common import ARTIFACTS, DATA, RESULTS, VARIANTS, classification_metrics, numerical_metrics, read_json, safe_json
from .prepare import normalize_text


def close_dict(actual,expected):
    for key,value in actual.items():
        if isinstance(value,(float,int)) and not isinstance(value,bool) and np.isfinite(value):
            if not np.isclose(value,expected[key],rtol=1e-9,atol=1e-10): raise ValueError(f'Metric mismatch: {key}')
        elif safe_json(value)!=expected[key]: raise ValueError(f'Metric mismatch: {key}')


def main():
    for task in ('cv','text'):
        manifest=read_json(DATA/task/'manifest.json')
        if task=='cv':
            a={r['source_sha256'] for r in manifest['records']['calibration']}
            b={r['source_sha256'] for r in manifest['records']['evaluation']}
        else:
            a={normalize_text(r['sentence']) for r in manifest['records']['calibration']}
            b={normalize_text(r['sentence']) for r in manifest['records']['evaluation']}
        if a&b: raise ValueError('Calibration/evaluation leakage')
        summary=read_json(RESULTS/task/'evaluation_summary.json')
        model=read_json(ARTIFACTS/task/'model_info.json')
        diagnostic=RESULTS/task/'diagnostic_tflite_static.json'
        if diagnostic.is_file() and read_json(diagnostic)['model_sha256']!=read_json(
                RESULTS/task/'conversion_tflite_static.json')['sha256']:
            raise ValueError('Diagnostic/model mismatch')
        with np.load(RESULTS/task/'outputs.npz',allow_pickle=False) as outputs:
            reference=outputs['tensorflow_fp32']
            labels=outputs['labels']
            np.testing.assert_array_equal(labels,[r['label'] for r in manifest['records']['evaluation']])
            for variant,record in summary['results'].items():
                if record['status']!='ok': continue
                candidate=outputs[variant]
                metrics={**numerical_metrics(reference,candidate),**classification_metrics(candidate,labels,task)}
                close_dict(metrics,record['metrics'])
                npy=io.BytesIO();np.save(npy,candidate,allow_pickle=False)
                if hashlib.sha256(npy.getvalue()).hexdigest()!=record['output_sha256']:
                    raise ValueError('Raw output checksum mismatch')
                expected_model=model['original_sha256'] if variant=='tensorflow_fp32' else read_json(
                    RESULTS/task/f'conversion_{variant}.json')['sha256']
                if record['model_sha256']!=expected_model: raise ValueError('Model provenance mismatch')
                timing=read_json(RESULTS/task/f'benchmark_{variant}.json')
                if timing['model_sha256']!=expected_model: raise ValueError('Benchmark/model mismatch')
                if timing['canonical_input_sha256']!=manifest['tensor_hashes']['evaluation']:
                    raise ValueError('Benchmark/input mismatch')
                samples=np.asarray(timing['latency_samples_ms'])
                if len(samples)!=timing['runs'] or not np.isfinite(samples).all() or (samples<=0).any():
                    raise ValueError('Invalid latency samples')
                close_dict({'mean_ms':float(samples.mean()),'median_ms':float(np.median(samples)),
                            'p95_ms':float(np.percentile(samples,95))},timing)
        print(f'PASS offline evidence verification: {task}',flush=True)


if __name__=='__main__': main()
