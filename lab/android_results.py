"""Kiem tra va tong hop output/latency lay tu Android, giu ket qua PC rieng."""
import argparse
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .common import ARTIFACTS, RESULTS, ROOT, numerical_metrics, read_json, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_output(directory, record):
    path = (directory / record['file']).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Output path escapes the result directory.')
    if sha256(path) != record['sha256']:
        raise ValueError(f'Output checksum mismatch: {path.name}')
    output = np.fromfile(path, dtype='<f4')
    if output.shape != (1000,) or not np.isfinite(output).all():
        raise ValueError('Output must have 1000 finite FP32 values.')
    if (output.min() < -1e-6 or output.max() > 1.000001
            or not np.isclose(output.sum(dtype=np.float64), 1, atol=1e-4, rtol=0)):
        raise ValueError('Output is not a softmax probability vector.')
    return output


def timing_statistics(benchmark, runs):
    samples = np.asarray(benchmark['latency_samples_ms'], dtype=np.float64)
    if samples.shape != (runs,) or not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError('Latency samples do not match the run configuration.')
    computed = {'mean_ms': float(samples.mean()), 'median_ms': float(np.median(samples)),
                'p95_ms': float(np.percentile(samples, 95)),
                'images_per_second': float(1000 / samples.mean())}
    for key, value in computed.items():
        if not np.isclose(value, benchmark[key], rtol=1e-9, atol=1e-9):
            raise ValueError(f'Latency summary mismatch: {key}')
    return computed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Thu muc chua android_run.json va cac file .f32')
    parser.add_argument('--output', type=Path, help='Thu muc bao cao; mac dinh results/android/<run hash>')
    args = parser.parse_args()
    directory = args.directory.resolve()
    source = directory / 'android_run.json'
    run = read_json(source)
    packed = read_json(ROOT / 'android/app/src/main/assets/lab/bundle.json')
    if run['schema_version'] != 1 or run['bundle'] != packed:
        raise ValueError('Android bundle does not match the locally packaged model/inputs.')
    if packed['original_model_sha256'] != sha256(ARTIFACTS / 'mobilenetv2.keras'):
        raise ValueError('Original model changed after packaging.')
    if packed['reference_outputs_npz_sha256'] != sha256(RESULTS / 'outputs.npz'):
        raise ValueError('Reference output changed after packaging.')
    for runtime in ('onnx', 'tflite'):
        if sha256(ARTIFACTS / packed['models'][runtime]['asset']) != packed['models'][runtime]['sha256']:
            raise ValueError('Exported model changed after packaging.')
    if set(run['benchmarks']) != {'onnx', 'tflite'}:
        raise ValueError('Android run is incomplete: both runtimes are required.')
    if run['batch_size'] != 1 or min(run['threads'], run['runs'], run['warmup']) < 1:
        raise ValueError('Invalid Android benchmark configuration.')
    with np.load(RESULTS / 'outputs.npz', allow_pickle=False) as saved:
        reference = saved['tensorflow'].copy()
    records = packed['records']
    if reference.shape != (len(records), 1000):
        raise ValueError('Reference dataset shape mismatch.')
    comparisons, per_image, benchmarks = {}, [], {}
    for runtime, benchmark in run['benchmarks'].items():
        files = {item['filename']: item for item in benchmark['outputs']}
        if len(files) != len(records) or len(benchmark['outputs']) != len(records):
            raise ValueError('Output filenames/count do not match the dataset.')
        if benchmark['input_image'] != records[0]['filename']:
            raise ValueError('Benchmark does not use the same first image as the PC run.')
        candidate = np.stack([load_output(directory, files[item['filename']]) for item in records])
        metrics = numerical_metrics(reference, candidate)
        metrics['top1_agreement'] = float(np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1)))
        original_top5 = np.argsort(-reference, axis=1)[:, :5]
        new_top5 = np.argsort(-candidate, axis=1)[:, :5]
        metrics['top5_overlap'] = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(original_top5, new_top5)]))
        metrics['allclose'] = bool(np.allclose(candidate, reference, atol=packed['atol'], rtol=packed['rtol']))
        comparisons[runtime] = metrics
        benchmarks[runtime] = timing_statistics(benchmark, run['runs'])
        for index, record in enumerate(records):
            per_image.append({'filename': record['filename'], 'runtime': runtime,
                              **numerical_metrics(reference[index], candidate[index]),
                              'reference_top1': int(reference[index].argmax()),
                              'candidate_top1': int(candidate[index].argmax())})
    target = args.output or RESULTS / 'android' / sha256(source)[:12]
    target.mkdir(parents=True, exist_ok=True)
    summary = {'source': str(source), 'system': run['system'], 'num_images': len(records),
               'threads': run['threads'], 'warmup': run['warmup'], 'runs': run['runs'],
               'timing_scope': run['timing_scope'], 'runtime_order': run['runtime_order'],
               'comparisons': comparisons, 'benchmarks': benchmarks, 'per_image': per_image,
               'accuracy': None, 'reference': 'TensorFlow FP32 on PC',
               'atol': packed['atol'], 'rtol': packed['rtol']}
    if 'started_at_unix_ms' in run:
        summary['measured_at_utc7'] = datetime.fromtimestamp(
            run['started_at_unix_ms'] / 1000, timezone(timedelta(hours=7))).isoformat()
    write_json(target / 'summary.json', summary)
    system = run['system']
    lines = ['# Kết quả MobileNetV2 trên Android', '',
             f"Thời điểm đo (UTC+7): {summary.get('measured_at_utc7', 'không có trong metadata')}.", '',
             f"Thiết bị báo trong runtime: {system['manufacturer']} {system['model']}; "
             f"Android {system['android_release']} (API {system['android_api']}); ABI {system['abis']}.", '',
             f"CPU, FP32, batch=1, {run['threads']} thread, warm-up={run['warmup']}, runs={run['runs']}.",
             f"Thứ tự runtime: {run['runtime_order']}. Phạm vi timer: {run['timing_scope']}.", '',
             '## Sai khác output so với TensorFlow gốc trên PC', '',
             f"So sánh {len(records)} ảnh × 1.000 lớp; không có nhãn thật nên chưa tính accuracy.", '',
             '| Metric | ONNX | TFLite |', '|---|---:|---:|']
    labels = [('MAE','mae'),('RMSE','rmse'),('Max abs','max_abs_error'),
              ('Relative L2','relative_l2'),('Cosine','cosine_similarity'),('SNR dB','snr_db'),
              ('Top-1 agreement','top1_agreement'),('Top-5 overlap','top5_overlap'),('Allclose','allclose')]
    for label, key in labels:
        def display(value):
            if isinstance(value, bool): return str(value)
            return f'{value:.17g}' if key == 'cosine_similarity' else f'{value:.6g}'
        lines.append(f'| {label} | {display(comparisons["onnx"][key])} | {display(comparisons["tflite"][key])} |')
    lines += ['', '## Tốc độ trên Android', '', '| Runtime | Mean ms | Median ms | P95 ms |', '|---|---:|---:|---:|']
    for runtime, stats in benchmarks.items():
        lines.append(f"| {runtime} | {stats['mean_ms']:.3f} | {stats['median_ms']:.3f} | {stats['p95_ms']:.3f} |")
    lines += ['', '## Cách diễn giải và giới hạn', '',
              '- Đối chiếu model thiết bị/Android với cấu hình test Firebase; ghi rõ physical device hay emulator.',
              '- Chỉ hai ảnh mẫu; agreement không phải accuracy. Dữ liệu latency là các lần lặp của ảnh đầu tiên.',
              '- ONNX Runtime Android 1.20.0/PC 1.20.1 và TFLite Android 2.15.0/PC 2.15.1 khác phiên bản; timer Java khác timer Python.',
              '- Hai runtime chạy nối tiếp trong một test process; thứ tự, nhiệt độ và tải nền có thể ảnh hưởng kết quả.',
              '- Một thread intra-op không có nghĩa toàn bộ hệ thống hoặc runtime chỉ có một thread.',
              '- Xem metadata runtime và nhiệt độ/thermal status trước và sau đo trong android_run.json.',
              '- Các số liệu PC ngày 07/10/2026 được giữ riêng; không suy ra mọi điện thoại từ một thiết bị.', '']
    (target / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Android report saved: {target / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
