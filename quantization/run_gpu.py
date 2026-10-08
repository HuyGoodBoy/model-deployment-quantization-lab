"""Sequential GPU/CPU experiment; preserve the earlier CPU results."""
import argparse
import subprocess
import sys

from qlab.common import RESULTS, ROOT, read_json, write_json
from qlab.gpu_runtime import GPU_VARIANTS


def launch(name, args, required=True):
    path = ROOT / 'logs/gpu' / f'{name}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    print('START ' + name, flush=True)
    with path.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    print(('PASS ' if result.returncode == 0 else 'FAILED ') + name, flush=True)
    if required and result.returncode:
        raise RuntimeError(f'{name} failed; see {path.relative_to(ROOT)}')
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cv', 'text', 'all'), default='all')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args()
    launch('inventory', ['-m', 'qlab.gpu_experiment', '--action', 'inventory'])
    for task in ('cv', 'text') if args.task == 'all' else (args.task,):
        fp16 = launch(task+'_convert_fp16', ['-m', 'qlab.gpu_experiment', '--action', 'convert',
                                            '--task', task], required=False)
        for variant in GPU_VARIANTS:
            if variant == 'onnx_fp16' and not fp16:
                error = read_json(RESULTS / 'gpu' / task / 'conversion_onnx_fp16.json')
                for device in ('cpu', 'cuda'):
                    write_json(RESULTS / 'gpu' / task / f'{device}_{variant}.json',
                               {'status': 'conversion_failed', 'task': task,
                                'variant': variant, 'device': device, 'conversion_error': error})
                continue
            for device in ('cpu', 'cuda'):
                launch(f'{task}_{device}_{variant}', ['-m', 'qlab.gpu_experiment', '--task', task,
                       '--variant', variant, '--device', device, '--warmup', str(args.warmup),
                       '--runs', str(args.runs)], required=variant == 'onnx_fp32')
    print('Measurements complete. Generate the daily report with python -m qlab.daily_report.', flush=True)


if __name__ == '__main__':
    main()
