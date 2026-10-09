"""Run conversions first, then measurements sequentially; never parallel benchmarks.

Requires the legacy lab artifacts and successful M-LSD FP32 bridge preparation.
Failures are recorded and subsequent independent jobs still execute.
"""
import json
from pathlib import Path
import subprocess
import time

DAY=Path(__file__).resolve().parent
ROOT=DAY.parents[1]
OLD=ROOT/'.venv/Scripts/python.exe'
NEW=ROOT/'.venv-day09/Scripts/python.exe'


def main():
    jobs=[]
    mlsd=DAY/'experiments/03-mlsd/src'
    for variant in ('fp16','dynamic','static','decoded_fp32'):
        jobs.append((OLD,mlsd/'port_keras.py',['--variant',variant]))
    for variant in ('baseline','strict','mixed_mask1e4','mixed_mask1e2'):
        jobs.append((OLD,DAY/'experiments/02-distilbert-int8/src/run.py',
                     ['measure','--variant',variant,'--threads','4']))
    for task in ('cv','text'):
        for variant in ('onnx_fp32','onnx_dynamic','onnx_static','tflite_fp32','tflite_dynamic','tflite_static'):
            for runtime,exe in [('old',OLD),('new',NEW)]:
                jobs.append((exe,DAY/'experiments/01-reproduce/src/runtime_compare.py',
                             ['--task',task,'--variant',variant,'--runtime',runtime]))
    for variant in ('fp32','fp16','dynamic','static','decoded_fp32','official'):
        jobs.append((OLD,mlsd/'measure.py',['--variant',variant]))
    for variant in ('decoded_fp32','official'):
        jobs.append((NEW,mlsd/'measure.py',['--variant',variant,'--runtime','new']))
    folder=DAY/'experiments/01-reproduce/results/run-01'
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,(exe,script,args) in enumerate(jobs,1):
        name=f'{i:02d}_'+script.stem+'_'+'_'.join(args).replace('--','')
        print(f'[{i}/{len(jobs)}] {name}',flush=True)
        start=time.time()
        with (folder/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(exe),str(script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        records.append({'name':name,'command':[str(exe.relative_to(ROOT)),str(script.relative_to(ROOT)),*args],
                        'exit_code':result.returncode,'elapsed_seconds':time.time()-start})
        (folder/'execution.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        print(f'Exit {result.returncode}, elapsed {records[-1]["elapsed_seconds"]:.1f}s',flush=True)
    if any(r['exit_code'] for r in records):
        raise SystemExit('Some jobs failed; inspect recorded logs and evidence')


if __name__=='__main__': main()
