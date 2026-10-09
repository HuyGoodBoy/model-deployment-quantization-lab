"""Sequential focused checks for observed latency variation and box smoke input."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

DAY=Path(__file__).resolve().parent;ROOT=DAY.parents[1]


def main():
    folder=DAY/'experiments/01-reproduce/results/run-02-threadcheck';folder.mkdir(parents=True,exist_ok=True)
    base=[str(ROOT/'.venv/Scripts/python.exe'),str(DAY/'experiments/01-reproduce/src/runtime_compare.py'),
          '--task','text','--variant','tflite_static','--runtime','old','--threads','4','--run-id','run-02-threadcheck']
    for name,extra in [('environment_only',[]),('explicit_tf',['--initialize-tf-threads'])]:
        print('Thread check:',name,flush=True)
        with (folder/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(base+extra,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for runtime,env in [('old','.venv'),('new','.venv-day09')]:
        subprocess.run([str(ROOT/env/'Scripts/python.exe'),str(DAY/'experiments/01-reproduce/src/delegate_audit.py'),
                        '--runtime',runtime],cwd=ROOT,check=True)
    java=shutil.which('java')
    if not java:
        java=str(next((ROOT/'.cache/android-tools/java').glob('*/bin/java.exe')))
    boxes=DAY/'experiments/04-box-postprocess';out=boxes/'results/run-01';out.mkdir(parents=True,exist_ok=True)
    subprocess.run([java,'-cp',str(boxes/'artifacts/classes'),'BoxPostProcessor',
                    str(DAY/'experiments/03-mlsd/results/run-01/lines_fp32_old_t4.csv'),
                    str(out/'demo_boxes.json')],cwd=ROOT,check=True)
    path=out/'demo_boxes.json';record=json.loads(path.read_text(encoding='utf-8'))
    source=DAY/'experiments/03-mlsd/results/run-01/lines_fp32_old_t4.csv'
    record.update(geometry_cases_passed=5,input_csv=str(source.relative_to(ROOT)),
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  equivalence='not a complete NAVER port; input demo lines are unmerged')
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
