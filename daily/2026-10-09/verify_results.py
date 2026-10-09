"""Offline evidence checks: hashes, recomputed output metrics and timing statistics."""
import json
from pathlib import Path
import sys

import numpy as np

DAY=Path(__file__).resolve().parent; ROOT=DAY.parents[1]
sys.path.insert(0,str(ROOT/'quantization'))
from qlab.common import classification_metrics,numerical_metrics,sha256,write_json


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def close(expected,actual):
    if isinstance(expected,str):
        assert expected==str(actual), (expected,actual)
    else:
        assert np.isclose(expected,actual,rtol=1e-8,atol=1e-10), (expected,actual)
def metrics_check(expected,reference,output):
    for key,value in numerical_metrics(reference,output).items():
        if isinstance(value,float) and np.isinf(value):value='inf' if value>0 else '-inf'
        close(expected[key],value)
def timing(record):
    b=record['benchmark']; values=np.asarray(b['samples_ms'])
    assert len(values)==b['runs'] and len(values)==200
    assert np.isfinite(values).all() and (values>0).all()
    close(b['median_ms'],np.median(values));close(b['p95_ms'],np.percentile(values,95))


def main():
    counts={'runtime':0,'distilbert':0,'mlsd':0,'hashes':0}
    references={}; labels={}
    for task in ('cv','text'):
        with np.load(ROOT/f'quantization/results/{task}/outputs.npz',allow_pickle=False) as a:
            references[task]=a['tensorflow_fp32'].copy();labels[task]=a['labels'].copy()
    folder=DAY/'experiments/01-reproduce/results/run-01'
    for p in folder.glob('*.json'):
        r=read(p)
        if not isinstance(r,dict) or 'metrics' not in r:continue
        output_file=folder/f"output_{r['name']}.npy"
        assert sha256(output_file)==r['output_sha256'];counts['hashes']+=1
        output=np.load(output_file,allow_pickle=False);task=r['task']
        assert output.shape==references[task].shape
        metrics_check(r['metrics'],references[task],output)
        for key,value in classification_metrics(output,labels[task],task).items():close(r['metrics'][key],value)
        assert sha256(ROOT/f"quantization/data/{task}/evaluation.npz")==r['input_sha256']
        suffix='.onnx' if r['variant'].startswith('onnx') else '.tflite'
        assert sha256(ROOT/f"quantization/artifacts/{task}/{r['variant']}{suffix}")==r['model_sha256']
        timing(r);counts['runtime']+=1
    folder=DAY/'experiments/02-distilbert-int8/results/run-01'
    for p in folder.glob('measure_*.json'):
        r=read(p); output_file=folder/f"output_{r['variant']}_t{r['threads']}.npy"
        assert sha256(output_file)==r['output_sha256'];counts['hashes']+=1
        output=np.load(output_file,allow_pickle=False)
        metrics_check(r['metrics'],references['text'],output)
        for key,value in classification_metrics(output,labels['text'],'text').items():close(r['metrics'][key],value)
        timing(r);counts['distilbert']+=1
    folder=DAY/'experiments/03-mlsd/results/run-01'
    reference=np.load(folder/'torch_reference.npy',allow_pickle=False)
    for p in folder.glob('measure_*.json'):
        r=read(p);timing(r)
        if 'metrics_vs_torch' in r:
            output_file=folder/f"output_{r['variant']}_{'new' if r['runtime']=='2.2.0' else 'old'}_t{r['threads']}.npy"
            assert sha256(output_file)==r['output_sha256'];counts['hashes']+=1
            output=np.load(output_file,allow_pickle=False)
            metrics_check(r['metrics_vs_torch'],reference,output)
            metrics_check(r['center_logit_metrics'],reference[:,:,:,0],output[:,:,:,0])
            metrics_check(r['displacement_metrics'],reference[:,:,:,1:5],output[:,:,:,1:5])
        counts['mlsd']+=1
    manifest=read(folder/'torch_reference.json')
    for split in ('calibration','evaluation'):
        for row in manifest['records'][split]:
            assert sha256(ROOT/'quantization'/row['path'])==row['sha256']
    assert not(set(r['sha256'] for r in manifest['records']['calibration']) & set(r['sha256'] for r in manifest['records']['evaluation']))
    assert sha256(folder/'torch_reference.npy')==manifest['reference_sha256']
    for name,r in read(DAY/'experiments/03-mlsd/sources.lock.json').items():
        assert sha256(ROOT/'.cache/mlsd'/name)==r['sha256'];counts['hashes']+=1
    execution=read(DAY/'experiments/01-reproduce/results/run-01/execution.json')
    assert len(execution)==40 and all(r['exit_code']==0 for r in execution)
    assert counts['runtime']==24 and counts['distilbert']==4 and counts['mlsd']==8,counts
    counts['thread_controls']=0
    folder=DAY/'experiments/01-reproduce/results/run-02-threadcheck'
    for p in folder.glob('text_*.json'):
        r=read(p);output_file=folder/f"output_{r['name']}.npy"
        assert sha256(output_file)==r['output_sha256']
        metrics_check(r['metrics'],references['text'],np.load(output_file,allow_pickle=False))
        timing(r);counts['thread_controls']+=1
    assert counts['thread_controls']==2
    box=read(DAY/'experiments/04-box-postprocess/results/run-01/demo_boxes.json')
    assert box['geometry_cases_passed']==5 and len(box['samples_ms'])==200
    close(box['median_ms'],np.median(box['samples_ms']))
    assert sha256(ROOT/box['input_csv'])==box['input_sha256']
    counts['box_candidates']=len(box['boxes'])
    write_json(DAY/'experiments/01-reproduce/results/run-01/verification.json',
               {'status':'passed','checks':counts,'execution_jobs':len(execution),
                'scope':'offline hashes, labels/metrics and 200-sample timing recomputation; not a new benchmark'})
    print('Offline verification passed:',counts)


if __name__=='__main__':main()
