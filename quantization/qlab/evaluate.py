"""Infer all labelled evaluation samples for one variant, or aggregate outputs."""
import argparse

import numpy as np

from .common import (ARTIFACTS, RESULTS, VARIANTS, classification_metrics, load_inputs, model_path,
                     numerical_metrics, read_json, samples, sha256, write_json)
from .runtime import create_runner


def worker(task,variant):
    inputs,labels,manifest=load_inputs(task,'evaluation')
    run,info=create_runner(task,variant)
    result=[]
    for i,inputs_one in enumerate(samples(inputs)):
        y=np.asarray(run(inputs_one),dtype=np.float32)
        expected=(1,1000 if task=='cv' else 2)
        if y.shape!=expected or not np.isfinite(y).all(): raise ValueError('Invalid model output')
        result.append(y[0])
        if (i+1)%25==0: print(f'{task}/{variant}: {i+1}/{len(labels)} evaluated',flush=True)
    output=np.stack(result)
    folder=RESULTS/task
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'output_{variant}.npy'
    np.save(path,output,allow_pickle=False)
    write_json(folder/f'evaluation_{variant}.json',{'status':'ok','output_sha256':sha256(path),
        'model_sha256':sha256(model_path(task,variant)),
        'model_size_bytes':model_path(task,variant).stat().st_size,
        'input_sha256':manifest['tensor_hashes']['evaluation'],'runtime':info,
        'classification':classification_metrics(output,labels,task)})


def aggregate(task):
    _,labels,manifest=load_inputs(task,'evaluation')
    folder=RESULTS/task
    reference=np.load(folder/'output_tensorflow_fp32.npy',allow_pickle=False)
    results={};archived={'labels':labels}
    for variant in VARIANTS:
        path=folder/f'output_{variant}.npy'
        evaluation=folder/f'evaluation_{variant}.json'
        if not path.exists() or not evaluation.exists():
            results[variant]={'status':'failed or not run'}
            continue
        meta=read_json(evaluation)
        if sha256(path)!=meta['output_sha256'] or meta['input_sha256']!=manifest['tensor_hashes']['evaluation']:
            raise ValueError('Evaluation provenance mismatch')
        candidate=np.load(path,allow_pickle=False)
        metrics=numerical_metrics(reference,candidate)
        metrics.update(classification_metrics(candidate,labels,task))
        metrics['top1_agreement']=float(np.mean(reference.argmax(axis=1)==candidate.argmax(axis=1)))
        atol=1e-5 if task=='cv' else 1e-4
        metrics['allclose_fp32_tolerance']=bool(np.allclose(candidate,reference,atol=atol,rtol=1e-4))
        if task=='cv':
            a=np.argsort(-reference,axis=1,kind='stable')[:,:5]
            b=np.argsort(-candidate,axis=1,kind='stable')[:,:5]
            metrics['top5_overlap']=float(np.mean([len(set(x)&set(y))/5 for x,y in zip(a,b)]))
        results[variant]={'status':'ok','metrics':metrics,'runtime':meta['runtime'],
                          'model_sha256':meta['model_sha256'],'output_sha256':meta['output_sha256']}
        archived[variant]=candidate
    np.savez_compressed(folder/'outputs.npz',**archived)
    write_json(folder/'evaluation_summary.json',{'task':task,'num_examples':len(labels),
        'domain':'softmax probabilities' if task=='cv' else 'logits',
        'results':results,'atol_fp32':1e-5 if task=='cv' else 1e-4,'rtol_fp32':1e-4})
    # A bad FP32 export must be fixed before interpreting a quantization experiment.
    for variant in ('onnx_fp32','tflite_fp32'):
        if results[variant]['status']!='ok' or not results[variant]['metrics']['allclose_fp32_tolerance']:
            raise ValueError(f'FP32 baseline validation failed: {task}/{variant}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=('cv','text'))
    parser.add_argument('--variant',choices=VARIANTS)
    args=parser.parse_args()
    worker(args.task,args.variant) if args.variant else aggregate(args.task)


if __name__=='__main__': main()
