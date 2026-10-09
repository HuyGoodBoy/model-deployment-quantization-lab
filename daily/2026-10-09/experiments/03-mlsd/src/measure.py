"""Separate raw-graph accuracy from comparable decoded-graph latency."""
import argparse
import csv
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[1]; REPO=TASK.parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from qlab.common import numerical_metrics,sha256,system_info,write_json


def decode(raw):
    center=raw[0,:,:,0]
    heat=1/(1+np.exp(-np.clip(center,-80,80)))
    padded=np.pad(heat,1,constant_values=0)
    maximum=np.maximum.reduce([padded[y:y+256,x:x+256] for y in range(3) for x in range(3)])
    heat=np.where(heat==maximum,heat,0)
    indices=np.argsort(-heat.ravel(),kind='stable')[:200]
    return np.stack([indices//256,indices%256],axis=-1)[None],heat.ravel()[indices][None],raw[:,:,:,1:5]


def lines(decoded):
    points,scores,vmap=decoded; result=[]
    for (y,x),score in zip(points[0],scores[0]):
        delta=vmap[0,int(y),int(x)]
        if score<=.1 or np.linalg.norm(delta[:2]-delta[2:])<=20: continue
        result.append([2*(x+delta[0]),2*(y+delta[1]),2*(x+delta[2]),2*(y+delta[3]),float(score)])
    return np.asarray(result,np.float32).reshape(-1,5)


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS=str(args.threads),
                      TF_NUM_INTRAOP_THREADS=str(args.threads),TF_NUM_INTEROP_THREADS='1')
    if args.runtime=='new':
        from ai_edge_litert.interpreter import Interpreter
        version=importlib.metadata.version('ai-edge-litert')
    else:
        import tensorflow as tf
        Interpreter=tf.lite.Interpreter; version=tf.__version__
    path=REPO/'.cache/mlsd/official_tiny_fp32.tflite' if args.variant=='official' else TASK/'artifacts'/f'{args.variant}.tflite'
    interpreter=Interpreter(model_path=str(path),num_threads=args.threads)
    interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0]; outs=interpreter.get_output_details()
    def run(x):
        if x.dtype!=inp['dtype']:
            scale,zero=inp['quantization']; lim=np.iinfo(inp['dtype'])
            x=np.clip(np.rint(x.astype(np.float64)/scale)+zero,lim.min,lim.max).astype(inp['dtype'])
        interpreter.set_tensor(inp['index'],x); interpreter.invoke(); values=[]
        for d in outs:
            y=interpreter.get_tensor(d['index'])
            if np.issubdtype(y.dtype,np.integer) and d['quantization'][0]>0:
                scale,zero=d['quantization']; y=(y.astype(np.float32)-zero)*scale
            values.append(y)
        return values
    canonical=np.load(TASK/'artifacts/inputs.npz',allow_pickle=False)
    first=canonical['demo']
    folder=TASK/'results'/args.run_id; folder.mkdir(parents=True,exist_ok=True)
    name=f'{args.variant}_{args.runtime}_t{args.threads}'
    record={'status':'ok','variant':args.variant,'runtime':version,'threads':args.threads,
            'model_sha256':sha256(path),'size_bytes':path.stat().st_size,
            'input_sha256':sha256(TASK/'artifacts/inputs.npz'),'system':system_info(),
            'output_contract':[{'name':d['name'],'shape':d['shape'].tolist(),'dtype':np.dtype(d['dtype']).name,
                                'quantization':list(d['quantization'])} for d in outs]}
    tensor_details=interpreter.get_tensor_details()
    record['tensor_dtypes']={name:sum(np.dtype(d['dtype']).name==name for d in tensor_details)
                            for name in ('float32','float16','int8','int32')}
    if args.variant not in ('official','decoded_fp32'):
        output=np.concatenate([run(x[None])[0] for x in canonical['evaluation']])
        reference=np.load(folder/'torch_reference.npy',allow_pickle=False)
        np.save(folder/f'output_{name}.npy',output,allow_pickle=False)
        record['output_sha256']=sha256(folder/f'output_{name}.npy')
        record['metrics_vs_torch']=numerical_metrics(reference,output)
        record['center_logit_metrics']=numerical_metrics(reference[:,:,:,0],output[:,:,:,0])
        record['displacement_metrics']=numerical_metrics(reference[:,:,:,1:5],output[:,:,:,1:5])
        record['allclose_fp32']=bool(np.allclose(reference,output,atol=1e-4,rtol=1e-4))
    for _ in range(args.warmup): run(first)
    latency=[]
    for _ in range(args.runs):
        start=time.perf_counter_ns(); values=run(first)
        latency.append((time.perf_counter_ns()-start)/1e6)
    record['benchmark']={'batch':1,'warmup':args.warmup,'runs':args.runs,
                         'median_ms':float(np.median(latency)),'p95_ms':float(np.percentile(latency,95)),
                         'samples_ms':latency,'scope':'inference API, input quantization and output copy/dequantization; excludes preprocessing/load',
                         'embedded_decoding':args.variant in ('official','decoded_fp32')}
    if len(values)==1:
        decoded=decode(values[0])
    else:
        points=next(y for y in values if y.shape==(1,200,2))
        scores=next(y for y in values if y.shape==(1,200))
        vmap=next(y for y in values if y.shape==(1,256,256,4))
        decoded=points,scores,vmap
    segments=lines(decoded)
    with (folder/f'lines_{name}.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['x1','y1','x2','y2','score']); writer.writerows(segments.tolist())
    record['demo_line_count']=len(segments)
    np.savez_compressed(folder/f'demo_{name}.npz',points=decoded[0],scores=decoded[1],vmap=decoded[2],lines=segments)
    write_json(folder/f'measure_{name}.json',record)
    print(f"{name}: median {record['benchmark']['median_ms']:.3f} ms, lines {len(segments)}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('fp32','fp16','dynamic','static','decoded_fp32','official'),required=True)
    parser.add_argument('--runtime',choices=('old','new'),default='old')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=30); parser.add_argument('--runs',type=int,default=200)
    parser.add_argument('--run-id',default='run-01')
    args=parser.parse_args()
    if min(args.threads,args.warmup,args.runs)<1: parser.error('Positive benchmark settings required')
    main(args)
