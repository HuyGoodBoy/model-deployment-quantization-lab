"""One variant per process, same canonical input and end-to-end inference timer."""
import argparse
import time

import numpy as np

from .common import RESULTS, VARIANTS, load_inputs, model_path, samples, sha256, system_info, write_json
from .runtime import create_runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=('cv','text'))
    parser.add_argument('--variant',required=True,choices=VARIANTS)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=30)
    parser.add_argument('--runs',type=int,default=200)
    args=parser.parse_args()
    if min(args.threads,args.warmup,args.runs)<1: parser.error('Positive settings required')
    inputs,_,manifest=load_inputs(args.task,'evaluation')
    x=next(samples(inputs))
    run,info=create_runner(args.task,args.variant,args.threads)
    for _ in range(args.warmup): run(x)
    latency=[]
    for _ in range(args.runs):
        start=time.perf_counter_ns()
        y=run(x)
        latency.append((time.perf_counter_ns()-start)/1e6)
    if not np.isfinite(y).all(): raise ValueError('Nonfinite benchmark output')
    values=np.asarray(latency)
    record={'task':args.task,'variant':args.variant,'threads':args.threads,'batch_size':1,
        'warmup':args.warmup,'runs':args.runs,'input_record':manifest['records']['evaluation'][0],
        'canonical_input_sha256':manifest['tensor_hashes']['evaluation'],
        'model_sha256':sha256(model_path(args.task,args.variant)),
        'mean_ms':float(values.mean()),'median_ms':float(np.median(values)),
        'p95_ms':float(np.percentile(values,95)),'std_ms':float(values.std()),
        'latency_samples_ms':latency,'images_or_sentences_per_second':float(1000/values.mean()),
        'runtime':info,'system':system_info(),
        'timing_scope':'inference API including in-memory copy, Python overhead, input/output quantize/dequantize when needed; excludes load, image preprocessing/tokenization and disk I/O'}
    write_json(RESULTS/args.task/f'benchmark_{args.variant}.json',record)
    print(f'{args.task}/{args.variant}: median {record["median_ms"]:.3f} ms, P95 {record["p95_ms"]:.3f} ms',flush=True)


if __name__=='__main__': main()
