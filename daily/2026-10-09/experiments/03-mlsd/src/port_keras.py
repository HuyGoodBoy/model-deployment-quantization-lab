"""Copy the pinned Tiny state into equivalent NHWC inference operations.

This is a weight-preserving bridge, not NAVER weights and not retraining.
Every FP32 bridge must pass the same-checkpoint output gate before PTQ.
Architecture adapted from lhwcv/mlsd_pytorch, revision
2312205254e66911703decf775f626995d260f17, under Apache-2.0.
See LICENSE-MLSD-PYTORCH.txt and sources.lock.json in the task folder.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np


TASK=Path(__file__).resolve().parents[1]
REPO=TASK.parents[3]
sys.path.insert(0,str(REPO/'quantization'))
from qlab.common import numerical_metrics, sha256, tensorflow, write_json


def build(state):
    tf=tensorflow(1)
    L=tf.keras.layers
    def conv(x,prefix,stride=1,depthwise=False,dilation=1):
        w=state[prefix+'.weight']; k=int(w.shape[-1])
        bias=prefix+'.bias' in state
        if stride==2:
            x=L.ZeroPadding2D(((0,1),(0,1)))(x)
        options=dict(kernel_size=k,strides=stride,padding='valid' if stride==2 or k==1 else 'same',
                     dilation_rate=dilation,use_bias=bias)
        layer=L.DepthwiseConv2D(**options) if depthwise else L.Conv2D(int(w.shape[0]),**options)
        y=layer(x)
        kernel=w.transpose(2,3,0,1) if depthwise else w.transpose(2,3,1,0)
        layer.set_weights([kernel,state[prefix+'.bias']] if bias else [kernel])
        return y
    def bn(x,prefix):
        layer=L.BatchNormalization(epsilon=1e-5)
        y=layer(x,training=False)
        layer.set_weights([state[prefix+'.weight'],state[prefix+'.bias'],
                           state[prefix+'.running_mean'],state[prefix+'.running_var']])
        return y
    def cbr(x,prefix,stride=1,depthwise=False):
        return L.ReLU(max_value=6)(bn(conv(x,prefix+'.0',stride,depthwise),prefix+'.1'))
    def decoder_conv(x,prefix,dilation=1):
        return L.ReLU()(bn(conv(x,prefix+'.0',dilation=dilation),prefix+'.1'))
    def resize(x,h,w):
        return L.Lambda(lambda v:tf.raw_ops.ResizeBilinear(images=v,size=[h,w],
                          align_corners=True,half_pixel_centers=False))(x)
    raw=L.Input((512,512,4),batch_size=1,name='rgba')
    x=L.Lambda(lambda v:v/127.5-1)(raw)
    x=cbr(x,'backbone.features.0',stride=2)
    feature=1; channels=32; selected=[]
    for ratio,out,n,stride in [(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2)]:
        for repeat in range(n):
            step=stride if repeat==0 else 1
            prefix=f'backbone.features.{feature}.conv'
            residual=x; offset=0
            if ratio!=1:
                x=cbr(x,prefix+'.0'); offset=1
            x=cbr(x,prefix+f'.{offset}',stride=step,depthwise=True)
            x=conv(x,prefix+f'.{offset+1}')
            x=bn(x,prefix+f'.{offset+2}')
            if step==1 and channels==out:
                x=L.Add()([x,residual])
            channels=out
            if feature in (3,6,10): selected.append(x)
            feature+=1
    c2,c3,c4=selected
    def block_a(a,b,prefix):
        a=decoder_conv(a,prefix+'.conv2')
        b=decoder_conv(b,prefix+'.conv1')
        b=resize(b,int(a.shape[1]),int(a.shape[2]))
        return L.Concatenate()([a,b])
    def block_b(x,prefix):
        x=L.Add()([decoder_conv(x,prefix+'.conv1'),x])
        return decoder_conv(x,prefix+'.conv2')
    x=block_b(block_a(c3,c4,'block12'),'block13')
    x=block_b(block_a(c2,x,'block14'),'block15')
    x=decoder_conv(x,'block16.conv1',dilation=5)
    x=decoder_conv(x,'block16.conv2')
    x=conv(x,'block16.conv3')
    x=L.Lambda(lambda v:v[:,:,:,7:])(x)
    x=resize(x,256,256)
    return tf.keras.Model(raw,x,name='pytorch_mlsd_tiny_bridge')


def main(args):
    folder=TASK/'results'/args.run_id; folder.mkdir(parents=True,exist_ok=True)
    record={'variant':args.variant,'route':'same PyTorch state -> checked Keras bridge -> TFLite'}
    try:
        state=np.load(TASK/'artifacts/torch_state.npz',allow_pickle=False)
        model=build(state)
        import tensorflow as tf
        inputs=np.load(TASK/'artifacts/inputs.npz',allow_pickle=False)
        if args.variant=='fp32':
            output=np.concatenate([model(x[None],training=False).numpy() for x in inputs['evaluation']])
            reference=np.load(folder/'torch_reference.npy',allow_pickle=False)
            np.save(folder/'keras_reference.npy',output,allow_pickle=False)
            difference=np.abs(reference.astype(np.float64)-output)
            metrics=numerical_metrics(reference,output)
            gate={**metrics,
                  'allclose_1e4':bool(np.allclose(reference,output,atol=1e-4,rtol=1e-4)),
                  'channel_max_abs':difference.max(axis=(0,1,2)).tolist(),
                  'gate_relative_l2_limit':1e-4,'gate_max_abs_limit':0.1,
                  'gate_rationale':'FP32 accumulation differs across kernels. 0.1 map units limits displacement drift to 0.2 input pixels; not bit equivalence.',
                  'gate_passed':bool(np.isfinite(output).all() and metrics['relative_l2']<1e-4 and difference.max()<0.1)}
            write_json(folder/'bridge_validation.json',gate)
            if not gate['gate_passed']:
                raise ValueError('Keras bridge fails same-checkpoint FP32 output gate')
        else:
            import json
            validation=json.loads((folder/'bridge_validation.json').read_text())
            if not validation.get('gate_passed',validation.get('allclose',False)):
                raise ValueError('Validate FP32 bridge first')
        if args.variant=='decoded_fp32':
            raw=model.output
            heat=tf.math.sigmoid(raw[:,:,:,0:1])
            maxima=tf.nn.max_pool2d(heat,ksize=3,strides=1,padding='SAME')
            heat=tf.where(tf.equal(heat,maxima),heat,tf.zeros_like(heat))
            scores,indices=tf.math.top_k(tf.reshape(heat,[1,-1]),k=200)
            points=tf.stack([indices//256,indices%256],axis=-1)
            model=tf.keras.Model(model.input,[points,scores,raw[:,:,:,1:5]])
        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        if args.variant in ('fp16','dynamic','static'):
            converter.optimizations=[tf.lite.Optimize.DEFAULT]
        if args.variant=='fp16': converter.target_spec.supported_types=[tf.float16]
        if args.variant=='static':
            converter.representative_dataset=lambda:([x[None]] for x in inputs['calibration'])
            converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type=tf.int8
            converter.inference_output_type=tf.int8
        path=TASK/'artifacts'/f'{args.variant}.tflite'
        start=time.perf_counter(); path.write_bytes(converter.convert())
        record.update(status='ok',seconds=time.perf_counter()-start,size_bytes=path.stat().st_size,
                      sha256=sha256(path),tensorflow=tf.__version__)
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(folder/f'conversion_{args.variant}.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('fp32','fp16','dynamic','static','decoded_fp32'),required=True)
    parser.add_argument('--run-id',default='run-01')
    main(parser.parse_args())
