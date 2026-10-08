"""Export and quantize exactly one variant, then audit actual graph/tensor dtypes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import time

import numpy as np

from .common import ARTIFACTS, DATA, RESULTS, VARIANTS, load_inputs, model_path, read_json, samples, sha256, write_json


def audit_onnx(path):
    import onnx
    model=onnx.load(path)
    onnx.checker.check_model(model)
    counts=Counter(n.op_type for n in model.graph.node)
    types=Counter()
    for tensor in model.graph.initializer:
        types[onnx.TensorProto.DataType.Name(tensor.data_type)]+=int(np.prod(tensor.dims))
    return {'operator_counts':dict(counts),'initializer_elements_by_dtype':dict(types),
            'qdq_nodes':counts['QuantizeLinear']+counts['DequantizeLinear'],
            'integer_matmul_nodes':counts['MatMulInteger'],
            'note':'QDQ storage graph retains float operator definitions; execution fusion depends on ORT/provider.'}


def audit_tflite(path,task,variant):
    import tensorflow as tf
    interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=1)
    interpreter.allocate_tensors()
    tensors=interpreter.get_tensor_details()
    counts=Counter(np.dtype(x['dtype']).name for x in tensors)
    ops=Counter(x['op_name'] for x in interpreter._get_ops_details())
    def spec(x):
        return {'name':x['name'],'dtype':np.dtype(x['dtype']).name,'shape':x['shape'].tolist(),
                'scales':x['quantization_parameters']['scales'].tolist(),
                'zero_points':x['quantization_parameters']['zero_points'].tolist()}
    inputs=interpreter.get_input_details();outputs=interpreter.get_output_details()
    if task=='cv' and variant=='tflite_static':
        if any(x['dtype']!=np.int8 for x in inputs+outputs) or counts['float32']:
            raise ValueError('CV full INT8 graph unexpectedly contains FP32 tensors/interface')
    if task=='text' and any(x['dtype']!=np.int32 for x in inputs):
        raise ValueError('Tokenizer IDs/masks were incorrectly quantized')
    return {'operator_counts':dict(ops),'tensor_counts_by_dtype':dict(counts),
            'inputs':[spec(x) for x in inputs],'outputs':[spec(x) for x in outputs],
            'float32_tensors':counts['float32'],'int8_tensors':counts['int8'],
            'int32_tensors':counts['int32'],'uses_select_tf_ops':any(k.startswith('Flex') for k in ops)}


def convert(task,variant):
    path=model_path(task,variant)
    path.parent.mkdir(parents=True,exist_ok=True)
    parameters={}
    if variant=='onnx_fp32':
        from .runtime import reference_function
        _,model,infer=reference_function(task)
        import tf2onnx
        tf2onnx.convert.from_function(infer,input_signature=infer.input_signature,opset=13,output_path=str(path))
        # Export separates the correctness baseline from the preprocessed quantization source.
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(path),str(path.with_name('onnx_preprocessed.onnx')),
                          skip_optimization=False,skip_symbolic_shape=False,skip_onnx_shape=False)
        parameters={'opset':13,'preprocessed':'onnx_preprocessed.onnx'}
    elif variant.startswith('onnx'):
        from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                              QuantType, quantize_dynamic, quantize_static)
        source=model_path(task,'onnx_fp32').with_name('onnx_preprocessed.onnx')
        ops=['Conv','MatMul','Gemm'] if variant=='onnx_static' and task=='cv' else ['MatMul','Gemm']
        if variant=='onnx_dynamic':
            quantize_dynamic(str(source),str(path),weight_type=QuantType.QInt8,
                             per_channel=True,op_types_to_quantize=ops)
            parameters={'strategy':'dynamic','weight_type':'QInt8','per_channel':True,'op_types':ops}
        else:
            inputs,_,manifest=load_inputs(task,'calibration')
            class Reader(CalibrationDataReader):
                def __init__(self): self.rewind()
                def get_next(self): return next(self.iterator,None)
                def rewind(self): self.iterator=iter(samples(inputs))
            quantize_static(str(source),str(path),Reader(),quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
                op_types_to_quantize=ops,calibrate_method=CalibrationMethod.MinMax)
            parameters={'strategy':'static QDQ S8S8','calibration_method':'MinMax','per_channel':True,
                        'op_types':ops,'calibration_sha256':manifest['tensor_hashes']['calibration']}
    else:
        from .runtime import reference_function
        tf,model,infer=reference_function(task)
        converter=tf.lite.TFLiteConverter.from_concrete_functions([infer.get_concrete_function()],model)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        if variant!='tflite_fp32': converter.optimizations=[tf.lite.Optimize.DEFAULT]
        if variant=='tflite_fp16': converter.target_spec.supported_types=[tf.float16]
        if variant=='tflite_static':
            inputs,_,manifest=load_inputs(task,'calibration')
            def representative():
                for item in samples(inputs): yield item
            converter.representative_dataset=representative
            if task=='cv':
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type=tf.int8
                converter.inference_output_type=tf.int8
            else:
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
                converter.inference_output_type=tf.int8
            parameters={'strategy':'full INT8' if task=='cv' else 'calibrated mixed INT8',
                        'calibration_sha256':manifest['tensor_hashes']['calibration'],
                        'select_tf_ops_allowed':False}
        if variant=='tflite_dynamic': parameters={'strategy':'dynamic range weights; inspect graph for coverage'}
        if variant=='tflite_fp16': parameters={'strategy':'FP16 weights; CPU may dequantize to FP32'}
        path.write_bytes(converter.convert())
    audit=audit_onnx(path) if variant.startswith('onnx') else audit_tflite(path,task,variant)
    return parameters,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=('cv','text'))
    parser.add_argument('--variant',required=True,choices=VARIANTS[1:])
    args=parser.parse_args()
    started=time.perf_counter()
    record={'task':args.task,'variant':args.variant,'converted_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        parameters,audit=convert(args.task,args.variant)
        path=model_path(args.task,args.variant)
        record.update({'status':'ok','sha256':sha256(path),'size_bytes':path.stat().st_size,
                       'parameters':parameters,'audit':audit,
                       'original_model_sha256':read_json(ARTIFACTS/args.task/'model_info.json')['original_sha256']})
        print(f'{args.task}/{args.variant}: {path.stat().st_size/1048576:.2f} MiB',flush=True)
    except Exception as error:
        record.update({'status':'failed','error_type':type(error).__name__,'error':str(error)[:4000]})
        write_json(RESULTS/args.task/f'conversion_{args.variant}.json',record)
        raise
    finally:
        record['conversion_seconds']=time.perf_counter()-started
        write_json(RESULTS/args.task/f'conversion_{args.variant}.json',record)


if __name__=='__main__': main()
