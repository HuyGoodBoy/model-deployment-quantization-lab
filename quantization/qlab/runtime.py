"""Use common canonical inputs; adapt only dtype at the TFLite interface."""
import numpy as np

from .common import CACHE, SEQ, configure_cpu, dequantize_tensor, model_path, quantize_tensor, tensorflow


def reference_function(task,threads=1):
    tf=tensorflow(threads)
    if task=='cv':
        model=tf.keras.models.load_model(model_path(task,'tensorflow_fp32'),compile=False)
        @tf.function(input_signature=[tf.TensorSpec([1,224,224,3],tf.float32,name='image')],autograph=False)
        def infer(image): return model(image,training=False)
    else:
        from transformers import TFDistilBertForSequenceClassification
        model=TFDistilBertForSequenceClassification.from_pretrained(CACHE/'text_model',local_files_only=True)
        @tf.function(input_signature=[tf.TensorSpec([1,SEQ],tf.int32,name='input_ids'),
                                     tf.TensorSpec([1,SEQ],tf.int32,name='attention_mask')],autograph=False)
        def infer(input_ids,attention_mask):
            return model(input_ids=input_ids,attention_mask=attention_mask,training=False).logits
    return tf,model,infer


def scalar_quantization(detail):
    spec=detail['quantization_parameters']
    if len(spec['scales'])!=1 or len(spec['zero_points'])!=1:
        raise ValueError('Expected per-tensor quantization at model interface')
    scale=float(spec['scales'][0]);zero=int(spec['zero_points'][0])
    if scale<=0: raise ValueError('Invalid interface quantization scale')
    return scale,zero


def create_runner(task,variant,threads=1):
    configure_cpu(threads)
    if variant=='tensorflow_fp32':
        tf,model,infer=reference_function(task,threads)
        names=('image',) if task=='cv' else ('input_ids','attention_mask')
        return lambda inputs:infer(*[tf.convert_to_tensor(inputs[k]) for k in names]).numpy(), {
            'version':tf.__version__,'mode':'tf.function CPU','inputs':list(names)}
    if variant.startswith('onnx'):
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session=ort.InferenceSession(str(model_path(task,variant)),sess_options=options,
                                     providers=['CPUExecutionProvider'])
        names=[x.name for x in session.get_inputs()]
        expected={'image'} if task=='cv' else {'input_ids','attention_mask'}
        if set(names)!=expected: raise ValueError(f'Unexpected ONNX input names: {names}')
        return lambda inputs:session.run(None,{k:inputs[k] for k in names})[0], {
            'version':ort.__version__,'providers':session.get_providers(),
            'inputs': [{'name':x.name,'type':x.type,'shape':x.shape} for x in session.get_inputs()]}
    import tensorflow as tf
    interpreter=tf.lite.Interpreter(model_path=str(model_path(task,variant)),num_threads=threads)
    interpreter.allocate_tensors()
    inputs=interpreter.get_input_details();outputs=interpreter.get_output_details()
    if len(outputs)!=1: raise ValueError('Expected one model output')
    names=('image',) if task=='cv' else ('input_ids','attention_mask')
    mapping={}
    for name in names:
        choices=[x for x in inputs if name in x['name']]
        if len(choices)!=1: raise ValueError(f'Cannot identify TFLite input {name}')
        mapping[name]=choices[0]
    if task=='text' and any(x['dtype']!=np.int32 for x in mapping.values()):
        raise ValueError('Text token IDs/masks must remain INT32')
    out=outputs[0]
    def run(canonical):
        for name,detail in mapping.items():
            value=canonical[name]
            if value.dtype!=detail['dtype']:
                value=quantize_tensor(value,detail['dtype'],*scalar_quantization(detail))
            interpreter.set_tensor(detail['index'],value)
        interpreter.invoke()
        result=interpreter.get_tensor(out['index'])
        if result.dtype in (np.int8,np.uint8): result=dequantize_tensor(result,*scalar_quantization(out))
        return result
    def describe(detail):
        value={'name':detail['name'],'shape':detail['shape'].tolist(),'dtype':np.dtype(detail['dtype']).name}
        if len(detail['quantization_parameters']['scales'])==1:
            value.update(dict(zip(('scale','zero_point'),scalar_quantization(detail))))
        return value
    return run, {'version':tf.__version__,'mode':'tf.lite.Interpreter CPU; default delegates, see log',
                 'inputs':[describe(x) for x in inputs],'output':describe(out)}
