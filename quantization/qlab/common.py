"""Paths, reproducible CPU configuration, quantization math and evaluation metrics."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
RESULTS = ROOT / 'results'
CACHE = ROOT / '.cache'
SEED = 20261008
SEQ = 64
VARIANTS = ('tensorflow_fp32', 'onnx_fp32', 'onnx_dynamic', 'onnx_static',
            'tflite_fp32', 'tflite_dynamic', 'tflite_fp16', 'tflite_static')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_json(value):
    if isinstance(value, dict): return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    if isinstance(value, np.generic): return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else '-inf' if value < 0 else None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def configure_cpu(threads=1):
    if threads < 1: raise ValueError('threads must be positive')
    for key in ('OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        os.environ[key] = str(threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(CACHE / 'huggingface')
    os.environ['KERAS_HOME'] = str(CACHE / 'keras')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def tensorflow(threads=1):
    configure_cpu(threads)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def model_path(task, variant):
    extension = '.onnx' if variant.startswith('onnx') else '.tflite'
    if variant == 'tensorflow_fp32':
        return ARTIFACTS / task / 'original.keras' if task == 'cv' else CACHE / 'text_model/tf_model.h5'
    return ARTIFACTS / task / (variant + extension)


def load_inputs(task, split):
    manifest = read_json(DATA / task / 'manifest.json')
    path = DATA / task / (split + '.npz')
    if sha256(path) != manifest['tensor_hashes'][split]:
        raise ValueError('Canonical input tensor checksum changed')
    with np.load(path, allow_pickle=False) as archive:
        inputs = {k: archive[k].copy() for k in archive.files if k != 'labels'}
        labels = archive['labels'].copy()
    if task == 'cv':
        assert inputs['image'].dtype == np.float32 and inputs['image'].shape[1:] == (224,224,3)
    else:
        assert set(inputs) == {'input_ids','attention_mask'}
        assert all(x.dtype == np.int32 and x.shape[1:] == (SEQ,) for x in inputs.values())
    if not all(np.isfinite(x).all() for x in inputs.values()): raise ValueError('Nonfinite input')
    return inputs, labels, manifest


def samples(inputs):
    count = len(next(iter(inputs.values())))
    for i in range(count):
        yield {name: np.ascontiguousarray(value[i:i+1]) for name, value in inputs.items()}


def quantize_tensor(value, dtype, scale, zero_point):
    if scale <= 0 or not np.isfinite(scale): raise ValueError('Invalid quantization scale')
    if not np.issubdtype(dtype, np.integer): raise ValueError('Quantized dtype must be integer')
    limits = np.iinfo(dtype)
    q = np.rint(np.asarray(value, dtype=np.float64) / scale) + zero_point
    return np.clip(q, limits.min, limits.max).astype(dtype)


def dequantize_tensor(value, scale, zero_point):
    if scale <= 0 or not np.isfinite(scale): raise ValueError('Invalid quantization scale')
    return ((np.asarray(value, dtype=np.float32) - zero_point) * scale).astype(np.float32)


def numerical_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.size == 0: raise ValueError('Output shape mismatch')
    a, b = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite output')
    e = b - a
    signal, noise = float(a @ a), float(e @ e)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return {'mae':float(np.abs(e).mean()), 'rmse':float(np.sqrt(np.mean(e*e))),
            'max_abs_error':float(np.abs(e).max()),
            'relative_l2':float(np.linalg.norm(e)/np.linalg.norm(a)) if signal else (0.0 if not noise else math.inf),
            'cosine_similarity':float(np.clip((a@b)/norms,-1,1)) if norms else None,
            'snr_db':10*math.log10(signal/noise) if signal and noise else math.inf if signal else None}


def classification_metrics(output, labels, task):
    predicted = output.argmax(axis=1)
    result = {'accuracy':float(np.mean(predicted == labels)),
              'top1_ties':int(np.sum(np.sum(output==output.max(axis=1,keepdims=True),axis=1)>1))}
    if task == 'cv':
        top = np.argsort(-output, axis=1, kind='stable')[:,:5]
        result['top5_accuracy'] = float(np.mean(np.any(top == labels[:,None],axis=1)))
        result['probability_sum_min'] = float(output.sum(axis=1).min())
        result['probability_sum_max'] = float(output.sum(axis=1).max())
    else:
        tp = int(np.sum((predicted==1)&(labels==1)))
        fp = int(np.sum((predicted==1)&(labels==0)))
        fn = int(np.sum((predicted==0)&(labels==1)))
        result['f1_positive'] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return result


def system_info():
    cpu_name=platform.processor()
    if platform.system()=='Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return {'platform':platform.platform(), 'processor':platform.processor(),'cpu_name':cpu_name,
            'logical_processors':os.cpu_count(),'machine':platform.machine(),'python':platform.python_version()}
