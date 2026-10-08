"""Dong goi model/input FP32 da kiem chung vao Android assets, khong chay lai model."""
import hashlib
import shutil

import numpy as np

from .common import ARTIFACTS, DATA, RESULTS, ROOT, read_json, validate_input, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / 'android/app/src/main/assets/lab'
    destination.mkdir(parents=True, exist_ok=True)
    evaluation = read_json(RESULTS / 'evaluation.json')
    manifest = read_json(DATA / 'manifest.json')
    if sha256(ARTIFACTS / 'mobilenetv2.keras') != evaluation['model_sha256']:
        raise ValueError('Model goc khong khop evaluation. Can evaluate lai truoc khi dong goi.')
    with np.load(RESULTS / 'outputs.npz', allow_pickle=False) as outputs:
        reference = outputs['tensorflow'].copy()
    if reference.shape != (len(manifest['records']), 1000):
        raise ValueError('Output tham chieu khong khop dataset.')
    bundle = {
        'schema_version': 1, 'model': 'MobileNetV2 Keras ImageNet FP32',
        'original_model_sha256': evaluation['model_sha256'],
        'input_shape': [1, 224, 224, 3], 'output_shape': [1, 1000],
        'dtype': 'float32', 'byte_order': 'little', 'models': {}, 'records': [],
        'reference_outputs_npz_sha256': sha256(RESULTS / 'outputs.npz'),
        'atol': evaluation['atol'], 'rtol': evaluation['rtol'],
    }
    conversion = read_json(ARTIFACTS / 'conversion_info.json')
    if conversion['original_sha256'] != evaluation['model_sha256']:
        raise ValueError('Conversion khong khop model da evaluate.')
    for runtime, filename in [('onnx', 'mobilenetv2.onnx'), ('tflite', 'mobilenetv2.tflite')]:
        source = ARTIFACTS / filename
        expected = conversion['files'][filename]['sha256']
        if sha256(source) != expected:
            raise ValueError(f'{runtime} artifact khong khop conversion.')
        shutil.copy2(source, destination / filename)
        bundle['models'][runtime] = {'asset': filename, 'sha256': sha256(source)}
    for index, record in enumerate(manifest['records']):
        source = ROOT / record['input_path']
        if sha256(source) != evaluation['inputs'].get(record['filename']):
            raise ValueError('Input khong khop evaluation.')
        x = np.load(source, allow_pickle=False)
        validate_input(x)
        filename = f'input_{index:04d}.f32'
        x.astype('<f4', copy=False).tofile(destination / filename)
        ref_name = f'reference_{index:04d}.f32'
        reference[index].astype('<f4', copy=False).tofile(destination / ref_name)
        bundle['records'].append({
            'filename': record['filename'], 'input_asset': filename,
            'input_sha256': sha256(destination / filename),
            'input_npy_sha256': sha256(source), 'reference_asset': ref_name,
            'reference_sha256': sha256(destination / ref_name),
        })
    write_json(destination / 'bundle.json', bundle)
    print(f'Android assets ready: {destination}; {len(bundle["records"])} images', flush=True)


if __name__ == '__main__':
    main()
