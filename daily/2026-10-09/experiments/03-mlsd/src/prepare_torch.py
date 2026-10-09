"""Pinned upstream Tiny reference and state export for a checked Keras bridge."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[3]
CACHE = REPO/'.cache/mlsd'
TORCH_REV = '2312205254e66911703decf775f626995d260f17'
TF_REV = '453cafa09467d0272760578d35c1fda38e8895a5'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = {
        'mbv2_mlsd_tiny.py': f'https://raw.githubusercontent.com/lhwcv/mlsd_pytorch/{TORCH_REV}/models/mbv2_mlsd_tiny.py',
        'mlsd_tiny_512_fp32.pth': f'https://raw.githubusercontent.com/lhwcv/mlsd_pytorch/{TORCH_REV}/models/mlsd_tiny_512_fp32.pth',
        'frame_1.jpg': f'https://raw.githubusercontent.com/lhwcv/mlsd_pytorch/{TORCH_REV}/data/frame_1.jpg',
        'official_tiny_fp32.tflite': f'https://raw.githubusercontent.com/navervision/mlsd/{TF_REV}/tflite_models/M-LSD_512_tiny_fp32.tflite',
    }
    previous = TASK/'sources.lock.json'
    locked = json.loads(previous.read_text()) if previous.exists() else {}
    result = {}
    for name, url in sources.items():
        path = CACHE/name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        hash_value = digest(path)
        if name in locked and locked[name]['sha256'] != hash_value:
            raise ValueError(f'Source checksum changed: {name}')
        result[name] = {'url':url, 'sha256':hash_value, 'bytes':path.stat().st_size}
    previous.write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')


def image(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR),np.float32)
    return np.concatenate([rgb,np.ones((512,512,1),np.float32)],axis=-1)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fetch()
    spec = importlib.util.spec_from_file_location('pinned_mlsd_tiny',CACHE/'mbv2_mlsd_tiny.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MobileV2_MLSD_Tiny().eval()
    state = torch.load(CACHE/'mlsd_tiny_512_fp32.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    artifacts = TASK/'artifacts'; artifacts.mkdir(exist_ok=True)
    np.savez(artifacts/'torch_state.npz',**{k:v.detach().numpy() for k,v in state.items()})
    manifest = json.loads((REPO/'quantization/data/cv/manifest.json').read_text())
    records = manifest['records']
    rows = {}
    arrays = {}
    for split, n in [('calibration',32),('evaluation',10)]:
        rng=np.random.default_rng(20261009)
        selected = [records[split][i] for i in rng.choice(len(records[split]),n,replace=False)]
        rows[split] = [{'path':r['image_path'],'sha256':r['source_sha256']} for r in selected]
        for r in selected:
            if digest(REPO/'quantization'/r['image_path'])!=r['source_sha256']:
                raise ValueError('Source image checksum mismatch: '+r['image_path'])
        arrays[split] = np.stack([image(REPO/'quantization'/r['image_path']) for r in selected])
    if set(r['sha256'] for r in rows['calibration']) & set(r['sha256'] for r in rows['evaluation']):
        raise ValueError('Calibration/evaluation overlap')
    arrays['demo'] = image(CACHE/'frame_1.jpg')[None]
    np.savez(artifacts/'inputs.npz',**arrays)
    folder = TASK/'results'/args.run_id; folder.mkdir(parents=True,exist_ok=True)
    outputs=[]
    with torch.inference_mode():
        for x in arrays['evaluation']:
            y = model(torch.from_numpy((x/127.5-1).transpose(2,0,1)[None]))
            outputs.append(y.numpy().transpose(0,2,3,1))
        sample = torch.from_numpy((arrays['demo']/127.5-1).transpose(0,3,1,2))
        for _ in range(30): model(sample)
        latency=[]
        for _ in range(200):
            start=time.perf_counter_ns(); model(sample).numpy()
            latency.append((time.perf_counter_ns()-start)/1e6)
    np.save(folder/'torch_reference.npy',np.concatenate(outputs),allow_pickle=False)
    record={'status':'ok','torch_version':torch.__version__,'threads':1,'batch':1,
            'input_contract':'raw RGB+alpha=1, Pillow bilinear 512; Torch receives NCHW normalized [-1,1]',
            'output_contract':'NHWC [N,256,256,9], channel 0 center logits, 1:5 displacement',
            'records':rows,'checkpoint_sha256':digest(CACHE/'mlsd_tiny_512_fp32.pth'),
            'canonical_inputs_sha256':digest(artifacts/'inputs.npz'),
            'reference_sha256':digest(folder/'torch_reference.npy'),
            'warmup':30,'runs':200,'median_ms':float(np.median(latency)),
            'p95_ms':float(np.percentile(latency,95)),'samples_ms':latency,
            'scope':'Torch raw graph API, normalized input already in RAM, output NumPy; excludes preprocess/load',
            'alpha_stem_max_abs_weight':float(np.abs(state['backbone.features.0.0.weight'].numpy()[:,3]).max())}
    (folder/'torch_reference.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f"Torch reference saved; median {record['median_ms']:.3f} ms",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='run-01')
    main(parser.parse_args())
