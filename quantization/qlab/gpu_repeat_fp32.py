"""One transparent repeat pair after the initially variable ResNet CPU baseline."""
import shutil

from .common import RESULTS, ROOT, read_json, write_json
from run_gpu import launch


def main():
    folder = RESULTS / 'gpu/cv'
    backup = folder / 'first_pass_fp32'
    if backup.exists():
        raise RuntimeError('A repeat was already staged; refusing to discard its first pass')
    backup.mkdir()
    for device in ('cpu','cuda'):
        for name in (f'{device}_onnx_fp32.json',f'output_{device}_onnx_fp32.npy'):
            shutil.copyfile(folder/name,backup/name)
        shutil.copyfile(ROOT/'logs/gpu'/f'cv_{device}_onnx_fp32.log',backup/f'{device}_onnx_fp32.log')
    write_json(backup/'reason.json',{'reason':'Initial ResNet CPU FP32 median ~320 ms differed substantially from earlier CPU ~81 ms and the new dynamic graph ~84 ms. Repeat CPU+CUDA once, preserve both passes, do not select the minimum latency.',
                'policy':'Primary tables use the final repeat pair; first pair remains reported separately; same protocol 30 warm-up/200 runs.'})
    for device in ('cpu','cuda'):
        launch(f'cv_{device}_onnx_fp32',['-m','qlab.gpu_experiment','--task','cv','--variant','onnx_fp32',
                                      '--device',device])


if __name__ == '__main__':
    main()
