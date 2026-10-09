"""Record installed versions and distinguish missing tools from measured runs."""
import importlib.metadata
import json
import platform
from pathlib import Path

folder=Path(__file__).parent/'experiments/01-reproduce/results/run-01'
folder.mkdir(parents=True,exist_ok=True)
packages={}
for name in ('torch','torchvision','numpy','Pillow','onnx','onnxruntime','ai-edge-litert','litert-torch','ai-edge-quantizer'):
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
record={'platform':platform.platform(),'python':platform.python_version(),'packages':packages,
        'peer_revision':'536b9ae1f9e1b3829ac4776a98fb05677df9b6aa',
        'exact_peer_environment':'not_reproduced',
        'installation_failure':'litert-torch 0.9.4 requires litert-converter==0.4.*; pip has no matching Windows distribution',
        'scope':'Windows software/runtime comparison, not Mac M1 Pro hardware reproduction; peer source remains read-only'}
(folder/'environment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
