"""Allowlisted submission ZIP with raw evidence and per-file SHA256 inventory."""
import hashlib
import json
from pathlib import Path
import zipfile

DAY=Path(__file__).resolve().parent; ROOT=DAY.parents[1]


def main():
    selected=[]
    for p in DAY.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(DAY)
        if any(x in ('artifacts','__pycache__') for x in relative.parts):continue
        if p.suffix.lower() in ('.py','.java','.md','.txt','.json','.csv','.npy','.npz'):
            selected.append(p)
    q=ROOT/'quantization'
    selected.extend(p for p in q.iterdir() if p.is_file() and p.suffix in ('.py','.md','.txt','.json'))
    selected.extend((q/'qlab').glob('*.py'))
    for task in ('cv','text'):
        selected.extend([q/f'data/{task}/manifest.json',q/f'results/{task}/outputs.npz'])
        selected.extend((q/f'results/{task}').glob('*.json'))
    selected.extend([ROOT/'README.md',ROOT/'.gitignore'])
    output=ROOT/'dist/huy-task-2026-10-09.zip';output.parent.mkdir(exist_ok=True)
    inventory=[]
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(set(selected)):
            if not p.exists():continue
            name=p.relative_to(ROOT).as_posix()
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            inventory.append({'path':name,'bytes':p.stat().st_size,'sha256':digest})
            archive.write(p,name)
        archive.writestr('SUBMISSION-MANIFEST.json',json.dumps({'date':'2026-10-09','author':'Huy',
            'scope':'CPU experiments; incomplete full peer/Android reproduction; see REPORT',
            'files':inventory},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(f'{output} ({output.stat().st_size/2**20:.1f} MiB), {len(inventory)} files, ZIP CRC verified')


if __name__=='__main__':main()
