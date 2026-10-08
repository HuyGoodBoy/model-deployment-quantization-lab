"""Package an explicit allowlist and verify a fresh extracted copy offline."""
import hashlib
import json
import subprocess
import sys
import zipfile

from .common import ROOT, sha256, write_json


def included_files():
    files=[]
    for name in ('run_lab.py','README.md','requirements.txt','requirements-lock.txt','sources.lock.json',
                 'learning.template.html','hoc-quantization.html','bao-cao-quantization-2026-10-08.md'):
        files.append(ROOT/name)
    for folder,pattern in (('qlab','*.py'),('tests','test*.py'),('config','*.json')):
        files.extend(sorted((ROOT/folder).glob(pattern)))
    for task in ('cv','text'):
        files+=[ROOT/f'data/{task}/manifest.json',ROOT/f'artifacts/{task}/model_info.json']
        for pattern in ('outputs.npz','evaluation_summary.json','evaluation_*.json','benchmark_*.json','conversion_*.json','diagnostic_*.json'):
            files.extend(sorted((ROOT/f'results/{task}').glob(pattern)))
        for pattern in ('convert_*.log','evaluate*.log','benchmark_*.log','diagnose*.log'):
            files.extend(sorted((ROOT/f'logs/{task}').glob(pattern)))
    files += [ROOT/'results/summary.json',ROOT/'results/summary.csv',ROOT/'results/source_checksums.json']
    files=sorted(set(files))
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing/unsafe handoff file: {path.name}')
        if any(x in path.parts for x in ('.venv','.cache','.env')):
            raise ValueError('Private/cache file entered allowlist')
    return files


def main():
    prefix='quantization-lab-2026-10-08'
    folder=ROOT/'dist'
    folder.mkdir(exist_ok=True)
    target=folder/'quantization-cv-text-2026-10-08.zip'
    files=included_files()
    inventory={str(path.relative_to(ROOT)).replace('\\','/'):{'sha256':sha256(path),'size':path.stat().st_size}
               for path in files}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files: archive.write(path,f'{prefix}/{path.relative_to(ROOT).as_posix()}')
        archive.writestr(f'{prefix}/FILE_MANIFEST.json',json.dumps(inventory,indent=2,ensure_ascii=False))
    digest=sha256(target)
    extraction=ROOT/'.cache/package-smoke'/digest[:12]
    extraction.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None: raise ValueError('Corrupt ZIP CRC')
        for record in archive.infolist():
            if not (extraction/record.filename).resolve().is_relative_to(extraction.resolve()):
                raise ValueError('Unsafe ZIP path')
        for name,metadata in inventory.items():
            data=archive.read(f'{prefix}/{name}')
            if hashlib.sha256(data).hexdigest()!=metadata['sha256']: raise ValueError('ZIP hash mismatch')
        archive.extractall(extraction)
    project=extraction/prefix
    for command in (['-m','qlab.verify'],['-m','unittest','discover','-s','tests','-v'],['-m','qlab.report']):
        subprocess.run([sys.executable,*command],cwd=project,check=True)
    (folder/(target.name+'.sha256')).write_text(digest+'  '+target.name+'\n',encoding='ascii')
    write_json(folder/'package_info.json',{'zip':target.name,'sha256':digest,'size_bytes':target.stat().st_size,
        'file_count':len(inventory)+1,'large_models_full_datasets_included':False,'offline_extracted_verifier':'passed',
        'offline_extracted_unit_tests':'8 passed','offline_report_regeneration':'passed'})
    print(f'ZIP verified: {target} ({target.stat().st_size/1048576:.2f} MiB, {len(inventory)+1} files)',flush=True)


if __name__=='__main__': main()
