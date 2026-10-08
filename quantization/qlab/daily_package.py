"""Package CPU/GPU code, reports and evidence, then verify an extracted copy."""
import hashlib
import json
import subprocess
import sys
import unittest
import zipfile

from .common import ROOT, sha256, write_json
from .package import included_files


def main():
    files = included_files()
    files += [ROOT / name for name in ('run_gpu.py', 'README-gpu.md', 'requirements-gpu.txt',
              'requirements-gpu-lock.txt', 'bao-cao-ngay-2026-10-08-cpu-gpu.md',
              'bao-cao-chat-2026-10-08.md')]
    files += sorted((ROOT / 'results/gpu').rglob('*.json'))
    files += sorted((ROOT / 'results/gpu').rglob('*.npy'))
    files += sorted((ROOT / 'results/gpu').rglob('*.log'))
    files += sorted((ROOT / 'results/gpu').glob('*.csv'))
    files += sorted((ROOT / 'logs/gpu').glob('*.log'))
    files = sorted(set(files))
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Missing/unsafe file: {path.name}')
        if any(part in path.parts for part in ('.venv','.venv-gpu','.cache','.env')):
            raise ValueError('Cache/environment entered handoff allowlist')
    inventory = {path.relative_to(ROOT).as_posix(): {'size_bytes': path.stat().st_size,
                                                    'sha256': sha256(path)} for path in files}
    prefix = 'quantization-cpu-gpu-2026-10-08'
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    target = dist / (prefix + '.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path, f'{prefix}/{path.relative_to(ROOT).as_posix()}')
        archive.writestr(f'{prefix}/FILE_MANIFEST.json',json.dumps(inventory,indent=2,ensure_ascii=False))
    digest = sha256(target)
    extraction = ROOT / '.cache/daily-package-smoke' / digest[:12]
    extraction.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC mismatch')
        for entry in archive.infolist():
            if not (extraction/entry.filename).resolve().is_relative_to(extraction.resolve()):
                raise ValueError('Unsafe archive path')
        for name, record in inventory.items():
            if hashlib.sha256(archive.read(f'{prefix}/{name}')).hexdigest() != record['sha256']:
                raise ValueError('Archive hash mismatch')
        archive.extractall(extraction)
    project = extraction / prefix
    for args in (['-m','qlab.verify'], ['-m','qlab.gpu_verify'],
                 ['-m','unittest','discover','-s','tests','-v'], ['-m','qlab.daily_report']):
        subprocess.run([sys.executable,*args],cwd=project,check=True)
    tests = unittest.TestLoader().discover(str(ROOT/'tests')).countTestCases()
    (dist/(target.name+'.sha256')).write_text(digest+'  '+target.name+'\n',encoding='ascii')
    write_json(dist/'daily_package_info.json',{'zip': target.name, 'size_bytes': target.stat().st_size,
        'sha256': digest, 'file_count': len(files)+1, 'offline_extracted_cpu_verifier': 'passed',
        'offline_extracted_gpu_verifier': 'passed', 'unit_tests': tests,
        'offline_report_regeneration': 'passed', 'large_weights_datasets_included': False})
    print(f'Verified daily ZIP: {target} ({target.stat().st_size/1048576:.2f} MiB)',flush=True)


if __name__ == '__main__':
    main()
