"""Fetch a pinned Windows wheel in validated HTTP ranges and check PyPI SHA256."""
import concurrent.futures
import hashlib
from pathlib import Path
import sys
import requests

package,version=sys.argv[1:3]
meta=requests.get(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30).json()
wheel=next(u for u in meta['urls'] if 'cp311-cp311-win_amd64.whl' in u['filename'])
root=Path(__file__).resolve().parents[2]/'.cache/day09-wheels'
root.mkdir(parents=True,exist_ok=True)
target=root/wheel['filename']
size=wheel['size']; chunk=1024*1024
def fetch(i):
    start=i*chunk; end=min(size,start+chunk)-1
    piece=target.with_suffix(f'.part{i}')
    if piece.exists() and piece.stat().st_size==end-start+1: return i
    response=requests.get(wheel['url'],headers={'Range':f'bytes={start}-{end}'},timeout=60)
    response.raise_for_status()
    if len(response.content)!=end-start+1: raise ValueError('Unexpected range length')
    piece.write_bytes(response.content)
    return i
if not target.exists():
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i in pool.map(fetch,range((size+chunk-1)//chunk)):
            if i%20==0: print(f'{package}: chunk {i}',flush=True)
    with target.open('wb') as f:
        for i in range((size+chunk-1)//chunk): f.write(target.with_suffix(f'.part{i}').read_bytes())
actual=hashlib.sha256(target.read_bytes()).hexdigest()
if actual!=wheel['digests']['sha256']: raise ValueError('Wheel SHA256 mismatch')
print(str(target),flush=True)
