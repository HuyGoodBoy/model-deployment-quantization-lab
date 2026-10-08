"""Download public pinned assets with checksum validation and resumable range parts."""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

from .common import CACHE, ROOT, read_json, sha256, write_json


def digest(path, kind):
    h = hashlib.new(kind)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def download(url, target, expected=None, size=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    kind = 'md5' if expected and len(expected)==32 else 'sha256'
    if target.exists() and expected and digest(target,kind)==expected: return
    if urlparse(url).scheme != 'https': raise ValueError('HTTPS required')
    if size is None:
        response = requests.head(url,allow_redirects=True,timeout=(10,30))
        response.raise_for_status()
        size = int(response.headers['Content-Length'])
    if size < 4*1024*1024:
        response=requests.get(url,timeout=(10,30))
        response.raise_for_status()
        target.write_bytes(response.content)
    else:
        parts = CACHE / 'parts' / hashlib.sha256(url.encode()).hexdigest()[:16]
        parts.mkdir(parents=True,exist_ok=True)
        chunk_size=256*1024
        def part(start):
            end=min(start+chunk_size,size)-1
            dest=parts/f'{start}-{end}.part'
            if dest.exists() and dest.stat().st_size==end-start+1: return dest
            previous_start=(start//(2*1024*1024))*(2*1024*1024)
            previous_end=min(previous_start+2*1024*1024,size)-1
            previous=parts/f'{previous_start}-{previous_end}.part'
            if previous.exists() and previous.stat().st_size==previous_end-previous_start+1:
                with previous.open('rb') as stream:
                    stream.seek(start-previous_start)
                    dest.write_bytes(stream.read(end-start+1))
                return dest
            for attempt in range(4):
                try:
                    effective=url.replace('https://s3.amazonaws.com/fast-ai-imageclas/',
                                          'https://fast-ai-imageclas.s3.amazonaws.com/')
                    response=requests.get(effective,headers={'Range':f'bytes={start}-{end}'},timeout=(10,15))
                    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Server did not honor byte range')
                    if len(response.content)!=end-start+1: raise ValueError('Incomplete byte range')
                    dest.write_bytes(response.content)
                    return dest
                except (requests.RequestException,ValueError):
                    if attempt==3: raise
                    time.sleep(1)
        starts=list(range(0,size,chunk_size))
        done=0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(part,s) for s in starts]):
                done+=future.result().stat().st_size
                if done==size or done%(16*1024*1024)==0:
                    print(f'{target.name}: {done/size:.0%}',flush=True)
        temporary=target.with_suffix(target.suffix+'.partial')
        with temporary.open('wb') as stream:
            for start in starts:
                end=min(start+chunk_size,size)-1
                stream.write((parts/f'{start}-{end}.part').read_bytes())
        temporary.replace(target)
    if target.stat().st_size!=size or (expected and digest(target,kind)!=expected):
        raise ValueError(f'Asset checksum/size mismatch: {target.name}')
    print(f'Verified {target.name}',flush=True)


def hf_assets(entry, target, dataset=False):
    for record in entry['files']:
        prefix='datasets/' if dataset else ''
        url=f"https://huggingface.co/{prefix}{entry['repo']}/resolve/{entry['revision']}/{record['rfilename']}"
        path=target/record['rfilename']
        expected=record.get('lfs',{}).get('sha256')
        download(url,path,expected,record['size'])
        if not expected:
            data=path.read_bytes()
            git_hash=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if git_hash != record['blobId']: raise ValueError('Git blob integrity mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('cv','text','all'),default='all')
    args=parser.parse_args()
    lock=read_json(ROOT/'sources.lock.json')
    if args.task in ('cv','all'):
        for key,name in (('resnet50','resnet50.h5'),('imagenette','imagenette2-160.tgz')):
            print(f'Download {key}',flush=True)
            download(lock[key]['url'],CACHE/name,lock[key]['md5'])
    if args.task in ('text','all'):
        hf_assets(lock['text_model'],CACHE/'text_model')
        hf_assets(lock['sst2'],CACHE/'sst2',dataset=True)
    inventory={str(p.relative_to(ROOT)):sha256(p) for p in CACHE.rglob('*')
               if p.is_file() and 'parts' not in p.parts and p.suffix!='.partial'}
    write_json(ROOT/'results/source_checksums.json',inventory)


if __name__=='__main__': main()
