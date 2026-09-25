#!/usr/bin/env python3
"""Exact HDF5 hyperslab access with verified public HTTP byte-range caching."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlencode,urlparse
import h5py
import numpy as np
import pandas as pd
from fsspec.spec import AbstractBufferedFile
import fsspec
from gwosc.locate import get_run_urls

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_bulk_noise_range_runtime_20260909 as transport
b=transport.b
n=b.n
ROOT=None
PORT=29742
ORIGINAL_READ=b.read
PILOT_NAME='LAZY_HDF_READER_PARALLEL_PILOT.json'


def fetch(url,lo,hi,total=None,etag=None):
    return transport.fetch('http://127.0.0.1:'+str(PORT)+'/?'+urlencode({'url':url}),lo,hi,total,etag)


def descriptor(url):
    path=ROOT/'raw_sources'/(Path(urlparse(url).path).name+'.lazy.json')
    if path.exists():
        return path
    _,total,etag,modified=fetch(url,0,0)
    if total>256*2**20:
        raise RuntimeError('HOLD_UNEXPECTED_SOURCE_SIZE')
    data={'UTC':n.utc(),'source_kind':'public_HDF5_range_descriptor_NOT_full_file',
        'url':url,'total':total,'ETag':etag,'Last-Modified':modified,
        'whole_original_SHA256':None,'block_SHA256':'cache/public_hdf_ranges/'+hashlib.sha256(url.encode()).hexdigest(),
        'decoder':'Unmodified h5py HDF5 parser; original float64 samples cast to float32 exactly as the frozen full-file reader.',
        'no_upsampling_or_synthetic_replacement':True}
    n.write_json(path,data)
    return path


class RangeFile(AbstractBufferedFile):
    def __init__(self,path):
        self.source=json.loads(Path(path).read_text())
        self.folder=ROOT/'cache/public_hdf_ranges'/hashlib.sha256(self.source['url'].encode()).hexdigest()
        self.folder.mkdir(parents=True,exist_ok=True)
        self.bytes_read=0
        super().__init__(None,self.source['url'],mode='rb',size=self.source['total'],
                         block_size=4*2**20,cache_type='blockcache',cache_options={'maxblocks':16})

    def _fetch_range(self,start,end):
        end=min(end,self.size)
        if end<=start:
            return b''
        path=self.folder/f'{start:012}_{end:012}.bin'
        receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            content=path.read_bytes()
            record=json.loads(receipt.read_text())
            if len(content)!=end-start or hashlib.sha256(content).hexdigest()!=record['sha256']:
                raise RuntimeError('Cached HDF5 byte block changed')
            return content
        if shutil.disk_usage(ROOT).free<25*2**30:
            raise RuntimeError('HOLD_DISK_LIMIT')
        content,_,etag,modified=fetch(self.source['url'],start,end-1,self.source['total'],self.source['ETag'])
        if etag!=self.source['ETag'] or modified!=self.source['Last-Modified']:
            raise RuntimeError('Public HDF5 source changed')
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as output:
            output.write(content);output.flush();os.fsync(output.fileno())
        temporary.replace(path)
        record={'url':self.source['url'],'start':start,'end_exclusive':end,'bytes':len(content),
                'ETag':etag,'Last-Modified':modified,'sha256':hashlib.sha256(content).hexdigest()}
        temporary=receipt.with_suffix('.tmp')
        n.write_json(temporary,record);temporary.replace(receipt)
        self.bytes_read+=len(content)
        return content


class LazyStrain:
    def __init__(self,path):
        self.file=None
        self.proxy=None
        self.proxy=RangeFile(path)
        self.file=h5py.File(self.proxy,'r')

    def __getitem__(self,index):
        dataset=self.file['strain/Strain']
        if isinstance(index,slice) and dataset.chunks is not None:
            lo,hi,step=index.indices(dataset.shape[0])
            if step>0 and hi>lo:
                size=dataset.chunks[0]
                blocks=set()
                for position in range((lo//size)*size,hi,size):
                    chunk=dataset.id.get_chunk_info_by_coord((position,))
                    if chunk.byte_offset is not None and chunk.size:
                        blocks.update(range(chunk.byte_offset//self.proxy.blocksize,
                            (chunk.byte_offset+chunk.size-1)//self.proxy.blocksize+1))
                # HDF5 determines native compressed chunk offsets. Network I/O runs
                # outside HDF5 calls, then the unchanged decoder reads cached bytes.
                with ThreadPoolExecutor(max_workers=min(8,max(1,len(blocks)))) as pool:
                    jobs=[pool.submit(self.proxy._fetch_range,k*self.proxy.blocksize,
                                      (k+1)*self.proxy.blocksize) for k in sorted(blocks)]
                    for job in jobs:
                        job.result()
        return np.asarray(dataset[index],np.float32)

    def close(self):
        if self.file is not None:
            self.file.close();self.file=None
        if self.proxy is not None:
            self.proxy.close();self.proxy=None

    def __del__(self):
        self.close()


def read(path):
    if not str(path).endswith('.lazy.json'):
        return ORIGINAL_READ(path)
    strain=LazyStrain(path)
    file=strain.file
    start=float(file['meta/GPSstart'][()]);duration=float(file['meta/Duration'][()])
    if file['strain/Strain'].shape[0]/duration!=4096:
        raise RuntimeError('Sample rate mismatch')
    names=[v.decode() if isinstance(v,bytes) else str(v) for v in file['quality/simple/DQShortnames'][:]]
    bits=file['quality/simple/DQmask'][:]
    mask=np.ones(len(bits),bool)
    for flag in ('DATA','CBC_CAT1'):
        mask&=(bits & (1<<names.index(flag)))!=0
    names=[v.decode() if isinstance(v,bytes) else str(v) for v in file['quality/injections/InjShortnames'][:]]
    inj=file['quality/injections/Injmask'][:]
    mask&=(inj & (1<<names.index('NO_CBC_HW_INJ')))!=0
    if len(mask)!=int(duration):
        raise RuntimeError('Unexpected dataquality clock')
    return strain,start,duration,mask


def download(url):
    path=ROOT/'raw_sources'/Path(urlparse(url).path).name
    receipt=path.with_suffix('.source.json')
    if path.exists() and receipt.exists():
        if n.sha(path)!=json.loads(receipt.read_text())['sha256']:
            raise RuntimeError('Protected downloaded file changed')
        return path
    return descriptor(url)


def pilot():
    out=ROOT/'audit'/PILOT_NAME
    if out.exists():
        raise RuntimeError('Pilot already exists')
    meta=pd.read_csv(ROOT/'data/gwtc3/noise/noise_manifest.csv')
    rows=[]
    refs=np.load(ROOT/'data/gwtc3/noise/reference.npy',mmap_mode='r')
    source=meta.iloc[0]
    for detector,index in (('H1',0),('L1',1)):
        original=Path(source[detector+'_path'])
        receipt=json.loads(original.with_suffix('.source.json').read_text())
        if n.sha(original)!=receipt['sha256']:
            raise RuntimeError('Original source checksum mismatch')
        tick=time.perf_counter()
        lazy=read(descriptor(receipt['url']))
        full=ORIGINAL_READ(original)
        if lazy[1:3]!=full[1:3] or not np.array_equal(lazy[3],full[3]):
            raise RuntimeError('Metadata/data-quality mismatch')
        offset=int(round(float(source.start_gps)-full[1]))*4096
        sample=lazy[0][offset:offset+256*4096]
        reference=np.asarray(refs[int(source.bank_index),index])
        if not np.array_equal(sample,reference,equal_nan=True):
            raise RuntimeError('Previously accepted 256s noise differs')
        for sl in (slice(0,8192),slice(600*4096,600*4096+8192),slice(-8192,None)):
            if not np.array_equal(lazy[0][sl],full[0][sl],equal_nan=True):
                raise RuntimeError('Independent sample hyperslab mismatch')
        rows.append({'detector':detector,'pass':True,'metadata_and_DQ_exact':True,
            'accepted256s_reference_exact':True,'extra_hyperslabs_exact':True,'end_of_file_hyperslab_exact':True,
            'reference_sha256':hashlib.sha256(sample.tobytes()).hexdigest(),
            'whole_file_SHA256_verified':receipt['sha256'],
            'source_total_bytes':original.stat().st_size,'range_bytes_transferred':lazy[0].proxy.bytes_read,
            'wall_seconds':time.perf_counter()-tick})
        lazy[0].close()
        print('LAZY_HDF_PILOT',detector,rows[-1],flush=True)
    n.write_json(out,{'UTC':n.utc(),'passed':all(r['pass'] for r in rows),'rows':rows,
        'no_scientific_threshold_changed':True,'h5py':h5py.__version__,'fsspec':fsspec.__version__,
        'source_arrays_cast_identically':True,'runtime_sha256':n.sha(Path(__file__))})


def generate(workers):
    check=json.loads((ROOT/'audit'/PILOT_NAME).read_text())
    if not check['passed'] or check['runtime_sha256']!=n.sha(Path(__file__)):
        raise RuntimeError('Lazy reader not validated')
    contract=ROOT/'contracts/LAZY_HDF_TRANSPORT_ADDENDUM.json'
    previous=[]
    for dep in n.DEPS:
        path=ROOT/f'data/{dep}/noise/noise_manifest.csv'
        if not path.exists():
            continue
        frame=pd.read_csv(path)
        ref=np.load(path.parent/'reference.npy',mmap_mode='r')
        for row in frame.itertuples():
            previous.append({'deployment':dep,'bank_index':int(row.bank_index),
                'sha256':hashlib.sha256(np.asarray(ref[row.bank_index]).tobytes()).hexdigest()})
    if not contract.exists():
        n.write_json(contract,{'UTC':n.utc(),'transport_only':True,'pilot_sha256':n.sha(ROOT/'audit'/PILOT_NAME),
            'change':'For not-yet-downloaded files, let standard h5py read metadata/DQ and only requested strain hyperslabs via standard fsspec buffering and validated206 ranges. No array approximation or replacement.',
            'exact_same':'Registered run/file URLs, proposal order, DQ/veto/finite/PSD rules, source/noise folds,64blocks/run,all frozen model and scoring definitions.',
            'parallelism':'h5py get_chunk_info_by_coord identifies native compressed blocks for the requested sample slice. At most8 HTTP byte-block fetches run outside HDF5 calls; standard h5py decoding is unchanged.',
            'source_provenance':'Already complete original files retained. New *.lazy.json descriptors name public original URI, version,ETag,size and block-SHA cache; they are NOT claimed to be complete original HDF5 files.',
            'authoritative_noise':'Selected256s float32 references and PSDs, with full array SHA. Original reader already cast these samples to float32; pilot demonstrates bit-exact equivalence.',
            'full_original_sha_limitation':'Not available for incompletely downloaded original files; do not label partial-cache SHA as whole-file SHA.',
            'previous_reference_rows':previous,'h5py':h5py.__version__,'fsspec':fsspec.__version__,
            'code_sha256':n.sha(Path(__file__)),
            'documentation':['https://docs.h5py.org/en/stable/high/file.html#python-file-like-objects',
                             'https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractBufferedFile']})
        shutil.copy2(__file__,ROOT/'scripts/lazy_public_hdf_runtime.py')
    else:
        previous=json.loads(contract.read_text())['previous_reference_rows']
    b.download=download;b.read=read
    b.get_urls=lambda det,start,end,dataset,sample_rate:get_run_urls(dataset,det,start,end,sample_rate=sample_rate)
    b.g.noise=b.noise;b.g.SEED=b.SEED
    for dep in n.DEPS:
        b.g.generate(dep,workers)
        ref=np.load(ROOT/f'data/{dep}/noise/reference.npy',mmap_mode='r')
        for row in previous:
            if row['deployment']==dep and hashlib.sha256(np.asarray(ref[row['bank_index']]).tobytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('Previously accepted noise reference changed')
    n.write_json(ROOT/'audit/LAZY_HDF_PREVIOUS_REFERENCE_REPLAY.json',{'UTC':n.utc(),
        'unchanged_rows':len(previous),'changed_rows':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('pilot','generate'),required=True)
    parser.add_argument('--port',type=int,default=29742)
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args();ROOT=b.ROOT=b.g.ROOT=args.root;PORT=args.port
    if args.stage=='pilot':
        pilot()
    else:
        generate(args.workers)
