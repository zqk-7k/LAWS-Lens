#!/usr/bin/env python3
"""Range-checkpointed public downloads through an authorized loopback relay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import shutil
import sys
from urllib.parse import urlencode

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_bulk_noise_range_runtime_20260909 as ranges
b=ranges.b
PORT=None
FETCH=ranges.fetch


def fetch(url,*args,**kwargs):
    return FETCH(f'http://127.0.0.1:{PORT}/?'+urlencode({'url':url}),*args,**kwargs)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args()
    PORT=args.port
    ranges.ROOT=b.ROOT=b.g.ROOT=args.root
    ranges.CHUNK=4*1024*1024
    ranges.fetch=fetch
    b.download=ranges.download
    b.g.noise=b.noise
    b.g.SEED=b.SEED
    path=args.root/'contracts/RANGE_RELAY_TRANSPORT_ADDENDUM.json'
    if not path.exists():
        b.n.write_json(path,{'UTC':b.n.utc(),'transport_only':True,
            'reason':'LongHTTPresponsesintermittentlytruncated;bounded4MiBrangesretaincompletedchunks.',
            'rule':'8concurrent4MiB206ranges,validateContent-Range/sizeandfullHDF5,fsyncbeforechunkreceipt,SHA256aftercomplete.',
            'unchanged':'Frozenproposals,sourceIDs,noisequality,folds,allscienceandmetrics.',
            'script_sha256':b.n.sha(Path(__file__))})
        shutil.copy2(__file__,args.root/'scripts/bulk_noise_relay_range_runtime.py')
    for dep in b.n.DEPS:
        b.g.generate(dep,args.workers)
