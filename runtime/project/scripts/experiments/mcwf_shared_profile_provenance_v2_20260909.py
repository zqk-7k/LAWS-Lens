#!/usr/bin/env python3
"""Snapshot scientific runtime dependencies and compare the archived R49 hashes."""
import argparse
from pathlib import Path
import re
import shutil
import sys
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as p


def main(root, prior):
    previous=pd.read_csv(prior/'manifest/RUNTIME_DEPENDENCIES.csv').set_index('source').sha256.to_dict()
    original=[]
    for source,expected in previous.items():
        actual=p.n.sha(Path(source))
        original.append({'source':source,'archived_sha256':expected,'current_sha256':actual,'unchanged':actual==expected})
    if not all(row['unchanged']for row in original):
        raise RuntimeError('A previous scientific runtime changed; retain current results for audit')
    records=[]
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source is None:
            continue
        source=Path(source).resolve()
        if source.suffix!='.py' or not source.is_relative_to(P/'scripts'):
            continue
        content=source.read_text(errors='replace')
        if re.search(r'-----BEGIN(?: RSA| OPENSSH| EC)? PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}-----END(?: RSA| OPENSSH| EC)? PRIVATE KEY-----',content):
            raise RuntimeError('Private-key content cannot be copied into delivery')
        target=root/'scripts/runtime_dependencies'/source.relative_to(P/'scripts')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if p.n.sha(target)!=p.n.sha(source):
                raise RuntimeError('Refuse to overwrite an earlier dependency snapshot')
        else:
            shutil.copy2(source,target)
        records.append({'source':str(source),'destination':str(target.relative_to(root)),
            'sha256':p.n.sha(source),'also_verified_against_R49':str(source)in previous})
    p.n.write_csv(root/'audit/R49_RUNTIME_HASH_UNCHANGED.csv',original)
    p.n.write_csv(root/'manifest/RUNTIME_DEPENDENCIES.csv',pd.DataFrame(records).drop_duplicates('source').to_dict('records'))
    shutil.copy2(__file__,root/'scripts/shared_profile_provenance.py')
    p.n.write_json(root/'contracts/RUNTIME_PROVENANCE_AUDIT.json',{
        'UTC':p.n.utc(),'prior_archive_root':str(prior),'prior_sources':len(previous),
        'prior_hash_failures':0,'snapshot_project_modules':len(records),
        'timing':'Dependency snapshot taken during computation; previous87-source R49 snapshot predates R50 and is unchanged.',
        'raw_strain_or_PE_copied':False,'status':p.n.STATUS})
    print('RUNTIME_PROVENANCE_PASS',len(previous),len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prior-root',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.prior_root)
