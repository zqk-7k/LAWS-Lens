"""Independent count checks, test log, allowlisted manifest and verified compact archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def js(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)


def main(root):
    commands=[sys.executable,'-B','-m','unittest','test_tail_audit','-v']
    result=subprocess.run(commands,cwd=root/'scripts',text=True,capture_output=True)
    (root/'logs/UNIT_TESTS.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Unit tests failed')
    hashes=pd.read_csv(root/'manifests/INPUT_SHA256.csv')
    assert all(sha(r.path)==r.sha256 for r in hashes.itertuples())
    fit=pd.read_csv(root/'results/gpd_fits.csv')
    assert len(fit)==36 and fit.numerical_pass.all()
    assert len(pd.read_csv(root/'results/gates.csv'))==12
    assert len(pd.read_csv(root/'results/heldout_predictions.csv'))==108
    assert len(pd.read_csv(root/'results/resampling_sensitivity.csv'))==4800
    count_checks=0
    for run in ('O3','O4a','O4b'):
        e=pd.read_csv(root/f'inputs/{run}_fit_events.csv')
        h=pd.read_csv(root/f'inputs/{run}_check_events.csv')
        for field in ('source_uid','global_source_id','noise_parent_uid','event_uid'):
            assert not set(e[field].astype(str))&set(h[field].astype(str))
        b=pd.read_csv(root/f'inputs/{run}_fit_scores.csv')
        np.testing.assert_allclose(b.mean_S,b[['2026091721','2026091722','2026091723']].mean(axis=1),atol=1e-12)
        real=pd.read_csv(root/f'results/{run}_real_Top50_empirical_preserved.csv')
        assert real.tail_model_FPP.isna().all()
        x=b.mean_S.to_numpy()
        for row in real.itertuples():
            k=int(np.sum(x>=row.final_score_POSITIVE))
            assert k==row.background_exceedances
            assert np.isclose(k/len(x),row.conditional_FPP,atol=1e-15)
            count_checks+=1
    plan=pd.read_csv(root/'results/OFFICIAL_O4A_REPLAY_PLAN.csv')
    assert plan.groupby('noise_component').split.nunique().max()==1
    assert len(plan)==254
    support=pd.read_csv(root/'results/REAL_TOP10_EXTRAPOLATION_SUPPORT.csv')
    assert len(support)==30
    no_model_publication=pd.read_csv(root/'results/gates.csv').publication_pass.eq(False).all()
    assert no_model_publication
    # Keep only our two forensic files from the failed startup, never unrelated /tmp scripts.
    old=root.parent/'gwlr_tail02_20260922T113500Z'
    failure=json.loads((old/'contracts/FAILED_STARTUP_RECORD.json').read_text())
    js(root/'logs/TECHNICAL_STARTUP_REPAIR.json',failure)
    js(root/'contracts/INDEPENDENT_VERIFICATION.json',dict(passed=True,unit_tests=7,
        protected_input_files=len(hashes),real_empirical_counts_verified=count_checks,
        fits=36,score_views=12,bootstrap_sensitivity_replicates=4800,
        official_noise_disjoint_components=int(plan.noise_component.nunique()),
        no_candidate_tail_model_published=True,no_annual_FAR=True))
    # The copied official metadata contain researchers' original file paths, not credentials.
    forbidden=(b'-----BEGIN RSA PRIVATE KEY-----',b'-----BEGIN OPENSSH PRIVATE KEY-----')
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.md','.json','.csv','.log'):
            data=p.read_bytes()
            if any(token in data for token in forbidden) and p.name!='verify_package.py':
                raise RuntimeError('Secret material found')
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    pd.DataFrame(manifest).to_csv(root/'manifests/OUTPUT_SHA256.csv',index=False)
    files.append(root/'manifests/OUTPUT_SHA256.csv')
    package=root.with_name(root.name+'_diagnostic_deliverables.tar.gz')
    if package.exists():raise RuntimeError('Refuse overwrite')
    with tarfile.open(package,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    expected={str(Path(root.name)/r['path']):r['sha256'] for r in manifest}
    with tarfile.open(package,'r:gz') as tar:
        for member in tar.getmembers():
            if member.name in expected:
                h=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                if h!=expected.pop(member.name):raise RuntimeError('Archive hash mismatch')
    assert not expected
    checksum=sha(package)
    package.with_suffix(package.suffix+'.sha256').write_text(checksum+'  '+package.name+'\n')
    delivery=dict(package=str(package),sha256=checksum,bytes=package.stat().st_size,
        payload_files=len(manifest),report=str(root/'reports/TAIL_AND_BACKGROUND_AUDIT_CN.md'),
        scope='Diagnostic and input preparation only; zero new scored background events',
        state='HOLD_MATCHED_BACKGROUND_AND_REAL_CALIBRATION_INCOMPLETE')
    js(package.with_suffix('.DELIVERY.json'),delivery)
    print(json.dumps(delivery),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
