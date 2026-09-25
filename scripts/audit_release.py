"""Release provenance, dependency, secret-pattern and license inventory."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import re
import subprocess
import sys

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    a = ap.parse_args()
    R = a.release
    import numpy as np
    import pandas as pd
    base = R/'runtime/completion/tables'
    draw = pd.read_csv(base/'catalog190_all_draws.csv')
    metrics = ['macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_50',
               'average_precision','false_at_recall_0p5','false_at_recall_0p9','roc_auc']
    groups = ['run','arm','split','method']
    per_model = draw.groupby(groups+['seed'])[metrics].mean()
    expected = pd.read_csv(base/'retrieval_summary_190.csv').set_index(groups)
    checks = []
    for stat in ('mean','std'):
        got = getattr(per_model.groupby(level=groups), stat)()
        for name in metrics:
            x = got[name].reindex(expected.index).to_numpy()
            y = expected[name+'_'+stat].to_numpy()
            checks.append(dict(stat=stat, metric=name, max_error=float(abs(x-y).max()),
                passed=bool(np.allclose(x,y,atol=1e-10,rtol=1e-12))))
    save(R/'verification/AGGREGATION_500_SUBSETS.json', dict(
        rows=len(draw), draws_per_run_seed_method=draw.groupby(groups+['seed']).draw.nunique().unique().tolist(),
        checks=checks, all_pass=all(x['passed'] for x in checks),
        pair_scores_not_recomputed_for_all_500_subsets=True))
    speed = next((R/'speed').iterdir())
    speed_checks = []
    paper = R/'uploaded_snapshot/paper_snapshot'
    for name in ('runtime_summary.csv','scaling_timings.csv'):
        x,y = speed/'tables'/name, paper/'source_data/runtime_comparison'/name
        speed_checks.append(dict(file=name, source_exists=x.exists(), paper_exists=y.exists(),
                                 byte_identical=sha(x)==sha(y) if x.exists() and y.exists() else False))
    save(R/'verification/SPEED_SOURCE_MAPPING.json', dict(checks=speed_checks,
        timings_rerun=False, note='Existing DOMAIN04 measurements preserved, not newly timed.'))
    supplement = R/'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT'
    records = json.loads((supplement/'inputs/RAW_INPUT_ACQUISITION.json').read_text())
    missing = [r['input_id'] for r in records if not Path(r['original_path']).is_file()]
    save(R/'audit/RAW_INPUT_AVAILABILITY.json', dict(records=len(records), missing=missing,
        complete_fresh_download_test=False, urls_live_rechecked=False,
        historical_input_hashes_preserved=True,
        acquisition_manifest=str((supplement/'inputs/RAW_INPUT_ACQUISITION.json').relative_to(R))))
    pkgs = []
    for d in sorted(md.distributions(), key=lambda x:x.metadata.get('Name','').lower()):
        pkgs.append(dict(name=d.metadata.get('Name'), version=d.version,
            license_expression=d.metadata.get('License-Expression'),
            license_text=d.metadata.get('License'), home_page=d.metadata.get('Home-page'),
            classifiers=[v for v in d.metadata.get_all('Classifier',[]) if v.startswith('License ::')]))
    save(R/'audit/INSTALLED_DEPENDENCIES_AND_LICENSE_METADATA.json', pkgs)
    proc = subprocess.run([sys.executable,'-m','pip','check'], capture_output=True, text=True)
    save(R/'audit/ENVIRONMENT_CHECK.json', dict(python=sys.executable,
        pip_check_returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        environment_recreated_this_run=Path(sys.executable).is_relative_to(R/'verification/fresh_env'),
        environment='hash-locked isolated environment; see fresh_environment_install for new installation evidence'))
    patterns = {
        'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
        'credential_assignment': re.compile(r'''(?i)\b[A-Za-z_]*(?:password|passwd|access_token|api_token|secret_key|zenodo_token|ssh_pass|github_token|auth_token)[A-Za-z_]*["']?\s*[=:]\s*["']([^"'\n]{6,})["']'''),
        'url_credentials': re.compile(r'https?://[^\s/:@]+:[^\s/@]{6,}@'),
    }
    findings, scanned, excluded_files = [], 0, []
    text_ext = {'.py','.md','.txt','.json','.csv','.yaml','.yml','.sh','.toml','.tex','.cfg','.ini','.log','.cff',
                '.html','.ipynb','.rst','.lock','.c','.h','.js','.ts','.css'}
    for prefix in ('runtime','uploaded_snapshot','speed','supplement','scripts',
                   'audit','contracts','reports','logs','fixtures','README.md'):
        source = R/prefix
        for p in ([source] if source.is_file() else source.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.suffix.lower() not in text_ext:
                continue
            if p.stat().st_size > 64*2**20:
                excluded_files.append(str(p.relative_to(R)))
                continue
            scanned += 1
            for number, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                for kind, regex in patterns.items():
                    found = regex.search(line)
                    if found:
                        # Do not print or store the matched secret or surrounding line.
                        findings.append(dict(path=str(p.relative_to(R)), line=number, pattern=kind,
                            matched_value_sha256=hashlib.sha256(found.group(0).encode()).hexdigest()))
    reviewed = []
    for hit in findings:
        hit['classification'] = 'UNREVIEWED'
        if hit['pattern'] == 'private_key':
            p = R/hit['path']
            if p.suffix == '.py':
                for node in ast.walk(ast.parse(p.read_text())):
                    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and node.lineno == hit['line']:
                        value = node.value.decode() if isinstance(node.value, bytes) else node.value
                        if re.fullmatch(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', value):
                            hit['classification'] = 'SCANNER_HEADER_LITERAL_NOT_A_KEY'
        reviewed.append(hit)
    findings = reviewed
    save(R/'audit/SECRET_PATTERN_SCAN.json', dict(scanned_text_files=scanned, findings=findings,
        excluded_large_files=excluded_files, archive_and_binary_contents_scanned=False,
        full_security_certification=False, secrets_not_echoed=True))
    excluded = sorted(set(x['path'] for x in findings if x['classification']=='UNREVIEWED'))
    save(R/'contracts/PUBLICATION_QUARANTINE.json', dict(paths=excluded,
        rule='Unreviewed pattern-hit text files excluded from new release tarballs; exact scanner header literals are not keys.'))
    models = []
    for p in sorted((R/'runtime/training/arms').rglob('*.pt')):
        models.append(dict(path=str(p.relative_to(R)), bytes=p.stat().st_size, sha256=sha(p)))
    save(R/'contracts/MODEL_ARTIFACTS.json', models)
    protected = json.loads((R/'audit/PROTECTED_INPUTS.json').read_text())
    changed = [r['path'] for r in protected if sha(r['path']) != r['sha256']]
    save(R/'audit/PROTECTED_INPUTS_POSTCHECK.json', dict(files=len(protected), changed=changed,
        pass_unchanged=not changed))
    print(json.dumps(dict(aggregation_pass=all(x['passed'] for x in checks), input_records=len(records),
        missing=len(missing), models=len(models), secret_pattern_hits=len(findings),
        quarantine_files=len(excluded), protected_changed=len(changed)), ensure_ascii=False))

if __name__ == '__main__':
    main()
