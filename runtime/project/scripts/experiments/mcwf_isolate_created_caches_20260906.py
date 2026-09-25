#!/usr/bin/env python3
"""Relocate only new pilot cross-seed caches, retaining protected old files."""
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_unified_waveform_20260906 as u

ROOT=dev.PROJECT/'results/mcwf_unified_mass_ensemble_diagnostic_20260906'
created=datetime.fromisoformat(json.loads((ROOT/'contracts/UNIFIED_METHOD.json').read_text())['created_utc']).timestamp()
protected=pd.read_csv(ROOT/'manifest/PROTECTED_INPUT_SHA256.csv')
protected_paths=set(protected.path)
rows=[]
for dep in u.DEPS:
    for ms in body.MODEL_SEEDS:
        for es in dev.SEEDS:
            if body.MODEL_SEEDS.index(ms)==dev.SEEDS.index(es):
                continue
            src=u.PREV/f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
            if not src.exists():
                continue
            for path in sorted(src.iterdir()):
                if not path.is_file() or path.stat().st_mtime<created or str(path) in protected_paths:
                    rows.append({'path':str(path),'action':'left_untouched_not_proven_new'})
                    continue
                dest=ROOT/'predictor_cache'/path.relative_to(u.PREV)
                if dest.exists():
                    raise RuntimeError('Destination exists; refuse overwrite')
                dest.parent.mkdir(parents=True,exist_ok=True)
                digest=dev.sha(path)
                path.rename(dest)
                if dev.sha(dest)!=digest:
                    raise RuntimeError('Cache relocation hash mismatch')
                rows.append({'path':str(path),'destination':str(dest),'sha256':digest,'action':'relocated_new_cross_seed_cache'})
            if not any(src.iterdir()):
                src.rmdir()
dev.csv_write(ROOT/'manifest/NEW_CACHE_RELOCATION.csv',pd.DataFrame(rows))
fail=[]
for row in protected.itertuples():
    if dev.sha(Path(row.path))!=row.sha256:
        fail.append(row.path)
dev.json_write(ROOT/'contracts/PROTECTED_INPUT_RECHECK.json',{'files':len(protected),'hash_failures':fail,'new_cache_relocation_count':sum(r['action'].startswith('relocated') for r in rows)})
if fail:
    raise RuntimeError('Protected input changed')
print(json.dumps({'protected_files':len(protected),'hash_failures':0,'cache_moves':sum(r['action'].startswith('relocated') for r in rows)}))
