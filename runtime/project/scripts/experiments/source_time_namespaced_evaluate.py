#!/usr/bin/env python3
"""Shared time evaluation with the corrected population namespace contract."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, '/root/autodl-tmp/gw-catalog/scripts/experiments')
import mcwf_source_time_evaluate_20260908 as implementation
import mcwf_source_time_namespaced_20260908 as science


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['evaluate', 'real', 'assess'], required=True)
    a = p.parse_args()
    implementation.science = science
    implementation.ev.scored = implementation.scored
    snapshot = a.root / 'scripts/source_time_namespaced_evaluate.py'
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
        shutil.copy2(implementation.__file__, a.root / 'scripts/frozen_time_evaluate_implementation.py')
    if a.stage == 'evaluate':
        implementation.evaluate(a.root)
    else:
        getattr(implementation.ev, a.stage)(a.root)
    if implementation.AUDIT:
        implementation.dev.csv_write(a.root / f'tables/EXPLICIT_CHANNEL_CHANGE_{a.stage}.csv',
                                     implementation.pd.DataFrame(implementation.AUDIT))
