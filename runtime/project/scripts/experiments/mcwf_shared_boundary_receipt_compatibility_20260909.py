#!/usr/bin/env python3
"""Add the legacy reader receipt for the already frozen R55 configuration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def main(root):
    config = root/'configs/SELECTED_CONFIGURATIONS.json'
    frozen = json.loads((root/'contracts/START_FREEZE.json').read_text())
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if digest != frozen['config_sha256']:
        raise RuntimeError('Original pre-evaluation hash mismatch')
    if list((root/'results').rglob('*.parquet')):
        raise RuntimeError('Expected failure before any new pair scoring')
    stamp = datetime.now(timezone.utc).isoformat()
    with (root/'contracts/CONFIGURATIONS_FROZEN.json').open('x') as stream:
        json.dump({'UTC':stamp,'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':digest,
            'original_freeze_UTC':frozen['UTC'],'compatibility_only':True,
            'no_refit_or_config_change':True},stream,indent=2)
    with (root/'audit/CONFIG_RECEIPT_COMPATIBILITY.json').open('x') as stream:
        json.dump({'UTC':stamp,'before_any_scoring':True,'original_hash_exact':True,
            'cause':'Legacy selections() expects a separate CONFIGURATIONS_FROZEN receipt.',
            'scientific_configuration_changed':False,'original_failure_log_preserved':True},stream,indent=2)
    shutil.copy2(__file__,root/'scripts/shared_boundary_receipt_compatibility.py')
    print('EXISTING_FREEZE_RECEIPT_ADDED',digest)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
