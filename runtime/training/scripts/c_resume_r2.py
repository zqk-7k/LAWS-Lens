"""Technical repair overlay; preserves the original frozen C scripts and logs."""
import argparse
import json
from pathlib import Path
import shutil
import time

import c_controller as controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', default='run')
    args = parser.parse_args()
    root = args.root
    controller.u.verify(root)
    for relative, digest in json.loads((root/'contracts/TECHNICAL_REPAIR_R2.json').read_text())['new_files'].items():
        if controller.u.sha(root/relative) != digest:
            raise RuntimeError('Repair artifact changed: '+relative)
    original = controller.task

    def repaired_task(root, label, script, *params):
        if script == 'train_components.py':
            script = 'train_components_r2.py'
        receipt = root/'contracts/tasks'/f'{label}.json'
        if receipt.exists() and json.loads(receipt.read_text())['exit_code'] != 0:
            archive = root/'contracts/failed_attempts'/f'{label}_{time.time_ns()}.json'
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(receipt, archive)
        return original(root, label, script, *params)

    controller.task = repaired_task
    controller.main()


if __name__ == '__main__':
    main()
