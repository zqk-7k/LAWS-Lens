"""Read-only search for current plotting entry points and public input routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--previous', type=Path, required=True)
P.add_argument('--output', type=Path, required=True)
a = P.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
missing = json.loads((a.previous / 'reports/MISSING_ITEMS.json').read_text())
terms = [Path(r['item']).stem for r in missing if r['category'] == 'current_figure_entrypoint']
hits = []
count = 0
for top in ('/root/autodl-tmp/gw-catalog/scripts', '/root/autodl-tmp/gw-catalog/results'):
    for folder, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules', 'site-packages', 'venv')]
        for name in files:
            path = Path(folder) / name
            if path.suffix != '.py' or path.is_symlink() or path.stat().st_size > 5 * 2**20:
                continue
            count += 1
            content = path.read_text(errors='replace')
            found = [t for t in terms if t in content]
            if found:
                hits.append({'path': str(path), 'terms': found,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(a.output / 'PLOT_ENTRY_SEARCH.json').write_text(json.dumps({'files_scanned': count, 'terms': terms, 'hits': hits}, indent=2))
manifest = a.previous / 'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/inputs/RAW_INPUT_ACQUISITION.json'
rows = json.loads(manifest.read_text())
(a.output / 'PUBLIC_TABLE_DOWNLOADS.json').write_text(json.dumps(rows[:3], indent=2))
print(json.dumps({'files_scanned': count, 'plot_matches': hits, 'public_tables': len(rows[:3])}, indent=2))
