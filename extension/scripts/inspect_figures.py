"""Inspect frozen table schemas and figure labels without modifying inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--paper', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
names = ['fig2_density_curves.csv','channel_distributions.csv','roc_per_model.csv','recall_per_model.csv',
         'retrieval_metrics_per_seed.csv','sky_area_before_after_summary.csv','sky_area_before_after_events.csv',
         'test_events.csv','real_PE_official_budget_summary.csv','fig5_O3_top10.csv','fig5_O4a_top10.csv',
         'fig5_rank1_PE_morphology_guides.csv','fig5_O4a_rank1_waveforms.csv']
records = []
for name in names:
    path = a.paper / 'source_data/c_current' / name
    frame = pd.read_csv(path)
    records.append({'file': name, 'rows': len(frame), 'columns': frame.columns.tolist(),
                    'head': frame.head(2).to_dict('records')})
(a.output / 'TABLE_SCHEMAS.json').write_text(json.dumps(records, indent=2, default=str))
texts = {}
for path in sorted((a.paper / 'figures').glob('*.pdf')):
    if path.name.startswith(('fig_supp_', 'fig2_', 'fig_retrieval', 'fig4_current', 'fig_runtime')):
        proc = subprocess.run(['pdftotext','-layout',str(path),'-'], capture_output=True, text=True)
        texts[path.name] = proc.stdout
(a.output / 'FIGURE_TEXT.json').write_text(json.dumps(texts, indent=2))
print(json.dumps(records, indent=2, default=str))
