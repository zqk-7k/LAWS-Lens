"""C adapter for unchanged frozen scoring/inference algorithms."""
import importlib.util
import json
from pathlib import Path
import sys
import time

spec = importlib.util.spec_from_file_location('legacy_completion', Path(__file__).with_name('legacy_completion.py'))
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
for name in dir(legacy):
    if not name.startswith('__'):
        globals()[name] = getattr(legacy, name)


def initialize(root, out):
    global ROOT, OUT, U
    U = legacy.initialize(root, out)
    ROOT, OUT = legacy.ROOT, legacy.OUT
    return U


def audit():
    U.verify(ROOT)
    records = [json.loads(p.read_text()) for p in (ROOT/'contracts/tasks').glob('*.json')]
    models = [d for d in records if d['state'].split('_')[0] in
              ('SHORT', 'rnc', 'ordered', 'multirate', 'conditional') and d['exit_code'] == 0]
    if len(models) != 45:
        raise RuntimeError('C requires all 45 completed component tasks')
    files, rows = [], []
    for run in U.RUNS:
        p = deployment(run, 'C_PHYSICAL')
        for seed in U.SEEDS:
            suffixes = [f'models/short_encoder/seed_{seed}/validation_selected_model.pt',
                f'rankncontrast_component_v2/models/RAW-PHASE-SOURCE/gwtc5/seed_{seed}/validation_selected_model.pt',
                f'ordered_mass_predictor/models/gwtc5/seed_{seed}/validation_selected_model.pt',
                f'models/MULTIRATE/gwtc5/seed_{seed}/selected.pt',
                f'models/CONDITIONAL-ETA-CHI/gwtc5/seed_{seed}/selected.pt']
            for suffix in suffixes:
                f = (p/suffix).resolve()
                files.append(f)
                rows.append(dict(run=run, arm='C_PHYSICAL', seed=seed, path=str(f), sha256=U.sha(f)))
    pd.DataFrame(rows).to_csv(OUT/'contracts/TRAINED_MODEL_MANIFEST.csv', index=False)
    files += [ROOT/'contracts/ANALYSIS_CONTRACT.json', ROOT/'contracts/SCORING_SPEC.json',
              ROOT/'plans/source_population.parquet']
    seal(OUT/'contracts/INPUT_AUDIT.json', files, model_tasks=45,
         old_models_used=False, test_opened_for_C=False, historical_AB_data_previously_examined=True)


def sky_worker_init(root, out, run, arm, split):
    initialize(root, out)
    import c_trigger
    c_trigger.worker_init(root)
    SKY.update(run=run, arm=arm, split=split, dest=OUT/'maps'/run/arm/split)


def localize_event(row):
    import c_trigger
    path = SKY['dest']/(row['event_uid']+'.json')
    if path.exists():
        rec = json.loads(path.read_text())
        if U.sha(rec['moc_path']) != rec['moc_sha256']:
            raise RuntimeError('C native MOC changed')
        return rec
    batch = '/'.join((SKY['run'], SKY['arm'], SKY['split']))
    trigger_row = {**row, 'strain_path': row['raw_strain_path'],
        'strain_sha256': row['raw_strain_sha256'], 'start_gps': row['raw_start_gps'],
        'optimal_snr': row['optimal_network_snr']}
    rec = c_trigger.run_event((trigger_row, batch, True))
    if rec['status'] != 'PASS':
        raise RuntimeError(rec.get('traceback', 'C localization failed'))
    record = {**row, **rec}
    write(path, record)
    return record


legacy.sky_worker_init = sky_worker_init
legacy.localize_event = localize_event
legacy.audit = audit


if __name__ == '__main__':
    # Other stages are dispatched by the single C controller entry point.
    raise SystemExit('Use c_controller.py --stage complete for C')
