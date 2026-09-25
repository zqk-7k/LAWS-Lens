"""Read-only staged replay diagnostic; never change calibration or archive."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'results/trilens_scaling_20260917_10to1891_r2/scripts'))
import trilens_scale_20260917 as run


def main(root):
    run.ROOT = root
    try:
        chosen, rows, recipes = run.init()
        shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
        run.js(root/'contracts/REPLAY_AUDIT_SCOPE.json', dict(
            purpose='Audit all62 freshly prepared events against frozen archive, then recompute subset46 from the same event features.',
            not_speed_measurement=True, no_calibration_changes=True, no_archive_writes=True))
        dest = root/'results/full62'
        dest.mkdir()
        cache = run.prepare_trilens(chosen, rows, recipes, 62, dest)
        for recipe in recipes:
            seed, slot = recipe['seed'], recipe['slot']
            c = cache['models'][seed]
            run.NP.savez_compressed(dest/f'event_predictions_{seed}.npz', **{
                key: value for key, value in c.items() if hasattr(value, 'shape')},
                **{'joint_'+key:value for key,value in c['joint'].items()})
        full = run.score_trilens(cache, recipes, dest, 0)
        checks = []
        for frame, recipe in zip(full['frames'], recipes):
            seed = recipe['seed']
            old = run.PD.read_parquet(run.B.ARCHIVE/f'development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/gwtc3/seed_{seed}/real_fusion_pairs.parquet').set_index('pair_key').loc[frame.pair_key]
            difference = frame.waveform_score.to_numpy()-old.upstream_joint_waveform.to_numpy()
            checks.append(dict(seed=seed, max_abs=float(abs(difference).max()), errors=int((abs(difference)>2e-4).sum())))
            frame.assign(delta_archive=difference).to_csv(dest/f'comparison_{seed}.csv', index=False)
        run.js(dest/'ALL62_REPLAY_CHECK.json', checks)
        dest = root/'results/subset46'
        dest.mkdir()
        subset = dict(rows=cache['rows'].iloc[:46], full=cache['full'][:46], maps=cache['maps'][:46], time=cache['time'], models={})
        for seed, c in cache['models'].items():
            subset['models'][seed] = {key:(value[:46] if hasattr(value,'shape') and key!='prior' else value) for key,value in c.items()}
            subset['models'][seed]['joint'] = {key:value[:46] for key,value in c['joint'].items()}
        run.score_trilens(subset, recipes, dest, 0)
        run.js(root/'contracts/AUDIT_COMPLETE.json', dict(complete=True, results=checks))
    except BaseException:
        (root/'logs/FAIL.txt').write_text(traceback.format_exc())
        raise
    finally:
        if root.exists():
            run.finish(False, 'READ_ONLY_REPLAY_AUDIT_NOT_TIMING_ACCEPTANCE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
