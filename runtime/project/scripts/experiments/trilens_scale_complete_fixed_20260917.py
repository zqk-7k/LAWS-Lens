"""Complete TriLens sizes after a recorded, unchanged Phazap support failure."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import traceback

P=Path('/root/autodl-tmp/gw-catalog')
FAILED=P/'results/trilens_scaling_20260917_10to1891_r2'
sys.path.insert(0,str(P/'scripts/experiments'))
import trilens_scale_fixed_replay_20260917 as run


def main(root):
    run.ROOT=root
    done=False
    error=None
    try:
        chosen,rows,recipes=run.init()
        shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
        (root/'reference_runs').mkdir()
        shutil.copy2(FAILED.parent/(FAILED.name+'_deliverables.tar.gz'),root/'reference_runs')
        shutil.copy2(FAILED/'tables/scaling_timings.csv',root/'reference_runs/r2_scaling_timings.csv')
        for name in ('selected_events.csv','FINAL_STATUS.json'):
            shutil.copy2(FAILED/'contracts'/name,root/'reference_runs'/('r2_'+name))
        audit=P/'results/trilens_scaling_20260917_10to1891_r2_nan_audit'
        shutil.copytree(audit,root/'support_audit')
        shutil.copytree(Path(str(audit)+'_full'),root/'support_audit_full')
        failed_replay=Path(str(FAILED)+'_trilens_complete')
        replay=root/'reference_runs/replay_failure'
        replay.mkdir()
        for name in ('contracts/FINAL_STATUS.json','results/checks.json','logs/FAIL.txt','tables/scaling_timings.csv'):
            shutil.copy2(failed_replay/name,replay/Path(name).name)
        feature_audit=Path(str(FAILED)+'_feature_audit')
        shutil.copy2(feature_audit/'tables/FEATURE_REPLAY.csv',replay/'FEATURE_REPLAY.csv')
        run.js(root/'contracts/EXECUTION_SCOPE_ADDENDUM.json',dict(
            code='TRILENS-SCALE-03-T-REPLAYFIX-PHAZAP-HOLD',
            frozen_before_these_timings=True,
            code_sha256=run.digest(Path(__file__)), inherited_contract_sha256=run.digest(root/'contracts/CONTRACT.json'),
            reason='Unmodified Phazap returned nonfinite 100 Hz phase evolution for GW200322_091133; all prefixes >=15 contain this event.',
            continue_only='TriLens sizes 5,15,46,62; no waveform/time/sky/weight changes',
            comparator='Retain r2 measured 5-event/10-pair Phazap point. >=15 not a successful timing comparison.',
            no_samples_removed=True,no_events_replaced=True,no_frequencies_changed=True,
            replay_adapter_fix='Fine-mass feature must receive full[:,:,-4096:] and normalize internally, matching archived mcwf_fine_mass_features_20260907.deployment. Prior wrapper incorrectly passed already normalized make_window_view. No weights or calibration changed.',
            replay_tolerance_unchanged=True,
            scientific_status=run.FINAL))
        for n in run.SIZES:
            run.guard()
            run.run_trilens(chosen,rows,recipes,n)
        done=True
    except BaseException as exc:
        error=repr(exc)
        if root.exists():(root/'logs/FAIL.txt').write_text(traceback.format_exc())
        raise
    finally:
        if root.exists():
            reason=error or 'HOLD_PHAZAP_ZERO_SIGNAL_PHASE_SUPPORT'
            run.finish(False,reason)
            p=root/'contracts/FINAL_STATUS.json'
            f=json.loads(p.read_text());f['trilens_all_scales_complete']=done;f['full_comparison_complete']=False
            run.js(p,f)
            run.status('TRILENS_COMPLETE_PHAZAP_HOLD' if done else 'HOLD_FAILURE',error=reason)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
