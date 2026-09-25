"""Create an isolated C experiment. Historical files are copy/read only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

BASE = Path('/root/autodl-tmp/gw-catalog/results/gwlr_unified_snr_ab_20260917T113500Z_r4')
COMPLETION = BASE/'completion_20260918T014623Z'


def install(root):
    if root.exists():
        raise RuntimeError('Independent output root must not exist')
    root.mkdir(parents=True)
    for part in ('contracts/tasks', 'plans', 'scripts', 'data', 'bank', 'logs', 'maps',
                 'timings', 'reports', 'manifests', 'package', 'completion/contracts/tasks',
                 'completion/logs', 'completion/scripts', 'completion/tables', 'completion/reports',
                 'completion/manifests', 'completion/figures'):
        (root/part).mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE/'scripts', root/'scripts', dirs_exist_ok=True)
    shutil.copy2(BASE/'scripts/unified_ab.py', root/'scripts/legacy_unified_ab.py')
    # Snapshot algorithms; C entry points supplied alongside this installer take precedence.
    for p in COMPLETION.joinpath('scripts').glob('*.py'):
        shutil.copy2(p, root/'completion/scripts'/p.name)
    shutil.copy2(COMPLETION/'scripts/uab_completion.py', root/'completion/scripts/legacy_completion.py')
    shutil.copy2(COMPLETION/'scripts/uab_completion.py', root/'scripts/legacy_completion.py')
    supplied = Path(__file__).parent
    for p in supplied.glob('*.py'):
        if p.name != 'legacy_unified_ab.py':
            shutil.copy2(p, root/'scripts'/p.name)
    for name in ('uab_completion.py',):
        shutil.copy2(supplied/name, root/'completion/scripts'/name)
    import sys
    sys.path.insert(0, str(root/'scripts'))
    import unified_ab as u
    shutil.copy2(BASE/'plans/source_population.parquet', root/'plans/source_population.parquet')
    shutil.copy2(BASE/'plans/catalog190_subsets.parquet', root/'plans/catalog190_subsets.parquet')
    pilot = []
    for run in u.RUNS:
        path = root/'plans'/run
        shutil.copytree(BASE/'plans'/run, path, ignore=shutil.ignore_patterns('*.npy'))
        for name in ('noise_reference_bank.npy', 'noise_psd_bank.npy', 'noise_psd_frequency.npy'):
            (path/name).symlink_to(BASE/'plans'/run/name)
        events = pd.read_parquet(path/'event_plan.parquet')
        events = events.drop(columns=[c for c in events if c.startswith('snr_')])
        events['noise_offset_samples'] = [u.stable('C64-offset', run, s, int(i), int(v)) % (192*4096+1)
            for s, i, v in zip(events.source_uid, events.image_number, events.variant)]
        events.to_parquet(path/'event_plan.parquet', index=False)
        sources = pd.read_parquet(path/'sources.parquet')
        training = sources[(sources.role == 'main') & (sources.split == 'train')]
        # Six parents per family, spanning mass in deterministic source order.
        for family, group in training.groupby('family'):
            group = group.sort_values(['mc_det', 'source_uid'])
            positions = np.linspace(0, len(group)-1, 6).round().astype(int)
            for row in group.iloc[positions].to_dict('records'):
                pilot.append(dict(run=run, source_uid=row['source_uid'], family=family))
        if sources.groupby('global_source_id').split.nunique().max() != 1:
            raise RuntimeError('Source/environment leakage')
        noise = pd.read_parquet(path/'noise_plan.parquet')
        if noise.groupby('parent_uid').split.nunique().max() != 1:
            raise RuntimeError('Noise-parent leakage')
    pd.DataFrame(pilot).to_parquet(root/'plans/C_PILOT_SOURCES.parquet', index=False)
    shutil.copytree(BASE/'shared_time', root/'shared_time')
    contract = dict(code='GWLR-UC-01', created_utc=u.now(), baseline=str(BASE),
        runs=list(u.RUNS), arms=list(u.ARMS), model_seeds=list(u.SEEDS),
        method='NEW-SCORE-ONLY-POSITIVE-CANDIDATE', no_outer_score_mixing=True,
        analysis_nside=512, main_validation_events_per_run=450, main_test_events_per_run=450,
        native_map_count=2700, model_components=45, size_control_events=190, size_control_draws=500,
        training_views_per_run=49568, total_waveform_views=154476,
        main_parent_sources_per_run=1800, auxiliary_train_parents_per_run=4096,
        auxiliary_validation_parents_per_run=512,
        data_reuse='A/B parent sources and noise blocks, already explored; not a new blind test',
        C_amplitude='one scalar shared by both images and every noise view of each physical source',
        distance_proposal='uniform Euclidean volume conditional on weakest reference-noise optimal SNR in [8,40]',
        brighter_image_upper_SNR_cap=None, no_individual_image_target_SNR=True,
        preserves_image_amplitude_ratio_after_detector_response=True,
        not_full_GW_LMC_cosmological_population=True,
        trigger_method='known catalogue time +/-0.25s; independent 8192-template IMRPhenomD bank; fixed top8 16-bin chi-square consistency rerank',
        waveform_injection='IMRPhenomXPHM; 64 seconds raw at4096Hz; same raw data feed waveform and trigger paths',
        truth_used_for_trigger_template=False, synthetic_Gaussian_trigger_errors=False,
        full_BBH_PE=False, blind_GW_search=False,
        pilot=dict(source_parents=54, events=90, train_only=True,
            numerical_relative_tolerance=1e-6, native_map_normalization_tolerance=1e-5,
            raw_HPD90_minimum_per_run=.5,
            minimum_is_gross_failure_screen_not_posterior_calibration=True),
        official_and_real_PE_used_for_model_selection=False,
        test_generation_after_FINAL_SCORE_FREEZE=True,
        sky_temperature='validation-only source-weighted frozen grid from A/B',
        time='identical frozen training-group 1D lookup from A/B, no 2D intensity channel',
        training='same five component architectures and optimizer settings, retrain on C',
        disk_safety_gib=25, dense_maps_persist=False,
        final_state='HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE')
    u.write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    spec = json.loads((BASE/'contracts/SCORING_SPEC.json').read_text())
    spec['experiment'] = 'GWLR-UC-01'
    spec['scope'] = 'Same C method and rules across O3/O4a/O4b; fit parameters run-specific'
    spec['acceptance'] = 'Deliver C regardless of improvement; report limitations; no automatic adoption'
    spec['sky']['trigger_inputs'] = 'matched-filter complex SNR recovered from the same raw noisy strain'
    u.write(root/'contracts/SCORING_SPEC.json', spec)
    u.write(root/'contracts/SCORING_SPEC_FREEZE.json', {'sha256': u.sha(root/'contracts/SCORING_SPEC.json')})
    protected = json.loads((BASE.parent/'gwlr_c_phys_e2e_timing_20260918T100541Z/contracts/PROTECTED_INPUTS.json').read_text())
    for path, digest in protected.items():
        if u.sha(path) != digest:
            raise RuntimeError('Historical baseline changed')
    u.write(root/'contracts/PROTECTED_INPUTS.json', protected)
    files = [p for p in (root/'plans').rglob('*') if p.is_file()]
    files += list((root/'scripts').rglob('*.py'))
    files += [root/'contracts/ANALYSIS_CONTRACT.json', root/'contracts/SCORING_SPEC.json']
    u.write(root/'contracts/PLAN_FREEZE.json', {str(p.relative_to(root)): u.sha(p) for p in files})
    u.write(root/'RUN_STATUS.json', dict(state='C_PLAN_FROZEN', utc=u.now(), complete_results=False))
    (root/'reports/START_README_CN.md').write_text(
        '# GWLR-UC-01\n\n独立 C 实验，不覆盖 A/B、ET 或论文。\n\n'
        '两个修改：每个源仅一个共同振幅缩放；BAYESTAR 的触发量来自与波形相同的带噪应变。'
        '仍是受控可探测源实验，不是完整宇宙学透镜总体。每运行期验证、测试各450事件；'
        '保留190事件规模控制。先做训练侧90事件端到端检查，数值或明显定位检查失败即HOLD。'
        '通过后重训45个波形组件、仅用validation校准，冻结后评估test，再做真实PE与官方阶段审计。'
        '当前没有新的Recall或真实排名结果。\n', encoding='utf-8')
    print(json.dumps(dict(root=str(root), state='C_PLAN_FROZEN', pilot_events=90)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    install(p.parse_args().root)
