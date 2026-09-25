"""Frozen, candidate-independent tail feasibility and background input audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto

BASE = Path('/root/autodl-tmp/gw-catalog/results/gwlr_unified_c_physical_20260918T134500Z_r1')
FPP = BASE.parent/'gwlr_fpp_injection_real_01_20260922T061000Z_r2'
OFFICIAL = BASE.parent/'gwlr_null_background_01_20260922T040000Z'
RUNS = ('O3', 'O4a', 'O4b')
SEEDS = (2026091721, 2026091722, 2026091723)
QUANTILES = (.9, .95, .975)
TRACKED = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def track(p):
    p = Path(p)
    digest = sha(p)
    if str(p) in TRACKED and TRACKED[str(p)] != digest:
        raise RuntimeError('Input changed: '+str(p))
    TRACKED[str(p)] = digest
    return p


def js(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def csv(p, f):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise RuntimeError('No overwrite: '+str(p))
    f.to_csv(p, index=False, encoding='utf-8-sig')


def fit_tail(scores, threshold, minimum=50):
    x = np.asarray(scores, float)
    if not np.isfinite(x).all() or not np.isfinite(threshold):
        raise ValueError('Nonfinite score')
    y = x[x > threshold] - threshold
    if len(y) < minimum or len(np.unique(y)) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        shape, loc, scale = genpareto.fit(y, floc=0.)
    if not np.isfinite([shape, scale]).all() or scale <= 0:
        return None
    if not np.isfinite(genpareto.logpdf(y, shape, loc=0, scale=scale)).all():
        return None
    return dict(threshold=float(threshold), shape=float(shape), scale=float(scale),
                tail_fraction=float(len(y)/len(x)), tail_n=int(len(y)), n=int(len(x)),
                endpoint=float(threshold-scale/shape) if shape < 0 else None)


def survival(model, values):
    values = np.atleast_1d(values).astype(float)
    if np.any(values < model['threshold']):
        raise ValueError('Below fitted tail threshold')
    return model['tail_fraction']*genpareto.sf(values-model['threshold'],
                                            model['shape'], scale=model['scale'])


def frozen_contract(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('contracts', 'inputs', 'results', 'reports', 'scripts', 'figures', 'logs', 'manifests'):
        (out/name).mkdir()
    for name in ('tail_audit.py', 'test_tail_audit.py'):
        p = Path(__file__).parent/name
        shutil.copy2(p, out/'scripts'/p.name)
    contract = dict(code='GWLR-TAIL-02', utc=datetime.now(timezone.utc).isoformat(),
        baseline=str(BASE), original_fpp=str(FPP), runs=list(RUNS), models=[str(s) for s in SEEDS]+['mean_S'],
        score='Frozen C final_score_POSITIVE; mean_S is mean of scores, not consensus rank',
        fit='C validation singleton pairs only', validation='C test singleton pairs only; already examined historically, not fresh blind validation',
        thresholds=list(QUANTILES), primary_threshold=.95, no_threshold_selection_using_real_candidates=True,
        minimum_fit_tail_pairs=50, minimum_distinct_tail_sources=30, minimum_distinct_tail_noise_parents=5,
        checks=dict(shape_range_max=.5, threshold_survival_ratio_max=3.,
                    deletion_survival_ratio_max=3., heldout_observed_predicted_ratio=[.5, 2.],
                    heldout_min_exceedances=10),
        diagnostic_score_grid='fit background quantiles .98,.99,.995; never real candidate scores',
        deletions=['maximum pair', 'every source node', 'every noise-parent block'],
        bootstrap=dict(draws=200, unit='source-node and noise-node multinomial weights, separately',
            interpretation='conditional dyadic resampling sensitivity, not calibrated population confidence interval'),
        publication_requires=dict(fresh_source_noise_disjoint_validation=True,
            selected_population_and_sky_pipeline_transfer_validated=True,
            at_least_20_independent_noise_parents_each_split=True,
            numerical_and_predictive_checks_pass=True),
        gates_are_prespecified_operational_screens_not_theorems=True,
        no_candidate_extrapolation_on_failure=True, empirical_counts_never_replaced=True,
        no_retraining=True, no_rank_changes=True, no_annual_FAR=True,
        importance_sampling='Not enabled until a normalized joint proposal and exact weights are available',
        references=['https://doi.org/10.1111/j.2517-6161.1990.tb01796.x',
                    'https://arxiv.org/html/2512.16347v3'])
    js(out/'contracts/TAIL_CONTRACT.json', contract)
    js(out/'contracts/TAIL_CONTRACT_FREEZE.json', dict(sha256=sha(out/'contracts/TAIL_CONTRACT.json')))


def load_null(run, split):
    frames = []
    for seed in SEEDS:
        p = BASE/f'completion/deployments/{run}/C_PHYSICAL/evaluation/{split}/seed_{seed}'
        receipt = json.loads(track(p/'COMPLETE.json').read_text())
        expected = {r['path']: r['sha256'] for r in receipt['files']}
        for name in ('events.parquet', 'all_pair_scores.parquet'):
            path = track(p/name)
            if sha(path) != expected[str(path)]:
                raise RuntimeError('Historical receipt mismatch')
        e = pd.read_parquet(p/'events.parquet')
        f = pd.read_parquet(p/'all_pair_scores.parquet')
        f['pair_key'] = ['--'.join(sorted((str(a),str(b)))) for a,b in zip(f.event_i,f.event_j)]
        if f.pair_key.duplicated().any():
            raise RuntimeError('Duplicate pair')
        chosen = e.family.eq('unlensed').to_numpy()
        f = f[chosen[f.idx_i] & chosen[f.idx_j]].copy()
        if f.is_true_pair.any() or len(f) != 4005:
            raise RuntimeError('Invalid null selection')
        if frames and not frames[0].pair_key.equals(f.pair_key):
            raise RuntimeError('Different model pair order')
        if frames and not events.equals(e):
            # Some per-model metadata may differ; physical IDs must not.
            fields = ['event_uid','source_uid','global_source_id','noise_parent_uid']
            if not events[fields].equals(e[fields]):
                raise RuntimeError('Different physical events across seeds')
        events = e
        frames.append(f)
    null = events[events.family.eq('unlensed')].copy().reset_index(drop=True)
    index = {s:i for i,s in enumerate(null.event_uid)}
    f = frames[0]
    i = np.array([index[x] for x in f.event_i]); j = np.array([index[x] for x in f.event_j])
    scores = {str(seed):a.final_score_POSITIVE.to_numpy() for seed,a in zip(SEEDS,frames)}
    scores['mean_S'] = np.mean(list(scores.values()), axis=0)
    return null, f, i, j, scores


def factor_range(a, reference=None):
    a = np.asarray(a, float)
    if not len(a) or not np.isfinite(a).all() or np.any(a <= 0):
        return np.inf
    if reference is None:
        return float(np.max(a, axis=0).__truediv__(np.min(a, axis=0)).max())
    return float(np.maximum(a/reference, reference/a).max())


def resample_tail(x, i, j, group, u, grid, rng):
    ng = int(group.max())+1
    rows = []
    for b in range(200):
        counts = rng.multinomial(ng, np.full(ng, 1/ng))
        # A node/block bootstrap of the dyadic empirical distribution; no self-pairs added.
        weights = counts[group[i]]*counts[group[j]]
        repeated = np.repeat(x, weights)
        fit = fit_tail(repeated, u) if len(repeated) else None
        rows.append(dict(draw=b, valid=fit is not None,
            **({f'p{k}':float(v) for k,v in enumerate(survival(fit,grid))} if fit else {})))
    return rows


def diagnose(out):
    fit_rows, deletion_rows, predictive_rows, bootstrap_rows, gates = [], [], [], [], []
    for run in RUNS:
        e, pairs, i, j, models = load_null(run, 'validation')
        held, hpairs, hi, hj, hmodels = load_null(run, 'test')
        for field in ('event_uid','source_uid','global_source_id','noise_parent_uid'):
            if set(e[field].astype(str)) & set(held[field].astype(str)):
                raise RuntimeError('Fit/validation overlap: '+field)
        csv(out/f'inputs/{run}_fit_events.csv',e)
        csv(out/f'inputs/{run}_check_events.csv',held)
        csv(out/f'inputs/{run}_fit_scores.csv',pd.DataFrame(dict(event_i=pairs.event_i,event_j=pairs.event_j,**models)))
        csv(out/f'inputs/{run}_check_scores.csv',pd.DataFrame(dict(event_i=hpairs.event_i,event_j=hpairs.event_j,**hmodels)))
        groups, labels = pd.factorize(e.noise_parent_uid.astype(str), sort=True)
        for model,x in models.items():
            grid = np.quantile(x,[.98,.99,.995])
            fits, predictions = [], []
            for q in QUANTILES:
                u = float(np.quantile(x,q)); fit = fit_tail(x,u)
                tail = x > u
                src_count = len(set(i[tail])|set(j[tail]))
                block_count = len(set(groups[i[tail]])|set(groups[j[tail]]))
                fits.append(fit)
                row = dict(run=run, model=model,q=q,tail_sources=src_count,tail_noise_parents=block_count,
                    numerical_pass=fit is not None, tail_support_pass=src_count>=30 and block_count>=5)
                if fit:
                    row.update(fit)
                    predictions.append(survival(fit,grid))
                    for level,t in zip((.98,.99,.995),grid):
                        predicted = float(survival(fit,[t])[0]*len(hmodels[model]))
                        observed = int(np.sum(hmodels[model] >= t))
                        ratio = observed/predicted if predicted>0 else np.nan
                        predictive_rows.append(dict(run=run,model=model,q=q,fit_quantile=level,
                            threshold=float(t),observed=observed,predicted=predicted,ratio=ratio,
                            predictive_screen_pass=observed>=10 and .5<=ratio<=2.,
                            not_iid_binomial_test=True))
                fit_rows.append(row)
            primary = fits[1]
            deletion_predictions = []
            if primary:
                u = primary['threshold']
                masks = [('max_pair','max', np.arange(len(x)) != np.argmax(x))]
                masks += [('source',str(k),(i!=k)&(j!=k)) for k in range(len(e))]
                masks += [('noise',str(lab),(groups[i]!=k)&(groups[j]!=k)) for k,lab in enumerate(labels)]
                for unit,name,mask in masks:
                    fit = fit_tail(x[mask],u)
                    pred = survival(fit,grid) if fit else np.full(3,np.nan)
                    deletion_predictions.append(pred)
                    deletion_rows.append(dict(run=run,model=model,unit=unit,deleted=name,
                        valid=fit is not None,**{f'p{k}':float(v) for k,v in enumerate(pred)}))
                for unit,g in [('source',np.arange(len(e))),('noise',groups)]:
                    seed = int.from_bytes(hashlib.sha256(f'{run}:{model}:{unit}:TAIL02'.encode()).digest()[:8],'little')
                    for row in resample_tail(x,i,j,g,u,grid,np.random.default_rng(seed)):
                        bootstrap_rows.append(dict(run=run,model=model,unit=unit,**row))
            localfit = [r for r in fit_rows if r['run']==run and r['model']==model]
            localpred = [r for r in predictive_rows if r['run']==run and r['model']==model]
            threshold_factor = factor_range(predictions) if len(predictions)==3 else np.inf
            deletion_factor = factor_range(deletion_predictions, survival(primary,grid)) if primary else np.inf
            shape_range = max(f['shape'] for f in fits)-min(f['shape'] for f in fits) if all(fits) else np.inf
            checks = dict(fits_and_support=all(r['numerical_pass'] and r['tail_support_pass'] for r in localfit),
                shape_stability=shape_range<=.5, threshold_stability=threshold_factor<=3.,
                deletion_stability=deletion_factor<=3.,
                heldout_prediction=len(localpred)==9 and all(r['predictive_screen_pass'] for r in localpred),
                independent_noise_support=len(labels)>=20 and held.noise_parent_uid.nunique()>=20,
                new_blind_validation=False, real_domain_transfer=False)
            gates.append(dict(run=run,model=model,**checks,
                threshold_factor=threshold_factor,deletion_factor=deletion_factor,shape_range=shape_range,
                publication_pass=all(checks.values()),fit_noise_parents=len(labels),
                check_noise_parents=int(held.noise_parent_uid.nunique())))
            print(json.dumps(dict(run=run,model=model,checks=checks)),flush=True)
    for name,rows in [('gpd_fits',fit_rows),('deletion_sensitivity',deletion_rows),
                      ('heldout_predictions',predictive_rows),('resampling_sensitivity',bootstrap_rows),('gates',gates)]:
        csv(out/f'results/{name}.csv',pd.DataFrame(rows))
    return gates


def inventory(out):
    rows, availability = [], []
    for run in RUNS:
        old = pd.read_parquet(track(BASE/f'plans/{run}/noise_plan.parquet'))
        original = (BASE.parent/'o4b_hl_bayestar_new_score_only_20260912T072746Z/workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
                    if run=='O4b' else BASE.parent/f'real_noise_injection_v5_physical_source_20260721/{dict(O3="gwtc3",O4a="gwtc4")[run]}/shared')
        p = original/('noise_reference_metadata.parquet' if run=='O4b' else 'noise_bank_manifest.csv')
        f = pd.read_parquet(track(p)) if p.suffix=='.parquet' else pd.read_csv(track(p))
        f['parent_uid'] = f['parent_raw_file_group' if run=='O4b' else 'source_event'].astype(str)
        start = next(c for c in ('reference_start_gps','start_gps') if c in f)
        oldstart = next(c for c in ('reference_start_gps','start_gps') if c in old)
        f['previously_used_parent'] = f.parent_uid.isin(old.parent_uid.astype(str))
        f['overlaps_C_noise_interval'] = [bool(np.any((float(s)<old[oldstart].to_numpy()+256)&(float(s)+256>old[oldstart].to_numpy()))) for s in f[start]]
        schedule = pd.read_csv(track(BASE/f'plans/{run}/live_schedule.csv'))
        f['in_calendar_span'] = (f[start]>=schedule.start_gps.min())&(f[start]+256<=schedule.end_gps.max())
        f['reserve_candidate'] = ~f.previously_used_parent & ~f.overlaps_C_noise_interval & f.in_calendar_span
        f['new_DQ_selection_and_parent_independence_not_yet_verified'] = True
        csv(out/f'results/{run}_noise_reserve_inventory.csv',f)
        availability.append(dict(run=run,original_blocks=len(f),original_parents=f.parent_uid.nunique(),
            C_used_parents=old.parent_uid.nunique(),unused_candidate_parents=f.loc[f.reserve_candidate,'parent_uid'].nunique(),
            unused_candidate_blocks=int(f.reserve_candidate.sum()),
            status='RESERVE_IDENTIFIED_NOT_YET_CERTIFIED_INDEPENDENT_OR_MATCHED'))
    public = json.loads(track(OFFICIAL/'inputs/official_population.json').read_text())
    for uid,event in public['population'].items():
        # Preserve the official selected sample; do not pretend it is an unconditional proposal.
        rows.append(dict(uid=uid,keys='|'.join(sorted(event.keys()))))
    csv(out/'results/official_population_keys.csv',pd.DataFrame(rows))
    shutil.copy2(OFFICIAL/'inputs/official_population.json',out/'inputs/official_O4a_population.json')
    shutil.copy2(track(OFFICIAL/'results/official_population_inventory.csv'),out/'results/official_population_inventory.csv')
    csv(out/'results/noise_reserve_summary.csv',pd.DataFrame(availability))
    return availability


def finish(out,gates,reserve,started):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fits=pd.read_csv(out/'results/gpd_fits.csv')
    for ax,run in zip(axes,RUNS):
        x=np.sort(pd.read_csv(out/f'inputs/{run}_fit_scores.csv').mean_S.to_numpy())
        ax.step(x[::-1],np.arange(1,len(x)+1)/len(x),where='post',label='Empirical fit background')
        h=np.sort(pd.read_csv(out/f'inputs/{run}_check_scores.csv').mean_S.to_numpy())
        ax.step(h[::-1],np.arange(1,len(h)+1)/len(h),where='post',alpha=.5,label='Held-out diagnostic')
        for q in QUANTILES:
            row=fits[(fits.run==run)&(fits.model=='mean_S')&(fits.q==q)].iloc[0]
            if row.numerical_pass:
                grid=np.linspace(row.threshold,max(x[-1],h[-1]),200)
                ax.plot(grid,survival(row,grid),label=f'GPD q={q} (diagnostic)')
        ax.set(yscale='log',ylim=(1e-5,.2),xlim=(np.quantile(x,.85),max(x[-1],h[-1])+.2),title=run,xlabel='Frozen mean score S',ylabel='Conditional tail fraction')
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'figures/tail_diagnostic.png',dpi=180);fig.savefig(out/'figures/tail_diagnostic.pdf');plt.close(fig)
    # Real candidate tables are consulted only after the no-publication decision.
    for run in RUNS:
        p=FPP/f'tables/{run}/real_GWTC_Top50_FPP.csv'
        f=pd.read_csv(track(p))
        f['tail_model_FPP']=np.nan
        f['tail_model_status']='NOT_PUBLISHED_GATE_NOT_PASSED'
        csv(out/f'results/{run}_real_Top50_empirical_preserved.csv',f)
    changed=[p for p,h in TRACKED.items() if sha(p)!=h]
    if changed: raise RuntimeError('Historical input changed')
    csv(out/'manifests/INPUT_SHA256.csv',pd.DataFrame([dict(path=p,sha256=h) for p,h in TRACKED.items()]))
    js(out/'contracts/INPUT_HASH_CHECK.json',dict(files=len(TRACKED),changed=changed))
    status=dict(code='GWLR-TAIL-02',state='GPD_DIAGNOSTIC_COMPLETE_MATCHED_BACKGROUND_NOT_COMPLETE_HOLD',
        elapsed_seconds=time.monotonic()-started,models_checked=len(gates),
        published_candidate_GPD_FPP=False,new_scored_background_events=0,
        FAR_status='NO_GO_EFFECTIVE_BACKGROUND_EXPOSURE_UNESTABLISHED',
        historical_scores_weights_ranks_unchanged=True,
        pending=['Matched selected null population for each run', 'Independent noise and DQ validation',
                 'Real public PE versus simulated BAYESTAR/network tail transfer',
                 'New source/noise-disjoint null generation and frozen inference'])
    js(out/'RUN_STATUS.json',status)
    text=['# GWLR-TAIL-02 尾部可行性与匹配背景准备\n',
        '本轮不修改模型、分数、权重或历史排名；不是新增真实候选显著性结论。\n',
        '## 已完成\n',
        '三个运行期、三个模型及平均分共12种统计量；固定90%、95%、97.5%尾部起点。'
        '用原validation孤立事件拟合，在source/noise-disjoint的原test孤立事件上做诊断。'
        '这些旧数据已被查看，不是新增盲测。每运行期两侧各90事件、4005相关pair、6噪声父块。\n',
        'GPD模型检验参考 Davison & Smith (1990): https://doi.org/10.1111/j.2517-6161.1990.tb01796.x 。'
        '人口驱动非透镜背景参考 LVK Appendix B: https://arxiv.org/html/2512.16347v3 。\n',
        '## 结果\n',
        '|运行期|阈值稳定|删除稳定|留出预测|可发布真实候选GPD-FPP|\n|---|---|---|---|---|']
    for r in gates:
        if r['model']=='mean_S':
            text.append(f"|{r['run']}|{r['threshold_stability']}|{r['deletion_stability']}|{r['heldout_prediction']}|否|")
    text += ['\n源和噪声节点重采样区间仅为条件敏感性，不是已验证总体置信区间。'
        '阈值、最小支持和容差是提前冻结的工程筛查规则，不是普适统计定理。'
        '没有把4005个共享事件的pair当作4005次独立伯努利试验。\n',
        '## 匹配背景尚未完成\n',
        '识别了历史C未用的噪声候选，并保存官方O4a的254事件人口元数据。'
        '未把重新抽取旧pair或复制官方元数据称作新模拟。此次新增已评分背景事件数为0。\n',
        '官方O4a样本是已选中的总体，不能直接作为O3/O4b总体，也不能假定其抽样密度或检测选择已知。'
        '当前真实目录含公开PE天空图，C注入为HL BAYESTAR。未完成这两种天空输入和网络的尾部可转移验证，'
        '因此不能声称已补齐真实目录匹配背景。\n',
        '## 不变的边界\n',
        '原始经验FPP和0/1/2次超越计数保留；候选尾部模型FPP不发布，不用GPD强行填空。'
        '年度FAR仍缺有效背景曝光。后续必须完成匹配人口、噪声和天空处理，再生成独立背景。'
        '本轮不支持透镜发现或确认。\n']
    (out/'reports/TAIL_AND_BACKGROUND_AUDIT_CN.md').write_text('\n'.join(text),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    started=time.monotonic();frozen_contract(a.out)
    reserve=inventory(a.out)
    gates=diagnose(a.out)
    finish(a.out,gates,reserve,started)
    print(json.dumps(dict(output=str(a.out),state='DIAGNOSTIC_COMPLETE_NO_CANDIDATE_GPD_PUBLICATION')),flush=True)


if __name__=='__main__':main()
