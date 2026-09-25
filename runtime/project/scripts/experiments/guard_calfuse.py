#!/usr/bin/env python3
"""Separate, disclosed adaptive sensitivity: do not weaken old negative scores."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import calfuse as c
import mcwf_summarize_20260905 as boot


def modified(frame,spec,mode,gamma):
    old=frame.waveform_score.to_numpy(float)
    new,ood,clipped=c.apply(frame,spec)
    proposal=np.minimum(new,old) if mode=='MINIMUM' else np.where(old<0,np.minimum(new,old),new)
    result=old+gamma*(proposal-old)
    assert np.all(result[old<0]<=old[old<0]+1e-12)
    if mode=='MINIMUM':assert np.all(result<=old+1e-12)
    return result,ood,clipped


def run(root,parent):
    if root.exists():raise RuntimeError('New guard output required')
    for folder in ('contracts','configs','tables','results','reports','audit','manifest','scripts','logs'):
        (root/folder).mkdir(parents=True)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    c.dump(root/'contracts/ANALYSIS_CONTRACT.json',{
        'code':'MCWF-CALFUSE-GUARD-02','UTC':datetime.now(timezone.utc).isoformat(),'parent':str(parent),
        'adaptive_exploration_disclosure':'Specified after reviewing CALFUSE-01; NOT an original preregistered arm or independent confirmation',
        'motivation':'Probe whether weakening old waveform negative evidence explains calibration deterioration',
        'same_method_O3_O4a':True,'unchanged':'encoders,raw time/sky,old C-fixed outer weights,scope',
        'calibration':'Frozen AFFINE/JOINT models from CALFUSE-01,not refitted',
        'modes':{'MINIMUM':'old+gamma*(min(old,calibrated)-old)',
                 'NEGATIVE-PRESERVE':'old+gamma*(where(old<0,min(old,calibrated),calibrated)-old)'},
        'gamma_grid':[0,.25,.5,.75,1.],
        'choice':'same blocked simulation tune cohort, VF50/VR10 objectives; both waveform/fusion guards; exact ties prefer gamma=0 then smaller gamma',
        'PE_and_official_selection':False,'time_and_sky_weights_retuned':False,
        'interpretation':'Conservative agreement rule,not probability or independent-evidence BF',
        'target':'same both-run PE/official budgets and reused-test guard as CALFUSE-01',
        'final_status':c.STATUS})
    paths=[parent/'configs/SELECTED_CONFIGURATIONS.json',parent/'contracts/ANALYSIS_CONTRACT.json']
    paths += list((parent/'configs').glob('*_CALIBRATION.json'))
    c.csv(root/'manifest/PARENT_INPUTS.csv',[{'path':str(p),'sha256':c.dev.sha(p)} for p in paths])
    configurations=[];grid=[]
    for dep in c.DEPS:
        for seed in c.SEEDS:
            frame=c.read(dep,seed,'validation')
            plan=pd.read_csv(parent/f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv')
            tune=c.subset(frame,plan[plan.calibration_fold==1].idx)
            w=c.frozen_weights(dep,seed)
            base=c.fast_metrics(tune,c.channels(tune,tune.waveform_score)@w)
            wfbase=c.fast_metrics(tune,tune.waveform_score.to_numpy(float))
            for family in ('AFFINE','JOINT'):
                spec=json.loads((parent/f'configs/{dep}_{seed}_{family}_CALIBRATION.json').read_text())
                for mode in ('MINIMUM','NEGATIVE-PRESERVE'):
                    options=[]
                    for gamma in (0.,.25,.5,.75,1.):
                        z,_,_=modified(tune,spec,mode,gamma)
                        m=c.fast_metrics(tune,c.channels(tune,z)@w);wf=c.fast_metrics(tune,z)
                        row={'deployment':dep,'seed':seed,'family':family,'mode':mode,'gamma':gamma,
                            'guard_pass':c.guard(m,base) and c.guard(wf,wfbase),**m}
                        options.append(row);grid.append(row)
                    for policy in ('VF50','VR10'):
                        eligible=[r for r in options if r['guard_pass']]
                        assert eligible
                        def key(r):
                            if policy=='VF50':
                                return (r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1'],r['gamma'])
                            return (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'],r['gamma'])
                        win=min(eligible,key=key)
                        configurations.append({'deployment':dep,'seed':seed,'family':family,'mode':mode,
                            'method':f'{family}-{mode}-{policy}','policy':policy,'gamma':win['gamma'],
                            'calibration':spec,'weights':w.tolist()})
    c.dump(root/'configs/SELECTED_CONFIGURATIONS.json',configurations)
    c.csv(root/'tables/VALIDATION_GAMMA_GRID.csv',grid)
    c.csv(root/'tables/SELECTED_GAMMAS.csv',[{k:v for k,v in x.items() if k not in ('calibration','weights')} for x in configurations])
    c.dump(root/'contracts/CONFIGURATIONS_FROZEN.json',{'sha256':c.dev.sha(root/'configs/SELECTED_CONFIGURATIONS.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'test_and_real_not_read_by_selection':True})
    metrics=[];cis=[];checks=[]
    for conf in configurations:
        dep,seed,method=conf['deployment'],conf['seed'],conf['method']
        for split in ('validation','test'):
            f=c.read(dep,seed,split);z,ood,clip=modified(f,conf['calibration'],conf['mode'],conf['gamma'])
            score=c.channels(f,z)@np.asarray(conf['weights']);oldscore=c.channels(f,f.waveform_score)@c.frozen_weights(dep,seed)
            out=f.copy();out['OMC_baseline_waveform_score']=f.waveform_score;out['waveform_score']=z
            out['final_score']=score;out['calibration_ood']=ood;out['calibration_clipped']=clip
            dest=root/f'results/{method}/{dep}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            out.to_parquet(dest/f'{split}_pairs.parquet',index=False)
            for view,s,b in (('fusion',score,oldscore),('waveform_only',z,f.waveform_score.to_numpy(float))):
                m=c.fast_metrics(f,s);base=c.fast_metrics(f,b)
                metrics.append({'deployment':dep,'seed':seed,'method':method,'split':split,'mode':view,
                    'gamma':conf['gamma'],'guard_pass':c.guard(m,base),**m})
                if split=='test':
                    ci,_=boot.ranks_bootstrap(f,s,b,seed,10000)
                    if view=='fusion':ci.update(boot.pair_bootstrap(f,s,b,c.dev.BASE.retained_event_plan(dep,seed,'test'),seed,1000))
                    cis.append({'deployment':dep,'seed':seed,'method':method,'mode':view,**ci})
            checks.append({'deployment':dep,'seed':seed,'method':method,'split':split,
                'negative_evidence_weakened':int((z[f.waveform_score<0]>f.loc[f.waveform_score<0,'waveform_score'].to_numpy()+1e-12).sum()),
                'time_exact':bool(np.array_equal(out.time_score,f.time_score)),
                'sky_exact':bool(np.array_equal(out.sky_raw_log_bf,f.sky_raw_log_bf))})
        print('GUARD_EVALUATED',dep,seed,method,flush=True)
    c.csv(root/'tables/RETRIEVAL_PER_SEED.csv',metrics);c.csv(root/'tables/SYSTEM_BOOTSTRAP.csv',cis)
    c.csv(root/'audit/NEGATIVE_EVIDENCE_AND_FROZEN_CHANNELS.csv',checks)
    budgets=[]
    for dep in c.DEPS:
        pe=pd.read_parquet(c.NOISE/f'audit/{dep}_external_reference.parquet')
        for method in sorted(set(x['method'] for x in configurations)):
            frames={}
            for conf in [x for x in configurations if x['deployment']==dep and x['method']==method]:
                seed=conf['seed'];f=c.read(dep,seed,'real')
                z,ood,clip=modified(f,conf['calibration'],conf['mode'],conf['gamma'])
                out=f.copy();out['OMC_baseline_waveform_score']=f.waveform_score;out['waveform_score']=z
                out['calibration_ood']=ood;out['calibration_clipped']=clip;frames[seed]=out
            b=c.dev.save_evaluation(root,method,dep,frames,pe)
            budgets+=b
    c.csv(root/'tables/PE_OFFICIAL_BUDGETS.csv',budgets)
    base=pd.read_csv(parent/'tables/PE_OFFICIAL_BUDGETS.csv')
    base=base[(base.config=='OMC-FIXED')&(base.seed.astype(str)=='consensus')&(base.method=='fusion')]
    b=pd.DataFrame(budgets);b=b[(b.seed.astype(str)=='consensus')&(b.method=='C_fixed')]
    m=pd.DataFrame(metrics);gates=[];compact=[]
    for dep in c.DEPS:
        ref=base[base.deployment==dep].set_index('budget')
        for method in b.config.unique():
            q=b[(b.deployment==dep)&(b.config==method)].set_index('budget').loc[[10,20]]
            r=ref.loc[[10,20]]
            pe_ok=(q.catastrophic_mc<=r.catastrophic_mc).all()
            for col in ('BC_mc_ge_0p5','median_BC_mc','Dmax_le_3'):pe_ok &= (q[col]>=r[col]-1e-12).all()
            pe_gain=pe_ok and (q.BC_mc_ge_0p5.sum()>r.BC_mc_ge_0p5.sum() or q.median_BC_mc.sum()>r.median_BC_mc.sum()+1e-10)
            off=all((q[k]>=r[k]).all() and q[k].sum()>r[k].sum() for k in ('official_frontend','official_hanabi'))
            sims=m[(m.deployment==dep)&(m.method==method)&(m.split=='test')]
            sim=bool(sims.guard_pass.all())
            gates.append({'deployment':dep,'method':method,'PE_improved':bool(pe_gain),'official_improved':bool(off),
                'simulation_guard':sim,'target_pass':bool(pe_gain and off and sim)})
            a=sims[sims['mode']=='fusion']
            compact.append({'deployment':dep,'method':method,'R10':f'{a.macro_r_at_10.mean():.4f} +/- {a.macro_r_at_10.std():.4f}',
                'AP':a.average_precision.mean(),'F50':a.false_at_recall_0p5.mean(),'F90':a.false_at_recall_0p9.mean(),
                'Mc Top10/20':f'{q.loc[10,"BC_mc_ge_0p5"]}/{q.loc[20,"BC_mc_ge_0p5"]}',
                'catastrophic Top10/20':f'{q.loc[10,"catastrophic_mc"]}/{q.loc[20,"catastrophic_mc"]}',
                'official Top10/20':f'{q.loc[10,"official_frontend"]}/{q.loc[20,"official_frontend"]}',
                'Hanabi Top10/20':f'{q.loc[10,"official_hanabi"]}/{q.loc[20,"official_hanabi"]}'})
    c.csv(root/'tables/TARGET_GATES.csv',gates);c.csv(root/'tables/COMPACT_RESULT_SUMMARY.csv',compact)
    both=pd.DataFrame(gates).groupby('method').target_pass.all()
    c.dump(root/'contracts/FINAL_DECISION.json',{'status':c.STATUS,'joint_target_achieved':bool(both.any()),
        'passing_methods':both[both].index.tolist(),'new_encoder_training':False,'raw_time_sky_changed':False,
        'outer_weights_changed':False,'real_PE_used_for_parameter_selection':False})
    report=['# MCWF-CALFUSE-GUARD-02：保留旧波形负证据的敏感性对照','',f'状态：{c.STATUS}','',
        '本轮是查看CALFUSE-01之后建立的新探索，不是其原预注册分析臂。原结果不覆盖。本轮只在原模拟tune集合选择gamma，未用真实PE/官方表选参数，仍不是新的独立确认。',
        '', '冻结全部encoder、原始时间/天空和外层C-fixed权重。MINIMUM不允许任意pair的波形分超过旧分数；NEGATIVE-PRESERVE只要求旧负分不被削弱。由gamma控制更新幅度，0明确表示不修改。两个运行期使用同一规则。这个保守规则不是proper Bayes factor。',
        '', '本轮是否共同达标：'+str(bool(both.any())), '',pd.DataFrame(compact).to_markdown(index=False,floatfmt='.4f'),'',
        '不能按真实结果从上表挑一个运行期的优胜者，再与另一个运行期的不同规则拼接为统一方法。全部逐seed、consensus、PE和官方预算均保存，负结果和gamma=0同样报告。',
        '', '数据定义、BAYESTAR高斯触发条件、O3日历范围和历史适应性限制与父实验一致，详见其报告。未运行新天空PE或Hanabi。',
        '', '复现入口：scripts/guard_calfuse.py --root <独立新目录> --parent <CALFUSE-01目录>。本包依赖历史模型/评分数据，不含原始strain。']
    (root/'reports/GUARD_02_REPORT_CN.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    verification=[]
    for r in pd.read_csv(root/'manifest/PARENT_INPUTS.csv').itertuples():
        actual=c.dev.sha(Path(r.path));assert actual==r.sha256
        verification.append({'path':r.path,'sha256':actual,'unchanged':True})
    for r in pd.read_csv(parent/'manifest/PROTECTED_INPUTS.csv').itertuples():
        actual=c.dev.sha(Path(r.path));assert actual==r.sha256
        verification.append({'path':r.path,'sha256':actual,'unchanged':True})
    c.csv(root/'manifest/HISTORICAL_HASH_RECHECK.csv',verification)
    c.dump(root/'audit/FINAL_VERIFICATION.json',{'all_pass':True,'negative_evidence_never_weakened':True,
        'protected_files_checked':len(verification),'selected_config_sha256':c.dev.sha(root/'configs/SELECTED_CONFIGURATIONS.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();run(a.root,a.parent)
