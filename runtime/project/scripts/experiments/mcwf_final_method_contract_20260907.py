#!/usr/bin/env python3
"""Clarify inherited schema prose without modifying selected numeric configurations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body


def run(root):
    path=root/'contracts/FINAL_METHOD_CLARIFICATION.json'
    if path.exists():
        raise RuntimeError('Final clarification already exists; do not overwrite')
    info=json.loads((root/'contracts/DISTILLED_MODEL_PROVENANCE.json').read_text())
    trained=Path(info['training_root'])
    configs=[]
    for dep in ('gwtc3','gwtc4'):
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            p=root/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json'
            cfg=json.loads(p.read_text())
            cp=trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            if cfg['recipe']!='finite_reference_companion_tail' or cfg['alpha']!=.05 or not ck.get('rankncontrast'):
                raise RuntimeError('Unexpected candidate identity')
            configs.append({'deployment':dep,'model_seed':ms,'eval_seed':es,'gamma':cfg['gamma'],
                'reference_systems':len(cfg['reference']),'selected_epoch':ck['epoch'],
                'temperature':ck['temperature'],'config_sha256':dev.sha(p),'checkpoint_sha256':dev.sha(cp)})
    contract={'utc':datetime.now(timezone.utc).isoformat(),'code':'MCWF-UNIFIED-RNC-FRT',
        'documentation_only':True,'selected_numeric_configs_not_modified':True,
        'reason':'Earlier compatibility-schema prose mentions convex ensembles and mass-weight grids. Those fields are not executed by this finite-reference score; this clarification records the actual implementation without rewriting historical contracts.',
        'actual_score':'Zwf_new=Zwf_Cfixed + gamma*min(log(p_tail/0.05),0)',
        'p_tail':'(1+sum_reference[negative_log_BC_reference >= negative_log_BC_pair])/(n_reference+1)',
        'new_inputs':'same peak2s4096 H1L1 waveform, with sample-resolved FFT quadrature bank whitened by the event off-source PSD',
        'network':'shared raw InceptionAttention branch plus1728-element quadrature-feature MLP;128D representation and64bin simulated logMc head',
        'training':'six independent50epoch fine-tunings; CE + source-SupCon + RNC(logMc); checkpoint chosen by validation CE+.2SupCon+.2RNC; no real PE training targets',
        'new_prediction':'temperature-calibrated64bin predictive mass distribution; not a PE posterior',
        'embedding_role':'used during shared representation training and diagnostic audits; the final correction uses its mass head distribution, not an additional direct new-cosine term',
        'old_waveform_role':'unchanged C-fixed learned waveform evidence supplies morphology evidence; same old-plus-new correction in BOTH runs and every seed',
        'no_old_only_O4_fallback':True,
        'calibration':'true-pair reference disagreements and gamma selected on simulation validation; gamma grid2**k,k=-6..1; same guardrails/objective in both runs',
        'ranking_not_probability':'finite-reference rank is conformal-style but no real-domain FPP/coverage guarantee is claimed; correction and total score are not a lens Bayes factor or posterior probability',
        'raw_input_and_training_caveat':'legacy O3 development noise includes O1/O2; O4a is run matched. Final new-noise confirmation uses O3-only/O4a-only noise.',
        'real_PE_role':'repeatedly inspected development feedback, not independent confirmation; official-stage overlap descriptive, never labels for fitting',
        'frozen_elsewhere':'time lookup, BAYESTAR sky, C-fixed outer weights, event scopes, historical results and paper',
        'configs':configs,'historical_contract_sha256':dev.sha(root/'contracts/UNIFIED_METHOD.json'),
        'source_training_contract_sha256':dev.sha(trained/'contracts/RNC_TRAINING.json')}
    dev.json_write(path,contract)
    print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
