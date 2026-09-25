"""Read-only physical support audit for the failed author phase transform."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

import h5py
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
PILOT=P/'results/lensrank_speed_scientific_pilot_20260917T073200Z'
sys.path[:0]=[str(PILOT/'vendor/site'),str(PILOT)]
import lensrank_speed_pilot_20260917 as old
import bilby
from phazap import gw_utils


def main(root,output):
    output.mkdir(exist_ok=False)
    (output/'contracts').mkdir()
    old.ROOT=output;old.track=lambda path:Path(path)
    selected=pd.read_csv(root/'contracts/selected_events.csv')
    rows=[]
    for path in (root/'results/Phazap_n15/phases').glob('*.hdf5'):
        with h5py.File(path,'r') as handle:
            nonfinite={k:int((~np.isfinite(handle[k][()])).sum()) for k in handle.keys() if not np.isfinite(handle[k][()]).all()}
            if not nonfinite:continue
            bad=np.flatnonzero(~np.isfinite(handle['Dphi_f'][()]))
            count=len(handle['Dphi_f'])
        row=selected[selected.event_name.eq(path.stem)].iloc[0].copy()
        row['sky_map_internal_group']=row.phazap_posterior_group
        samples,meta=old.read_pe(row)
        wg=bilby.gw.WaveformGenerator(duration=meta['duration'],sampling_frequency=meta['sampling_frequency'],
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(waveform_approximant=meta['waveform_approximant'],
                reference_frequency=meta['reference_frequency'],minimum_frequency=min(meta['reference_frequency'],20)-2/meta['duration'],
                maximum_frequency=100+2/meta['duration'],mode_array=[[2,2],[2,-2]]))
        frequencies=bilby.core.utils.series.create_frequency_series(meta['sampling_frequency'],meta['duration'])
        endpoint=np.flatnonzero((frequencies>=20)&(frequencies<=100))[-1]
        for index in [0,*bad]:
            pars=samples.iloc[index].to_dict()
            hp,hx=gw_utils.hp_hx(pars,wg)
            support=(abs(hp)+abs(hx))>0
            rows.append(dict(event=path.stem,posterior_index=int(index),total_posterior_samples=count,
                nonfinite_Dphi_samples=len(bad),sample_failed=bool(index in bad),
                detector_mass_total=pars['mass_1']+pars['mass_2'],
                f_endpoint_Hz=float(frequencies[endpoint]),abs_hp_endpoint=float(abs(hp[endpoint])),
                abs_hx_endpoint=float(abs(hx[endpoint])),highest_nonzero_frequency_Hz=float(frequencies[support].max()),
                zero_both_polarizations_at_endpoint=bool(hp[endpoint]==0 and hx[endpoint]==0)))
    pd.DataFrame(rows).to_csv(output/'ZERO_SUPPORT_AUDIT.csv',index=False)
    report=dict(events=sorted({r['event'] for r in rows}),audit_rows=len(rows),
        failed_samples_have_zero_endpoint_support=all(r['zero_both_polarizations_at_endpoint'] for r in rows if r['sample_failed']),
        no_posterior_samples_removed=True,no_phase_replaced=True,no_author_code_changes=True,
        implication='If both polarizations vanish, a detector phase at that frequency is undefined. Do not turn it into zero, remove samples, or lower fhigh silently.',
        original_scope_and_failure_retained=True)
    (output/'AUDIT.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
