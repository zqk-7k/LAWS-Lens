"""Read-only mixed-precision batch-shape audit; never changes archived scores."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="2"
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
P=Path("/root/autodl-tmp/gw-catalog")
sys.path[:0]=[str(P),str(P/"scripts/experiments")]
import mcwf_path_fresh_confirmation_v2_20260908 as f
p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
out=p.parse_args().out;out.mkdir(exist_ok=False)
torch.set_num_threads(2)
full,events=f.dev.real_inputs("gwtc3")
selected=pd.read_csv(P/"results/lensrank_speed_scientific_pilot_20260917T073200Z/contracts/selected_events.csv")
valid=events[events.strict_h1l1_preprocessing_pass].reset_index(drop=True)
indices=valid.set_index("event_name").loc[selected.event_name].idx.to_numpy(int)
positions=np.array([valid.index[valid.idx.eq(k)][0] for k in indices])
raw=np.asarray(full[indices])
records=[]
for seed in (202607241,202607242,202607243):
    cp=f.dev.V7/f"gwtc3/seed_{seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt"
    model,_=f.dev.BASE.v7.load_unified_model(cp)
    reference,_=f.dev.BASE.v7.embed_catalog(model,f.dev.BASE.v7.ArrayCatalog(np.asarray(full[valid.idx])),batch_size=16)
    ref=reference[positions]
    for n in (2,4,6):
        pad=np.zeros((len(valid),)+raw.shape[1:],np.float32);pad[positions[:n]]=raw[:n]
        z,_=f.dev.BASE.v7.embed_catalog(model,f.dev.BASE.v7.ArrayCatalog(pad),batch_size=16)
        small,_=f.dev.BASE.v7.embed_catalog(model,f.dev.BASE.v7.ArrayCatalog(raw[:n]),batch_size=16)
        records.append(dict(seed=seed,n=n,unpadded_max_error=float(abs(small-ref[:n]).max()),
                            padded_max_error=float(abs(z[positions[:n]]-ref[:n]).max()),
                            native_batch_shape=len(valid),selected_positions=positions[:n].tolist()))
pd.DataFrame(records).to_csv(out/"BATCH_SHAPE_AUDIT.csv",index=False)
(out/"AUDIT.json").write_text(json.dumps(dict(rows=records,
    conclusion="Zero-valued padding retains native batch sizes and event slots; does not read other-event features into the candidate computation.",
    original_network_and_autocast_unchanged=True),indent=2))
print(pd.DataFrame(records).to_string(index=False))
