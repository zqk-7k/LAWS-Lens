#!/usr/bin/env python3
"""Join published O4a results, without changing scores or historical files."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev


def key(a,b): return "--".join(sorted((str(a),str(b))))


def run(root):
    src=root/"source_data/official_O4"
    mapping=pd.read_csv(src/"GWTC41_OFFICIAL_NAME_MAPPING.csv")
    if mapping.gracedb_id.duplicated().any(): raise RuntimeError("Ambiguous official aliases")
    aliases=dict(zip(mapping.gracedb_id,mapping.event_name))
    release=src/"GWTC4_Lensing_DataRelease"
    po=pd.read_csv(release/"Figure2/PO_final_results.csv")
    ph=pd.read_csv(release/"Figure2/data_products_phazap_gwtc4.csv")
    fgol=json.loads((release/"Figure12/fast_golum_o4a_log10CLU.json").read_text())
    fg={key(*pair.split("_")):float(v) for pair,v in fgol.items()}
    po["sid_key"]=[key(a,b) for a,b in zip(po.event1,po.event2)]
    ph["sid_key"]=[key(a,b) for a,b in zip(ph.event_1,ph.event_2)]
    po=po.set_index("sid_key");ph=ph.set_index("sid_key")
    rows=[];unmapped=[]
    for sid in sorted(set(po.index)|set(ph.index)):
        a,b=sid.split("--")
        if a not in aliases or b not in aliases:
            unmapped.append({"sid_pair":sid,"reason":"not in frozen official GWTC4.1 name mapping; includes rejected instrumental S230630bu"})
            continue
        row={"pair_key":key(aliases[a],aliases[b]),"gracedb_i":a,"gracedb_j":b,
            "official_po_fpp":float(po.loc[sid,"FAPpair"]) if sid in po.index else np.nan,
            "official_phazap_fpp":float(ph.loc[sid,"FPP"]) if sid in ph.index else np.nan,
            "official_po_log10BLU":float(po.loc[sid,"log10BLU"]) if sid in po.index else np.nan,
            "official_fast_golum_table_overlap":sid in fg,
            "official_fast_golum_log10C":fg.get(sid,np.nan)}
        row["official_po_or_phazap_fpp_below_0p01"]=(row["official_po_fpp"]<.01 or row["official_phazap_fpp"]<.01)
        row["official_frontend_category"]= ("PO_and_Phazap" if row["official_po_fpp"]<.01 and row["official_phazap_fpp"]<.01 else
            "PO_only" if row["official_po_fpp"]<.01 else "Phazap_only" if row["official_phazap_fpp"]<.01 else "not_below_frontend_0p01")
        rows.append(row)
    official=pd.DataFrame(rows)
    for model in ("RzMD","Rzmin","Rzmax"):
        f=pd.read_csv(release/f"Figure3/o4a-hanabi-analysis-Bayes-factors-{model}.csv")
        f["pair_key"]=[key(a,b) for a,b in zip(f.event1_full_name,f.event2_full_name)]
        f=f[["pair_key","ln_Bayes_factor","ln_coherence_ratio"]].rename(columns={
            "ln_Bayes_factor":f"official_hanabi_lnB_{model}","ln_coherence_ratio":f"official_hanabi_lnC_{model}"})
        official=official.merge(f,on="pair_key",how="outer",validate="one_to_one")
    official["official_any_pair_resolved_hanabi_overlap"]=official.official_hanabi_lnB_RzMD.notna()
    official["official_hanabi_conclusion"]=np.where(official.official_any_pair_resolved_hanabi_overlap,
        "Published Bayesian follow-up; no strong-lensing evidence in adopted population analysis","not_in_published_pair_resolved_hanabi_table")
    official["official_screening_stage"]=np.where(official.official_any_pair_resolved_hanabi_overlap,"published_Hanabi",
        np.where(official.official_fast_golum_table_overlap.fillna(False),"published_Fast_GOLUM","PO_Phazap_frontend"))
    # Standard aliases are used only by the generic budget reporter.
    official["official_po_or_ml_fpp_below_0p01"]=official.official_po_or_phazap_fpp_below_0p01
    dev.csv_write(root/"audit/O4_OFFICIAL_PUBLISHED_RESULTS.csv",official)
    dev.csv_write(root/"audit/O4_OFFICIAL_UNMAPPED_ALIASES.csv",pd.DataFrame(unmapped))
    old=dev.pe_audit(root,"gwtc4")
    keep=[c for c in old if not c.startswith("official_") and c!="public_hanabi_table_overlap"]
    final=old[keep].merge(official,on="pair_key",how="left",validate="one_to_one")
    final.to_parquet(root/"audit/gwtc4_all_pair_pe_official_verified.parquet",index=False)
    summary={"official_mapped_pairs":len(official),"scope_pairs":len(final),
        "scope_with_PO":int(final.official_po_fpp.notna().sum()),"scope_with_Phazap":int(final.official_phazap_fpp.notna().sum()),
        "scope_with_FastGOLUM":int(final.official_fast_golum_table_overlap.fillna(False).sum()),
        "scope_with_Hanabi":int(final.official_any_pair_resolved_hanabi_overlap.fillna(False).sum()),
        "source_record":"https://zenodo.org/records/18163632", "paper":"https://arxiv.org/abs/2512.16347",
        "no_scores_changed":True,"historical_placeholder_tables_preserved":True,
        "frontend_definition":"published Figure2 threshold .01, PO OR Phazap; NOT truth or detection",
        "generic_column_alias":"official_po_or_ml_fpp_below_0p01 means PO_OR_Phazap for O4a only"}
    dev.json_write(root/"audit/O4_OFFICIAL_JOIN_AUDIT.json",summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();run(a.root)
