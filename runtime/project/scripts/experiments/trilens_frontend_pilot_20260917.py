"""Bounded products-ready frontend replay, not a lens-detection experiment."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

P = Path("/root/autodl-tmp/gw-catalog")
ARCHIVE = P / "results/mcwf_unified_path875_devconf_20260908T181500Z"
PILOT = P / "results/lensrank_speed_scientific_pilot_20260917T073200Z"
LOW = P / "results/mcwf_multirate_lowband_exploratory_20260908T145551Z"
UAB = P / "results/gwlr_unified_snr_ab_20260917T113500Z_r4"
ROOT = None
TIMES, CHECKS, INPUTS = [], [], {}
CONTEXT = {}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def track(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        INPUTS[str(path)] = dict(sha256_before=sha(path), bytes=path.stat().st_size)
    return path


def record_check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))
    js(ROOT / "results/checks.json", CHECKS)
    if not passed:
        raise RuntimeError(name + ": " + str(details))


def sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def measured(stage, fn, **extra):
    sync()
    start, cpu = time.perf_counter(), time.process_time()
    result = fn()
    sync()
    TIMES.append(dict(**CONTEXT, stage=stage, wall_seconds=time.perf_counter()-start,
                      process_cpu_seconds=time.process_time()-cpu,
                      cumulative_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      **extra))
    pd.DataFrame(TIMES).to_csv(ROOT / "tables/timings.csv", index=False)
    return result


def load_science():
    sys.path[:0] = [str(P / "scripts/experiments"), str(P), str(PILOT)]
    sys.path.insert(0, str(PILOT / "vendor/site"))
    import mcwf_path_fresh_confirmation_v2_20260908 as frozen
    import main_o3official_cfixed_v1 as catalog
    import lensrank_speed_pilot_20260917 as oldpilot
    import phazap
    import phazap.pe_input
    import phazap.postprocess_phase
    return frozen, catalog, oldpilot


def prepare():
    ROOT.mkdir(exist_ok=False)
    for directory in ("contracts", "scripts", "results", "tables", "reports", "logs", "manifest", "phases", "figures"):
        (ROOT / directory).mkdir()
    shutil.copy2(__file__, ROOT / "scripts" / Path(__file__).name)
    selected = pd.read_csv(track(PILOT / "contracts/selected_events.csv"))
    selected.to_csv(ROOT / "contracts/selected_events.csv", index=False)
    js(ROOT / "contracts/CONTRACT.json", dict(
        code="TRILENS-FRONTEND-PILOT-02", algorithm="NEW-SCORE-ONLY / PATH1; alpha=1",
        selection="Same six name-hash-selected O3 events as PILOT-01; all within-subset pairs",
        subset_sizes=[2, 4, 6], technical_repeats=2, selected_before_new_timings=True,
        absolute_timeout_seconds=1800, cpu_threads=2, sampler_jobs_authorized=0,
        inputs_ready=["GWOSC strain on disk", "public PE posterior and native sky maps", "frozen models/calibrations/template spectra"],
        trilens_includes=["strain/reference read", "PSD estimation", "2s/16s preprocessing", "all feature banks",
                          "three seeds: short encoder, RNC, ordered mass, multirate mass, conditional intrinsic head",
                          "all frozen calibrations", "sky read/NESTED-to-RING/512/overlap", "time lookup",
                          "pair scores", "consensus", "output writes", "checkpoint IO"],
        phazap_includes=["full posterior read", "author phase reconstruction", "all pair comparisons", "output writes"],
        excluded_both=["PE production", "network download", "offline training/calibration", "software installation"],
        startup="Python imports separately timed; no OS page-cache drop, no claim of disk-cold measurement",
        replay_tolerance={"score_absolute": 2e-4, "score_relative": 1e-5, "rank": "exact within subset"},
        phazap_configuration={"version":"0.3.3", "flow":20, "fbest":40, "fhigh":100,
                              "samples":"all", "no_author_algorithm_changes":True},
        official_po="NOT_RUN until authentic joint implementation, prior, time population and FPP calibration verified",
        official_combined_frontend="NOT_RUN; standalone phase scores not official calibrated shortlist",
        matched_recall_cost="NOT_RUN; no independent injection joint PE currently authorized by quick scope",
        phazap_distance_not_catalog_fpp=True, no_true_lensing_labels_on_real_data=True,
        resource_contention="UAB training continues; native TriLens GPU and Phazap CPU; not equal-resource speedup proof",
        numerical_execution="Zero padding neural inputs to archived 62 event slots to preserve BF16 batch kernels. All dummy compute is charged. Features/sky/pairs only for 2/4/6 real events. This is not optimized small-catalog scaling.",
        previous_attempts="r1 import-path failure; r2 unpadded mixed-precision replay failure retained, not reclassified",
        historical_files_read_only=True, no_model_or_rank_adoption=True,
        status="HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"))
    js(ROOT / "contracts/FREEZE.json", dict(contract_sha256=sha(ROOT / "contracts/CONTRACT.json"),
        code_sha256=sha(Path(__file__)), event_sha256=sha(ROOT / "contracts/selected_events.csv")))
    return selected


def hardware():
    return dict(utc=pd.Timestamp.now(tz="UTC").isoformat(), load=os.getloadavg(),
        cpu_affinity=list(os.sched_getaffinity(0)),
        cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        ram_limit=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        disk_free_bytes=shutil.disk_usage(ROOT).free,
        gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip(),
        concurrent_training=json.loads((UAB / "RUN_STATUS.json").read_text()))


def inventory(frozen, catalog, selected):
    recipes = json.loads(track(ARCHIVE / "independent_confirmation/contracts/FRESH_FREEZE.json").read_text())
    CONTEXT.update(method="SETUP", n_events=6, repetition=-1)
    _, event_rows = frozen.dev.real_inputs("gwtc3")
    valid = event_rows[event_rows.strict_h1l1_preprocessing_pass].reset_index(drop=True)
    event_rows = event_rows.set_index("event_name").loc[selected.event_name].reset_index()
    event_rows["native_slot"] = [int(valid.index[valid.event_name.eq(name)][0]) for name in event_rows.event_name]
    event_rows["native_count"] = len(valid)
    event_rows.to_csv(ROOT / "contracts/strain_manifest.csv", index=False)
    for row in event_rows.itertuples(index=False):
        for detector in json.loads(row.detector_audit):
            track(detector["path"])
    for path in selected.sky_map_path:
        track(path)
    for recipe in recipes["recipes"]:
        if recipe["deployment"] != "gwtc3":
            continue
        seed, slot = recipe["seed"], recipe["slot"]
        paths = [catalog.checkpoint(seed),
            frozen.dev.V7 / f"gwtc3/seed_{seed}/results/waveform_channel_calibration_v7.json",
            frozen.e.TRAINED / f"models/RAW-PHASE-SOURCE/gwtc3/seed_{slot}/validation_selected_model.pt",
            frozen.e.BASE / f"calibration/gwtc3/seed_{seed}/SELECTED_CONFIG.json",
            frozen.t.PREVIOUS / f"ordered_mass_predictor/models/gwtc3/seed_{slot}/validation_selected_model.pt",
            frozen.INTRINSIC / f"models/CONDITIONAL-ETA-CHI/gwtc3/seed_{slot}/selected.pt",
            LOW / f"models/MULTIRATE/gwtc3/seed_{slot}/selected.pt",
            ARCHIVE / f"development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/gwtc3/seed_{seed}/real_fusion_pairs.parquet"]
        for path in paths:
            track(path)
    for path in [frozen.dev.V7 / "gwtc3/shared/time_delay_likelihood_ratio.json",
                 frozen.e.TRAINED / "cache/adaptive_psd/unwhitened_aligned_template_spectra.npy",
                 frozen.t.PREVIOUS / "fine_mass_context/cache/fine_mass_spectra.npy",
                 LOW / "cache/lowband_aligned_spectra.npy"]:
        track(path)
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(path).startswith(str(P)) and str(path).endswith(".py"):
            track(path)
    js(ROOT / "manifest/INPUTS_BEFORE.json", INPUTS)
    return event_rows, [r for r in recipes["recipes"] if r["deployment"] == "gwtc3"]


def read_raw(event_rows):
    raw, references = [], []
    for row in event_rows.itertuples(index=False):
        waves, refs = [], []
        audit = {d["detector"]:d for d in json.loads(row.detector_audit)}
        for detector in ("H1", "L1"):
            entry = audit[detector]
            with h5py.File(entry["path"], "r") as handle:
                data = handle["strain/Strain"]
                spacing, start = float(data.attrs["Xspacing"]), float(data.attrs["Xstart"])
                if abs(spacing - 1/4096) > 1e-12:
                    raise RuntimeError("Unexpected strain rate")
                a = int(round((float(row.gps_time)-24.75-start)*4096))
                b = int(round((float(entry["psd_reference_start_gps"])-start)*4096))
                waves.append(np.asarray(data[a:a+26*4096], np.float32))
                refs.append(np.asarray(data[b:b+256*4096], np.float32))
        raw.append(np.stack(waves)); references.append(np.stack(refs))
    return np.stack(raw), np.stack(references)


def all_features(frozen, full, lowviews, frequency, psds):
    feature = frozen.archived.feature
    adaptive = feature.adaptive
    banks = measured("spectral_bank_IO", lambda: (
        np.load(frozen.e.TRAINED / "cache/adaptive_psd/unwhitened_aligned_template_spectra.npy"),
        np.load(frozen.t.PREVIOUS / "fine_mass_context/cache/fine_mass_spectra.npy"),
        np.load(LOW / "cache/lowband_aligned_spectra.npy")))
    raw = frozen.dev.TRAIN.make_window_view(full, 2)
    coarse, refined, low = [], [], []
    for index in range(len(full)):
        def coarse_fn():
            bank = adaptive.whitened_bank(banks[0], frequency, psds[index])
            return feature.fine.features(raw[index:index+1], bank)
        coarse.append(measured("coarse_phase_feature", coarse_fn, event_index=index))
        def refined_fn():
            bank = adaptive.whitened_bank(banks[1], frequency, psds[index])
            return frozen.archived.fine.features(raw[index:index+1], bank)
        refined.append(measured("fine_mass_feature", refined_fn, event_index=index))
        def low_fn():
            bank = frozen.low.bank_for(banks[2], frequency, psds[index])
            return frozen.low.features(lowviews[index:index+1], bank)
        low.append(measured("low16s_feature", low_fn, event_index=index))
    return np.concatenate(coarse), np.concatenate(refined), np.concatenate(low), raw


def rnc(frozen, raw, x, slot):
    feature = frozen.archived.feature
    ck = torch.load(frozen.e.TRAINED / f"models/RAW-PHASE-SOURCE/gwtc3/seed_{slot}/validation_selected_model.pt",
                    map_location="cpu", weights_only=False)
    model = feature.body.Encoder("RAW-PHASE-SOURCE").cuda().eval()
    model.load_state_dict(ck["model"])
    logits, embedding = feature.body.infer(model, (x-ck["mu"])/ck["sd"], raw)
    if "independent_mass_component" in ck or "physical_mass_anchor" in ck:
        raise RuntimeError("Extra archived RNC branch requires explicit timing adapter")
    lp = logits.astype(float)/ck["temperature"]
    lp -= np.logaddexp.reduce(lp, axis=-1, keepdims=True)
    return np.exp(lp), embedding


def rank_consensus(frames):
    rows = []
    for frame in frames:
        frame = frame.sort_values(["final_score", "pair_key"], ascending=[False, True]).copy()
        frame["rank"] = np.arange(1, len(frame)+1)
        rows.append(frame)
    result = pd.concat(rows).groupby("pair_key").agg(mean_rank=("rank","mean"),
        max_rank=("rank","max"), mean_score=("final_score","mean")).reset_index()
    result = result.sort_values(["mean_rank","max_rank","mean_score","pair_key"], ascending=[True,True,False,True])
    result["rank"] = np.arange(1,len(result)+1)
    return result


def trilens(frozen, catalog, selected, event_rows, recipes, n, rep):
    dest = ROOT / f"results/TriLens_n{n}_r{rep}"
    dest.mkdir()
    rows = event_rows.iloc[:n]
    raw, refs = measured("strain_and_reference_IO", lambda: read_raw(rows))
    def psd_fn():
        values = [frozen.dev.BASE.v7.v3.estimate_psd(r) for r in refs]
        return values[0][0], np.stack([v[1] for v in values])
    frequency, psds = measured("PSD_estimation", psd_fn)
    full = measured("short_preprocessing", lambda: np.stack([
        frozen.dev.BASE.v7.v3.preprocess_24s(x, frequency, psds[k]).astype(np.float32) for k,x in enumerate(raw)]))
    lowviews = measured("low16s_preprocessing", lambda: np.stack([
        frozen.low.low_view(x, frequency, psds[k], frozen.dev.BASE.v7.v3) for k,x in enumerate(raw)]))
    coarse, refined, low, short = all_features(frozen, full, lowviews, frequency, psds)
    x = frozen.ordered.arrange(coarse, refined)
    slots = rows.native_slot.to_numpy(int)
    native_count = int(rows.native_count.iloc[0])
    def pad(array):
        padded = np.zeros((native_count,)+array.shape[1:],array.dtype)
        padded[slots] = array
        return padded
    padded_full, padded_short, padded_coarse, padded_x = pad(full), pad(short), pad(coarse), pad(x)
    i,j = np.triu_indices(n,1)
    names = rows.event_name.to_numpy()
    frame = pd.DataFrame(dict(idx_i=i, idx_j=j, event_i=names[i], event_j=names[j],
        pair_key=["--".join(sorted([a,b])) for a,b in zip(names[i],names[j])]))
    delta = abs(rows.gps_time.to_numpy()[i]-rows.gps_time.to_numpy()[j])/86400
    def time_fn():
        cal = json.loads((frozen.dev.V7 / "gwtc3/shared/time_delay_likelihood_ratio.json").read_text())
        return frozen.dev.BASE.v7.apply_time_likelihood_ratio(delta,cal)
    frame["time_score"] = measured("time_lookup_IO_and_scoring", time_fn)
    def sky_fn():
        maps, audits = [], []
        for _,row in selected.iloc[:n].iterrows():
            probability, audit = catalog.cbase.corrected_read_probability_map(row,512)
            maps.append(probability); audits.append(audit)
        tensor = torch.as_tensor(np.stack(maps), device="cuda", dtype=torch.float64)
        gram = (tensor @ tensor.T).cpu().numpy()
        bc = (torch.sqrt(tensor) @ torch.sqrt(tensor).T).cpu().numpy()
        js(dest / "sky_ordering.json",audits)
        return np.log(np.maximum(len(maps[0])*gram[i,j],catalog.SKY_BF_FLOOR)), bc[i,j]
    frame["sky_raw_log_bf"], frame["sky_bc"] = measured("public_sky_IO_ordering_512_and_overlap",sky_fn)
    outputs = []
    for recipe in recipes:
        seed, slot = recipe["seed"], recipe["slot"]
        def short_fn():
            original = frame.copy()
            original["idx_i"],original["idx_j"] = slots[i],slots[j]
            scored = frozen.old.old_waveform("gwtc3",seed,padded_full,original)
            scored["idx_i"],scored["idx_j"] = i,j
            return scored
        f = measured("short_encoder_and_original_calibration", short_fn, seed=seed)
        f["previous_waveform_score"] = f.waveform_score
        p, emb = measured("RNC_checkpoint_and_inference", lambda: rnc(frozen,padded_short,padded_coarse,slot),seed=seed)
        p,emb = p[slots],emb[slots]
        def frt_fn():
            v = frozen.archived.ev.features(f,p,emb)
            for name,key in (("new_encoder_cosine","cosine"),("new_mass_similarity","mass"),
                             ("new_mass_predictive_BC","predictive_BC"),("new_mass_pred_logmc_i","mean_i"),("new_mass_pred_logmc_j","mean_j")):
                f[name] = v[key]
            spec = json.loads((frozen.e.BASE / f"calibration/gwtc3/seed_{seed}/SELECTED_CONFIG.json").read_text())
            return frozen.archived.finite.score(f,spec)[0]
        f["waveform_score"] = measured("RNC_pair_and_FRT_calibration",frt_fn,seed=seed)
        def omc_fn():
            ck = torch.load(frozen.t.PREVIOUS / f"ordered_mass_predictor/models/gwtc3/seed_{slot}/validation_selected_model.pt",map_location="cpu",weights_only=False)
            model = frozen.ordered.Predictor().cuda().eval();model.load_state_dict(ck["model"])
            pp,boundary = frozen.ordered.probability(frozen.ordered.infer(model,(padded_x-ck["mu"])/ck["sd"]),ck["temperature"])
            pp,boundary = pp[slots],boundary[slots]
            values = frozen.masscal.pair_features({"p":pp,"outside":boundary},i,j,ck["prior"])
            return frozen.masscal.score(f,values,recipe["OMC_config"],"PRIOR")[0]
        baseline = measured("ordered_mass_checkpoint_inference_and_calibration",omc_fn,seed=seed)
        a = measured("multirate_and_joint_checkpoints_inference",lambda: frozen.joint_prediction(pad(np.concatenate([x,low],1)),"gwtc3",slot),seed=seed)
        a = {key:value[slots] for key,value in a.items()}
        def joint_fn():
            pen,inc,audit = frozen.joint_increment(a,f,recipe["joint_config"])
            spec = recipe["joint_config"]
            return baseline if spec.get("unchanged_OMC",False) else baseline+spec["gamma"]*pen+spec["beta"]*inc
        f["waveform_score"] = measured("joint_pair_and_frozen_calibration",joint_fn,seed=seed)
        f["seed"] = seed
        f["final_score"] = f[["waveform_score","time_score","sky_raw_log_bf"]].to_numpy() @ np.asarray(recipe["upstream_weights"])
        f.to_csv(dest / f"seed_{seed}.csv",index=False)
        outputs.append(f)
    consensus = measured("fusion_consensus_and_output", lambda: write_consensus(outputs,dest))
    return dict(frames=outputs,consensus=consensus,full=full,sky=frame)


def write_consensus(frames, dest):
    result = rank_consensus(frames)
    result.to_csv(dest / "consensus.csv",index=False)
    return result


def verify_trilens(frozen,result,rows,recipes,n,rep):
    expected_full = frozen.dev.real_inputs("gwtc3")[0][rows.iloc[:n].idx.to_numpy(int)]
    record_check(f"raw_reconstruction_n{n}_r{rep}",np.array_equal(expected_full,result["full"]),
                 maximum_difference=float(abs(expected_full-result["full"]).max()))
    oldframes = []
    for frame,recipe in zip(result["frames"],recipes):
        seed = recipe["seed"]
        old = pd.read_parquet(ARCHIVE / f"development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/gwtc3/seed_{seed}/real_fusion_pairs.parquet")
        old = old.set_index("pair_key").loc[frame.pair_key].reset_index()
        differences = {}
        for new_col, old_col in (("waveform_score","upstream_joint_waveform"),("time_score","time_score"),("sky_raw_log_bf","sky_raw_log_bf")):
            a,b = frame[new_col].to_numpy(),old[old_col].to_numpy()
            differences[new_col] = float(abs(a-b).max())
            record_check(f"score_replay_{new_col}_n{n}_r{rep}_s{seed}",np.allclose(a,b,atol=2e-4,rtol=1e-5),max_abs_difference=differences[new_col])
        old["final_score"] = old[["upstream_joint_waveform","time_score","sky_raw_log_bf"]].to_numpy() @ np.asarray(recipe["upstream_weights"])
        oldframes.append(old)
    expected = rank_consensus(oldframes)
    record_check(f"consensus_replay_n{n}_r{rep}",expected.pair_key.tolist()==result["consensus"].pair_key.tolist())


def phazap_run(oldpilot,selected,n,rep):
    sys.path.insert(0,str(PILOT / "vendor/site"))
    from phazap.pe_input import ParameterEstimationInput
    from phazap.postprocess_phase import postprocess_phase
    from phazap import phazap
    dest = ROOT / f"results/Phazap_n{n}_r{rep}"
    dest.mkdir()
    oldpilot.ROOT = ROOT
    # Full-file integrity was checked in inventory, outside both timers.
    oldpilot.track = lambda path: Path(path)
    phases,counts = {},{}
    for row in selected.iloc[:n].itertuples(index=False):
        samples,meta = measured("public_PE_read",lambda: oldpilot.read_pe(row),event=row.event_name)
        counts[row.event_name] = len(samples)
        pe = ParameterEstimationInput(samples,**meta)
        phases[row.event_name] = measured("full_posterior_to_phase",lambda:postprocess_phase(pe,
            flow=20,fhigh=100,fbest=40,superevent_name=row.event_name,label=f"n{n}_r{rep}",
            output_dir=str(ROOT / "phases"),output_filename=f"{row.event_name}_n{n}_r{rep}.hdf5"),event=row.event_name)
    pairs=[]
    for a,b in itertools.combinations(selected.iloc[:n].event_name,2):
        v = measured("phase_pair_comparison",lambda:phazap(phases[a],phases[b],plot=False),event_i=a,event_j=b)
        pairs.append(dict(event_i=a,event_j=b,pair_key="--".join(sorted([a,b])),DJ=float(v[0]),
            volume=float(v[1]),phase_shift=float(v[2]),upstream_p_value=float(v[4]),
            posterior_samples_i=counts[a],posterior_samples_j=counts[b]))
    f = pd.DataFrame(pairs)
    # DJ ordering is a diagnostic, not the LVK background-calibrated selection.
    f.sort_values(["DJ","pair_key"]).to_csv(dest / "all_pair_statistics.csv",index=False)
    return f


def verify_phazap(frame,n,rep):
    old = pd.read_csv(PILOT / "fullposterior_diagnostic/tables/phazap_full_pairs.csv")
    old = old[old.samples.eq("full")].copy()
    old["pair_key"] = ["--".join(sorted([a,b])) for a,b in zip(old.event_i,old.event_j)]
    old = old.set_index("pair_key").loc[frame.pair_key]
    record_check(f"Phazap_replay_n{n}_r{rep}",np.allclose(frame.DJ.to_numpy(),old.DJ.to_numpy(),atol=1e-8,rtol=1e-8),
                 max_abs_delta_DJ=float(abs(frame.DJ.to_numpy()-old.DJ.to_numpy()).max()))


def finish(state):
    rows=[]
    for path,before in INPUTS.items():
        after=sha(path)
        rows.append(dict(path=path,**before,sha256_after=after,unchanged=after==before["sha256_before"]))
    pd.DataFrame(rows).to_csv(ROOT / "manifest/PROTECTED_INPUTS.csv",index=False)
    js(ROOT / "contracts/FINAL_STATUS.json",dict(state=state,
        historical_inputs_unchanged=all(r["unchanged"] for r in rows),
        checks_passed=sum(c["passed"] for c in CHECKS),checks_total=len(CHECKS),
        no_recall_or_speedup_claim=True,scientific_status="HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"))
    js(ROOT / "contracts/HARDWARE_AFTER.json",hardware())


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    ROOT=parser.parse_args().root
    selected=prepare()
    try:
        js(ROOT / "contracts/HARDWARE_BEFORE.json",hardware())
        CONTEXT.update(method="SETUP",n_events=6,repetition=-1)
        frozen,catalog,oldpilot=measured("Python_science_imports",load_science)
        threadpool_limits(2);torch.set_num_threads(2)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        rows,recipes=inventory(frozen,catalog,selected)
        for rep in range(2):
            for n in (2,4,6):
                order=("TriLens","Phazap") if rep==0 else ("Phazap","TriLens")
                for method in order:
                    if shutil.disk_usage(ROOT).free<25*2**30:
                        raise RuntimeError("HOLD_DISK_LIMIT")
                    CONTEXT.update(method=method,n_events=n,repetition=rep)
                    print(json.dumps(CONTEXT),flush=True)
                    js(ROOT / "contracts/RUN_STATUS.json",CONTEXT)
                    if method=="TriLens":
                        result=measured("TOTAL_PRODUCTS_READY_TO_SCORES_AND_RANKS",lambda:trilens(frozen,catalog,selected,rows,recipes,n,rep))
                        verify_trilens(frozen,result,rows,recipes,n,rep)
                    else:
                        result=measured("TOTAL_PRODUCTS_READY_TO_SCORES_AND_RANKS",lambda:phazap_run(oldpilot,selected,n,rep))
                        verify_phazap(result,n,rep)
                    del result;gc.collect()
        js(ROOT / "results/GPU_MEMORY.json",dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory_is_process_specific=True))
        finish("QUICK_TIMING_COMPLETE_OFFICIAL_PO_AND_MATCHED_RECALL_PENDING")
    except Exception:
        (ROOT / "logs/FAIL.txt").write_text(traceback.format_exc())
        finish("HOLD_REPLAY_OR_EXECUTION_FAILURE")
        raise


if __name__=="__main__":
    main()
