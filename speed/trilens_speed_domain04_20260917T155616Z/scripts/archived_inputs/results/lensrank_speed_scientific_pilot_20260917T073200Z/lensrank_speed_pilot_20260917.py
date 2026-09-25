"""Bounded component benchmark. No PE sampling or scientific model selection."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ast
import hashlib
import itertools
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil

P = Path("/root/autodl-tmp/gw-catalog")
A = P / "results/mcwf_unified_path875_devconf_20260908T181500Z"
O3 = P / "results/main_o3official_cfixed_v1_20260904_20260904T072435Z"
ROOT = None
TIMES = []
CHECKS = []
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def track(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        INPUTS[str(path)] = {"sha256_before": sha(path), "bytes": path.stat().st_size}
    return path


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def measure(stage, unit, fn, rep=0, **metadata):
    cpu = time.process_time()
    start = time.perf_counter()
    value = fn()
    row = dict(stage=stage, unit=unit, repetition=rep,
               wall_seconds=time.perf_counter() - start,
               process_cpu_seconds=time.process_time() - cpu,
               cumulative_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               **metadata)
    TIMES.append(row)
    pd.DataFrame(TIMES).to_csv(ROOT / "tables/timings.csv", index=False)
    return value


def check(name, passed, **details):
    CHECKS.append(dict(check=name, passed=bool(passed), **details))
    js(ROOT / "results/checks.json", CHECKS)


def frozen_table(dep, seed):
    return A / f"development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/{dep}/seed_{seed}/real_fusion_pairs.parquet"


def rank_catalog(tables, recipes, dep):
    per_seed = []
    for seed, d in tables.items():
        recipe = next(r for r in recipes if r["deployment"] == dep and r["seed"] == seed)
        w = np.asarray(recipe["upstream_weights"])
        x = d[["upstream_joint_waveform", "time_score", "sky_raw_log_bf"]].to_numpy(float)
        out = d[["pair_key"]].copy()
        out["score"] = x @ w
        out = out.sort_values(["score", "pair_key"], ascending=[False, True])
        out["rank"] = np.arange(1, len(out) + 1)
        per_seed.append(out)
    c = pd.concat(per_seed).groupby("pair_key", sort=True).agg(
        mean_rank=("rank", "mean"), max_rank=("rank", "max"), mean_score=("score", "mean"))
    c = c.reset_index().sort_values(["mean_rank", "max_rank", "mean_score", "pair_key"],
                                  ascending=[True, True, False, True])
    c["rank"] = np.arange(1, len(c) + 1)
    return c


def prepare():
    for folder in ("contracts", "tables", "results", "reports", "logs", "figures", "phases", "manifest", "scripts"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    contract_path = ROOT / "contracts/PILOT_CONTRACT.json"
    if contract_path.exists():
        raise RuntimeError("Refusing to overwrite an existing pilot")
    # Select before reading any timing, phase-consistency or PE-audit result.
    manifest = pd.read_csv(track(O3 / "data/official70_event_manifest.csv"))
    d = pd.read_parquet(track(frozen_table("gwtc3", 202607241)), columns=["event_i", "event_j"])
    names = set(d.event_i) | set(d.event_j)
    eligible = manifest[manifest.event_name.isin(names)].copy()
    eligible["selection_hash"] = eligible.event_name.map(
        lambda s: hashlib.sha256(("GWLR-SPEED-PILOT-01:" + s).encode()).hexdigest())
    selected = eligible.sort_values(["selection_hash", "event_name"]).head(6)
    selected.to_csv(ROOT / "contracts/selected_events.csv", index=False)
    contract = dict(
        id="GWLR-SPEED-PILOT-01", status="EXPLORATORY_COMPONENT_TIMING_NOT_SPEEDUP_CLAIM",
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        selection="First 6 SHA256-sorted strict-scope O3 event names; no quality/rank-based replacements",
        phazap_version="0.3.3", posterior_samples=[512, 1024], sample_seed=2026091701,
        flow_Hz=20, fbest_Hz=40, fhigh_Hz=100,
        pair_set="All 15 unordered within-pilot pairs, no filtering by outcome",
        repeats_cached_pair=20, repeats_cached_catalog=30, threads=2,
        stability_screen=dict(max_DJ_abs_difference=0.2, min_spearman=0.95,
                              meaning="Pilot readiness heuristic, not a publication-level convergence proof"),
        comparison_boundaries=["Public PE creation excluded and not claimed free",
            "Phazap phase construction timed separately from cached pair comparison",
            "LensRank cached-channel fusion timed, not full inference",
            "16s matching component separate, not full waveform branch",
            "No full PO until joint sampling-prior and coordinate contract validated",
            "No full PE, Hanabi, retraining, reweighting or historical file writes",
            "No recall on real data; public candidates not truth labels"],
        no_speedup_ratio=True, no_new_blind_test=True,
        final_status="HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE")
    js(contract_path, contract)
    js(ROOT / "contracts/CONTRACT_SHA256.json", {"sha256": sha(contract_path)})
    host = dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)),
                RAM_bytes=psutil.virtual_memory().total, cwd=str(ROOT),
                cpu_model=next((l.strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                                if l.startswith("model name")), "unknown"))
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
        host[f] = Path(f).read_text().strip() if Path(f).exists() else "unavailable"
    js(ROOT / "contracts/HARDWARE.json", host)
    return selected


def cached_rankings():
    freeze = json.loads(track(A / "independent_confirmation/contracts/FRESH_FREEZE.json").read_text())
    for dep in ("gwtc3", "gwtc4"):
        paths = {seed: track(frozen_table(dep, seed)) for seed in (202607241, 202607242, 202607243)}
        columns = ["pair_key", "upstream_joint_waveform", "time_score", "sky_raw_log_bf"]
        tables = measure("LensRank_channel_table_IO", dep,
                         lambda: {s: pd.read_parquet(p, columns=columns) for s, p in paths.items()})
        base = measure("LensRank_cached_3seed_fusion_consensus", dep,
                       lambda: rank_catalog(tables, freeze["recipes"], dep), n_pairs=len(tables[202607241]))
        for rep in range(1, 31):
            result = measure("LensRank_cached_3seed_fusion_consensus", dep,
                             lambda: rank_catalog(tables, freeze["recipes"], dep), rep,
                             n_pairs=len(base))
            if not result.equals(base):
                raise RuntimeError("Non-deterministic cached ranking")
        base.to_csv(ROOT / f"results/{dep}_path1_replayed_ranks.csv", index=False)
        check(dep + "_cached_rank_determinism", True, repeats=30, pairs=len(base))
        check(dep + "_new_score_finite", np.isfinite(base.mean_score).all())


def decode(ds):
    v = np.asarray(ds[()]).reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def read_pe(row):
    path = track(row.sky_map_path)
    with h5py.File(path, "r") as f:
        group = f[row.sky_map_internal_group]
        cfg = group["config_file"]
        if "config" in cfg and "reference-frequency" in cfg["config"]:
            cfg = cfg["config"]
            meta = dict(reference_frequency=float(decode(cfg["reference-frequency"])),
                        sampling_frequency=float(decode(cfg["sampling-frequency"])),
                        duration=float(decode(cfg["duration"])),
                        waveform_approximant=decode(cfg["waveform-approximant"]),
                        ifo_list=ast.literal_eval(decode(cfg["detectors"])))
        elif "engine" in cfg:
            import lalsimulation
            e = cfg["engine"]
            approx = decode(e["approx"])
            approx = lalsimulation.GetStringFromApproximant(lalsimulation.GetApproximantFromString(approx))
            meta = dict(reference_frequency=float(decode(e["fref"])),
                        sampling_frequency=float(decode(e["srate"])), duration=float(decode(e["seglen"])),
                        waveform_approximant=approx, ifo_list=ast.literal_eval(decode(cfg["analysis/ifos"])))
        else:
            raise ValueError("Unsupported original PE configuration; no guessed metadata")
        from phazap.pe_input import _required_parameters
        arr = group["posterior_samples"][()]
        data = pd.DataFrame({p: arr[p] for p in _required_parameters})
        if len(data) < 1024 or not np.isfinite(data.to_numpy()).all():
            raise ValueError("Insufficient or invalid posterior samples")
        priors = {}
        if "priors/analytic" in group:
            for key, item in group["priors/analytic"].items():
                if isinstance(item, h5py.Dataset) and not key.startswith("recalib"):
                    priors[key] = decode(item)
        js(ROOT / f"contracts/{row.event_name}_PE_provenance.json",
           dict(file=str(path), group=row.sky_map_internal_group, metadata=meta,
                n_samples=len(data), sampling_priors=priors,
                group_source="Frozen official70 sky_map_internal_group, explicit; no automatic alternative"))
    return data, meta


def phazap_pilot(selected):
    sys.path.insert(0, str(ROOT / "vendor/site"))
    from phazap.pe_input import ParameterEstimationInput
    from phazap.postprocess_phase import postprocess_phase
    from phazap import phazap
    track(ROOT / "vendor/phazap-0.3.3-py3-none-any.whl")
    phases = {}
    for row in selected.itertuples(index=False):
        print("EVENT", row.event_name, flush=True)
        try:
            samples, meta = measure("PE_HDF5_read_and_metadata", row.event_name, lambda: read_pe(row))
            seed = int(hashlib.sha256((row.event_name + ":2026091701").encode()).hexdigest()[:16], 16)
            indices = np.random.default_rng(seed).permutation(len(samples))[:1024]
            js(ROOT / f"contracts/{row.event_name}_sample_indices.json", indices.tolist())
            for n in (512, 1024):
                pe = ParameterEstimationInput(samples.iloc[indices[:n]].reset_index(drop=True), **meta)
                phases[row.event_name, n] = measure("Phazap_posterior_to_phase", row.event_name,
                    lambda: postprocess_phase(pe, flow=20, fhigh=100, fbest=40,
                        superevent_name=row.event_name, label=f"pilot_n{n}",
                        output_dir=str(ROOT / "phases"), output_filename=f"{row.event_name}_n{n}.hdf5"),
                    posterior_samples=n, includes_file_write=True)
            check(row.event_name + "_phase_generation", True)
        except Exception:
            error = traceback.format_exc()
            (ROOT / f"logs/{row.event_name}_FAIL.txt").write_text(error)
            check(row.event_name + "_phase_generation", False, error=error)
    pair_rows = []
    for a, b in itertools.combinations(selected.event_name, 2):
        for n in (512, 1024):
            if (a, n) not in phases or (b, n) not in phases:
                continue
            try:
                fn = lambda: phazap(phases[a, n], phases[b, n], plot=False)
                value = measure("Phazap_cached_pair", a + "--" + b, fn, posterior_samples=n)
                for rep in range(1, 21):
                    repeat = measure("Phazap_cached_pair", a + "--" + b, fn, rep, posterior_samples=n)
                    if not np.allclose(value, repeat, atol=1e-12, rtol=0):
                        raise RuntimeError("Pair nondeterminism")
                swapped = phazap(phases[b, n], phases[a, n], plot=False)
                check(f"swap_{a}_{b}_{n}", np.isclose(value[0], swapped[0], atol=1e-10, rtol=1e-10),
                      DJ=float(value[0]), swapped_DJ=float(swapped[0]))
                pair_rows.append(dict(event_i=a, event_j=b, n_samples=n, DJ=float(value[0]),
                                      volume=float(value[1]), phase_shift=float(value[2]),
                                      upstream_p_value=float(value[4])))
            except Exception:
                check(f"pair_{a}_{b}_{n}", False, error=traceback.format_exc())
    data = pd.DataFrame(pair_rows)
    data.to_csv(ROOT / "tables/phazap_pairs.csv", index=False)
    if not data.empty:
        joined = data[data.n_samples == 512].merge(data[data.n_samples == 1024],
            on=["event_i", "event_j"], suffixes=("_512", "_1024"))
        joined["abs_delta_DJ"] = abs(joined.DJ_512 - joined.DJ_1024)
        joined.to_csv(ROOT / "tables/posterior_sample_sensitivity.csv", index=False)
        from scipy.stats import spearmanr
        rho = float(spearmanr(joined.DJ_512, joined.DJ_1024).statistic)
        delta = float(joined.abs_delta_DJ.max())
        check("512_vs_1024_pilot_readiness", len(joined) == 15 and delta <= .2 and rho >= .95,
              n_pairs=len(joined), max_abs_delta_DJ=delta, spearman=rho)
    check("all_6_events_and_30_pair_settings", len(phases) == 12 and len(data) == 30,
          phase_products=len(phases), pair_settings=len(data))


def finish():
    rows = []
    for path, info in INPUTS.items():
        after = sha(path)
        rows.append(dict(path=path, **info, sha256_after=after, unchanged=after == info["sha256_before"]))
    pd.DataFrame(rows).to_csv(ROOT / "manifest/INPUT_SHA256.csv", index=False)
    check("all_tracked_inputs_unchanged", all(r["unchanged"] for r in rows), n_files=len(rows))
    t = pd.DataFrame(TIMES)
    if not t.empty:
        t["posterior_samples"] = t.get("posterior_samples", pd.Series(index=t.index, dtype=float)).fillna(0)
        summary = t.groupby(["stage", "unit", "posterior_samples"]).agg(
            n=("wall_seconds", "size"), median_s=("wall_seconds", "median"),
            p90_s=("wall_seconds", lambda x: x.quantile(.9)), min_s=("wall_seconds", "min"),
            max_s=("wall_seconds", "max"), median_cpu_s=("process_cpu_seconds", "median"))
        summary.to_csv(ROOT / "tables/timing_summary.csv")
    js(ROOT / "results/FINAL_STATUS.json", dict(status="HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
        checks=CHECKS, full_pipeline_speedup_demonstrated=False, full_PO_implemented=False,
        real_catalog_recall_available=False))


def main():
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    ROOT = args.root.resolve()
    selected = prepare()
    try:
        cached_rankings()
        phazap_pilot(selected)
    except Exception:
        (ROOT / "logs/pilot_FAIL.txt").write_text(traceback.format_exc())
        check("top_level_execution", False, error=traceback.format_exc())
    finally:
        finish()


if __name__ == "__main__":
    main()
