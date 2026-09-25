"""Isolated SI-unit repair; original inputs and all ranking coefficients stay frozen."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

P = Path("/root/autodl-tmp/gw-catalog")
A = P / "results/mcwf_unified_path875_devconf_20260908T181500Z"
B = P / "results/o4b_hl_bayestar_new_score_only_20260912T072746Z"
OLD = P / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z/scripts/bayestar_injection_sky_full_experiment.py"
FIX = Path(__file__).with_name("bayestar_injection_sky_full_experiment.py")
STATE = {}
SOURCE_COLUMNS = ["m1_det", "m2_det", "theta_jn", "phijl", "tilt1", "tilt2", "phi12", "a1", "a2", "phase", "psi", "dl_source"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def disk_guard(root):
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise RuntimeError("HOLD_DISK_LIMIT: preserve at least20GiB; no historical deletion")


def protected_read(path, hashes):
    path = Path(path)
    hashes[str(path)] = sha(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def verify(root):
    c = json.loads((root / "contracts/ANALYSIS_CONTRACT.json").read_text())
    if sha(root / "contracts/ANALYSIS_CONTRACT.json") != (root / "contracts/FREEZE.sha256").read_text().strip():
        raise RuntimeError("Analysis contract changed")
    for path, expected in c["protected_inputs"].items():
        if sha(path) != expected:
            raise RuntimeError("Protected input changed: " + path)
    return c


def prepare(root):
    if root.exists():
        raise RuntimeError("Fresh output directory required")
    root.mkdir(parents=True)
    for name in ("contracts", "maps", "pilot", "tables", "logs", "reports", "scripts", "results"):
        (root / name).mkdir()
    tests = Path(__file__).with_name("test_spin_units.py")
    check = subprocess.run([sys.executable, "-B", str(tests)], capture_output=True, text=True)
    (root / "logs/unit_tests.log").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError("Spin unit tests failed")
    write(root / "contracts/UNIT_TESTS.json", {"state": "PASS", "test_sha256": sha(tests), "source_sha256": sha(FIX)})
    hashes = {str(OLD): sha(OLD), str(FIX): sha(FIX), str(Path(__file__)): sha(Path(__file__))}
    hashes[str(tests)] = sha(tests)
    old_tree, new_tree = ast.parse(OLD.read_text()), ast.parse(FIX.read_text())
    for tree in (old_tree, new_tree):
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "spin_components")]
    if ast.dump(old_tree) != ast.dump(new_tree):
        raise RuntimeError("Changes outside spin_components are forbidden")
    freeze = protected_read(A / "independent_confirmation/contracts/FRESH_FREEZE.json", hashes)
    delivery = protected_read(A / "contracts/DELIVERY_CONTRACT.json", hashes)
    physical_root = Path(delivery["confirmation_root"])
    temps = protected_read(OLD.parents[1] / "contracts/selected_config.json", hashes)
    groups = []
    for dep in ("gwtc3", "gwtc4"):
        noise = physical_root / f"confirmation/{dep}/noise"
        archived_noise = A / f"independent_confirmation/confirmation/{dep}/noise/noise_manifest.csv"
        protected_read(archived_noise, hashes)
        protected_read(noise / "noise_manifest.csv", hashes)
        if sha(archived_noise) != sha(noise / "noise_manifest.csv"):
            raise RuntimeError("Physical noise bank does not match the delivery")
        for cs in (202609941, 202609942, 202609943):
            parent = A / f"independent_confirmation/confirmation/{dep}/catalog_{cs}"
            physical = physical_root / f"confirmation/{dep}/catalog_{cs}"
            events = protected_read(parent / "event_manifest.parquet", hashes)
            sources = protected_read(parent / "source_systems.parquet", hashes).set_index("source_uid")
            for filename in ("event_manifest.parquet", "source_systems.parquet"):
                protected_read(physical / filename, hashes)
                if sha(parent / filename) != sha(physical / filename):
                    raise RuntimeError("Physical catalog does not match the delivery")
            records = []
            for event in events.to_dict("records"):
                records.append({
                    "record": {k: event[k] for k in ["idx", "event_uid", "source_uid", "family_slot", "gps_obs", "ra_true", "dec_true", "noise_bank_index", "target_network_snr", "measurement_seed", "morse_index"]},
                    "source": {k: float(sources.loc[event["source_uid"], k]) for k in SOURCE_COLUMNS},
                    "old_moc": str(physical / f"events/{event['event_uid']}.fits"),
                    "old_moc_sha256": event["moc_sha256"],
                    "old_dense": str(physical / f"events/{event['event_uid']}_sky512.npy"),
                    "old_waveform": event["sky_audit"]["waveform"],
                })
            recipes = []
            for r in freeze["recipes"]:
                if r["deployment"] != dep:
                    continue
                pair = parent / f"model_{r['slot']}/PATH25_pairs.parquet"
                hashes[str(pair)] = sha(pair)
                recipes.append({**r, "pair_path": str(pair), "temperature": temps["deployments"][dep][str(r["seed"])]["posterior_temperature"]})
            groups.append({"id": f"{dep}_{cs}", "deployment": dep, "catalog": cs,
                           "psd": str(noise / "psd.npy"), "frequency": str(noise / "frequency.npy"),
                           "records": records, "recipes": recipes, "split": "published_confirmation_recomputation"})
    source_path = next((P.parent / "GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1").glob("*_SourceParams.csv"))
    sources = protected_read(source_path, hashes)
    weights = protected_read(B / "tables/selected_waveform_and_fusion_weights.csv", hashes)
    noise = B / "workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared"
    for seed in (2026091221, 2026091222, 2026091223):
        parent = B / f"event_maps/seed_{seed}/test"
        events = protected_read(parent / "event_metrics.parquet", hashes)
        records = []
        for event in events.to_dict("records"):
            record = {k: event[k] for k in ["idx", "event_uid", "gps_obs", "ra_true", "dec_true", "noise_bank_index", "target_network_snr", "measurement_seed", "morse_index"]}
            record.update(source_uid=str(event["global_source_id"]), family_slot=event["family"])
            records.append({"record": record,
                            "source": {k: float(sources.iloc[int(event["gwlmc_row"])][k]) for k in SOURCE_COLUMNS},
                            "old_moc": event["moc_path"], "old_moc_sha256": event["moc_sha256"],
                            "old_waveform": event["waveform"]})
        w = weights[(weights.seed == seed) & (weights.constraint == "POSITIVE")].iloc[0]
        t = protected_read(B / f"event_maps/seed_{seed}/validation/TEMPERATURE_SELECTED.json", hashes)
        pair = B / f"results/injection/seed_{seed}/all_pair_scores.parquet"
        hashes[str(pair)] = sha(pair)
        groups.append({"id": f"o4b_{seed}", "deployment": "o4b", "catalog": seed, "split": "published_test_recomputation",
                       "psd": str(noise / "noise_psd_bank.npy"), "frequency": str(noise / "noise_psd_frequency.npy"),
                       "records": records, "recipes": [{"seed": seed, "slot": seed, "pair_path": str(pair), "temperature": t["temperature"],
                                                        "upstream_weights": [float(w.waveform), float(w.time), float(w.sky)]}]})
    for g in groups:
        for name in ("psd", "frequency"):
            if g[name] not in hashes:
                hashes[g[name]] = sha(g[name])
        if [r["record"]["idx"] for r in g["records"]] != list(range(len(g["records"]))):
            raise RuntimeError("Non-contiguous event indexing")
    contract = {"id": "GWLR-SKY-SI-FIX-01", "utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Fix only the low-level spin-conversion SI units in published O3/O4a/O4b injection sky; no tuning",
                "analysis_nside": 512, "ordering": "NESTED probability mass, same for both operands",
                "map_temperatures": "unchanged archived values; isolates code correction, not a newly calibrated model",
                "frozen": ["source", "event time", "PSD", "measurement_seed", "waveform score", "time score", "fusion weights", "scope"],
                "training": False, "weight_search": False, "real_PE_for_selection": False,
                "limitations": ["Reused open data, not a new blind test", "Conditional Gaussian triggers, not identical-noisy-strain PE",
                                "O3 archived cumulative response calendar retained", "O4b500x190 manifest unavailable; original450 scope only"],
                "pilot_rule": "First complete system of each family in first catalog/model per run; legacy replay before correction",
                "pilot_legacy_replay_max_pixel_abs": 1e-9, "pilot_legacy_replay_L1": 1e-6,
                "disk_min_free_GiB": 20, "native_maps_only_persistent": True,
                "protected_inputs": hashes, "groups": groups,
                "final_status": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"}
    write(root / "contracts/ANALYSIS_CONTRACT.json", contract)
    (root / "contracts/FREEZE.sha256").write_text(sha(root / "contracts/ANALYSIS_CONTRACT.json") + "\n")
    for file in [Path(__file__), FIX, Path(__file__).with_name("test_spin_units.py"), Path(__file__).with_name("lensrank_main_results_audit_20260916.py")]:
        shutil.copy2(file, root / "scripts" / file.name)
    print(json.dumps({"state": "FROZEN", "groups": len(groups), "events": sum(len(g['records']) for g in groups), "root": str(root)}), flush=True)


def worker_init(root):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    logging.getLogger("ligo.skymap").setLevel(logging.ERROR)
    logging.getLogger("bilby").setLevel(logging.ERROR)
    import ligo.skymap.core as core
    core.set_num_threads(1)
    STATE.update(root=Path(root), fixed=load(FIX, "si_fixed_sky"), old=load(OLD, "si_legacy_sky"), banks={})


def simulate(module, record, source):
    try:
        sky, audit = module._simulate_with_waveform(record, source, module.PRIMARY_WAVEFORM)
        return sky, audit, False, None
    except Exception as exc:
        sky, audit = module._simulate_with_waveform(record, source, module.FALLBACK_WAVEFORM)
        return sky, audit, True, repr(exc)


def map_job(job):
    from ligo.skymap.io.fits import write_sky_map, read_sky_map
    group, item, replay = job
    root, fixed, old = STATE["root"], STATE["fixed"], STATE["old"]
    record = item["record"]
    out = root / "maps" / group["id"]
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f"event_{record['idx']:04d}.json"
    path = out / f"event_{record['idx']:04d}.fits.gz"
    if marker.exists():
        result = json.loads(marker.read_text())
        if sha(path) != result["moc_sha256"]:
            raise RuntimeError("New output hash mismatch")
        if not replay or "legacy_replay" in result:
            return result
    disk_guard(root)
    key = group["psd"]
    if key not in STATE["banks"]:
        STATE["banks"][key] = (np.load(group["frequency"]), np.load(key, mmap_mode="r"))
    freq, bank = STATE["banks"][key]
    for module in (old, fixed):
        module._WORKER_PSD_FREQ, module._WORKER_PSD_BANK = freq, bank
    started = time.monotonic()
    try:
        if sha(item["old_moc"]) != item["old_moc_sha256"]:
            raise RuntimeError("Archived native MOC changed")
        src = pd.Series(item["source"])
        legacy = None
        if replay:
            oldsky, oldaudit, _, _ = simulate(old, record, src)
            p = old.raster_probability(oldsky, 512)
            expected = old.raster_probability(read_sky_map(item["old_moc"], moc=True), 512)
            legacy = {"max_pixel_abs": float(np.max(abs(p - expected))), "L1": float(np.sum(abs(p - expected))),
                      "waveform": oldaudit["waveform"], "archived_waveform": item["old_waveform"]}
            if legacy["max_pixel_abs"] > 1e-9 or legacy["L1"] > 1e-6 or legacy["waveform"] != item["old_waveform"]:
                raise RuntimeError("Legacy replay failed: " + json.dumps(legacy))
        sky, audit, fallback, error = simulate(fixed, record, src)
        density = np.asarray(sky["PROBDENSITY"], dtype=np.float64)
        if not np.isfinite(density).all() or (density < 0).any():
            raise RuntimeError("Invalid raw probability density; do not silently repair")
        prob = fixed.raster_probability(sky, 512)
        metrics = fixed.posterior_metrics(prob, record["ra_true"], record["dec_true"])
        temp = path.with_name(path.name + ".tmp.fits.gz")
        write_sky_map(temp, sky, nest=True)
        # Verify serialization and probability units before committing the MOC.
        back = fixed.raster_probability(read_sky_map(temp, moc=True), 512)
        if not np.array_equal(prob, back):
            raise RuntimeError("MOC write/read probability changed")
        temp.replace(path)
        original = fixed.raster_probability(read_sky_map(item["old_moc"], moc=True), 512)
        result = {"group": group["id"], **record, **audit, **metrics, "moc_path": str(path), "moc_sha256": sha(path),
                  "sky_L1_vs_archived": float(np.sum(abs(prob - original))), "fallback_used": fallback,
                  "primary_error": error, "wall_seconds": time.monotonic() - started,
                  "ordering": "NESTED", "analysis_nside": 512, "mass_units_at_spin_API": "SI kg"}
        if legacy is not None:
            result["legacy_replay"] = legacy
        write(marker, result)
        return result
    except Exception:
        write(out / f"event_{record['idx']:04d}.failure_{time.time_ns()}.json", {"record": record, "traceback": traceback.format_exc()})
        raise


def jobs_for_pilot(c):
    jobs = []
    for dep in ("gwtc3", "gwtc4", "o4b"):
        group = next(g for g in c["groups"] if g["deployment"] == dep)
        families = {}
        for item in group["records"]:
            r = item["record"]
            families.setdefault(r["family_slot"], r["source_uid"])
        for item in group["records"]:
            if item["record"]["source_uid"] in families.values():
                jobs.append((group, item, True))
    return jobs


def generate(root, workers, pilot):
    c = verify(root)
    if pilot:
        jobs = jobs_for_pilot(c)
    else:
        if json.loads((root / "pilot/PASS.json").read_text())["state"] != "PASS":
            raise RuntimeError("Pilot must pass first")
        jobs = [(g, item, False) for g in c["groups"] for item in g["records"]]
    small_groups = {g["id"]: {k: v for k, v in g.items() if k not in ("records", "recipes")} for g in c["groups"]}
    jobs = [(small_groups[g["id"]], item, replay) for g, item, replay in jobs]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker_init, initargs=(str(root),)) as pool:
        futures = [pool.submit(map_job, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or pilot or len(rows) == len(jobs):
                state = {"phase": "pilot" if pilot else "maps", "complete": len(rows), "total": len(jobs), "last_group": rows[-1]["group"]}
                write(root / "PROGRESS.json", state)
                print(json.dumps(state), flush=True)
    frame = pd.DataFrame(rows).sort_values(["group", "idx"])
    frame.to_parquet(root / ("pilot/events.parquet" if pilot else "tables/map_events.parquet"), index=False)
    if pilot:
        write(root / "pilot/PASS.json", {"state": "PASS", "events": len(rows), "legacy_replay_pass": True,
                                         "unit_tests_required_separately": True, "physics_fix_only_no_performance_gate": True})
    else:
        write(root / "MAPS_COMPLETE.json", {"state": "PASS", "events": len(rows), "maps_sha256": sha(root / "tables/map_events.parquet")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("prepare", "pilot", "maps", "score"), required=True)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage in ("pilot", "maps"):
        generate(args.root, args.workers, args.stage == "pilot")
    else:
        from score_repair import run
        run(args.root)


if __name__ == "__main__":
    main()
