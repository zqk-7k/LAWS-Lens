"""Verify and package the completed repair without including raw strain or secrets."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

import pandas as pd
import repair_experiment as r


def run(root):
    r.verify(root)
    status = json.loads((root / "RESULT_STATUS.json").read_text())
    if not status["fixed_weight_comparison_complete"]:
        raise RuntimeError("Incomplete result; do not package as finished")
    for path, expected in json.loads((root / "contracts/REAL_RANKING_READONLY_MANIFEST.json").read_text()).items():
        if r.sha(path) != expected:
            raise RuntimeError("Historical real ranking changed")
    score = json.loads((root / "contracts/SCORING_CONTRACT.json").read_text())
    if r.sha(score["script"]) != score["sha256"]:
        raise RuntimeError("Scoring code changed after freeze")
    maps = pd.read_parquet(root / "tables/map_events.parquet")
    if len(maps) != 2490 or maps.duplicated(["group", "idx"]).any():
        raise RuntimeError("Map completeness failed")
    for row in maps.itertuples():
        if r.sha(row.moc_path) != row.moc_sha256:
            raise RuntimeError("Native MOC hash failed")
    contract = json.loads((root / "contracts/ANALYSIS_CONTRACT.json").read_text())
    deployment = {g["id"]: g["deployment"] for g in contract["groups"]}
    old_waveforms = {(g["id"], x["record"]["idx"]): x["old_waveform"]
                     for g in contract["groups"] for x in g["records"]}
    maps["deployment"] = maps.group.map(deployment)
    maps["old_waveform"] = [old_waveforms[row.group, row.idx] for row in maps.itertuples()]
    quality = []
    for dep, part in maps.groupby("deployment"):
        quality.append({"deployment": dep, "events": len(part),
                        "fallback_events": int(part.fallback_used.sum()),
                        "waveform_approximant_changed_events": int((part.waveform != part.old_waveform).sum()),
                        "median_raw_A90_deg2": float(part.area90_deg2.median()),
                        "raw_event_HPD90_fraction_descriptive": float((part.truth_credible_level_raw <= .9).mean()),
                        "median_map_L1_vs_archived": float(part.sky_L1_vs_archived.median()),
                        "p90_map_L1_vs_archived": float(part.sky_L1_vs_archived.quantile(.9)),
                        "median_event_wall_seconds": float(part.wall_seconds.median())})
    pd.DataFrame(quality).to_csv(root / "tables/map_quality_summary.csv", index=False, encoding="utf-8-sig")
    report = root / "reports/REPAIR_AND_FIXED_SCORE_RESULTS_CN.md"
    needs_supplement = "## 地图质量补充" not in report.read_text()
    with report.open("a") as f:
        if not needs_supplement:
            quality_to_append = []
        else:
            quality_to_append = quality
        if needs_supplement:
            f.write("\n## 地图质量补充\n\n")
        for row in quality_to_append:
            f.write(f"- {row['deployment']}: {row['events']}张图，fallback={row['fallback_events']}，"
                    f"与原版波形近似器不同的事件={row['waveform_approximant_changed_events']}，"
                    f"新旧地图L1差异中位数={row['median_map_L1_vs_archived']:.6f}。\n")
        if needs_supplement:
            f.write("\nmap_quality_summary.csv的raw HPD90比例只是事件级描述统计，"
                    "并非采用冻结temperature后的校准检验；双像及跨seed重复源不是独立样本。\n")
    for p in Path(__file__).parent.glob("*.py"):
        target = root / "scripts" / p.name
        if target.exists() and r.sha(target) != r.sha(p):
            raise RuntimeError("Packaged code differs: " + p.name)
        if not target.exists():
            shutil.copy2(p, target)
    patch = "".join(difflib.unified_diff(r.OLD.read_text().splitlines(True), r.FIX.read_text().splitlines(True),
                                        fromfile="archived/bayestar_injection_sky_full_experiment.py",
                                        tofile="SI-fixed/bayestar_injection_sky_full_experiment.py"))
    (root / "reports/EXACT_SOURCE_CHANGE.patch").write_text(patch)
    maps[["group", "idx", "event_uid", "moc_path", "moc_sha256", "wall_seconds", "fallback_used"]].to_csv(
        root / "tables/NATIVE_MAP_MANIFEST.csv", index=False, encoding="utf-8-sig")
    r.write(root / "DELIVERY_STATUS.json", {
        "state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
        "code_repair_verified": True, "maps_verified": len(maps), "fixed_score_comparison_complete": True,
        "archived_real_rank_files_unchanged": True, "historical_input_hashes_unchanged": True,
        "posterior_temperature_recalibrated": False, "fusion_weights_retuned": False,
        "package_excludes": ["native MOC payloads (retained on server and indexed)", "raw strain", "PE HDF5", "private keys", "passwords"],
    })
    files = sorted(p for p in root.rglob("*") if p.is_file() and "maps" not in p.relative_to(root).parts
                   and p.name != "OUTPUT_SHA256.csv")
    for p in files:
        data = p.read_bytes()
        patterns = (
            rb"(?m)^-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----\s*$",
            rb"(?m)^\s*sshpass\s+-p(?:\s|=)",
        )
        for pattern in patterns:
            if re.search(pattern, data):
                raise RuntimeError("Sensitive content in package: " + str(p))
    manifest = pd.DataFrame([{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": r.sha(p)} for p in files])
    manifest.to_csv(root / "OUTPUT_SHA256.csv", index=False)
    package = r.P / "packages" / (root.name + "_deliverables.tar.gz")
    with package.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for p in [*files, root / "OUTPUT_SHA256.csv"]:
                archive.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    with tarfile.open(package, "r:gz") as archive:
        for row in manifest.itertuples():
            data = archive.extractfile(str(Path(root.name) / row.path)).read()
            if hashlib.sha256(data).hexdigest() != row.sha256:
                raise RuntimeError("Internal package hash failed")
    digest = r.sha(package)
    with package.with_name(package.name + ".sha256").open("x") as f:
        f.write(digest + "  " + package.name + "\n")
    print(json.dumps({"package": str(package), "sha256": digest, "bytes": package.stat().st_size,
                      "members": len(files) + 1, "all_internal_hashes_pass": True}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
