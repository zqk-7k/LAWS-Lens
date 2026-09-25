#!/usr/bin/env python3
"""R76 exploratory, simulation-selected compatibility gate.

Only simulation development/validation selects the gate parameters. Time, sky,
encoders and frozen outer weights are unchanged.
"""
from pathlib import Path
import json, hashlib, tarfile, math
import numpy as np
import pandas as pd

P = Path("/root/autodl-tmp/gw-catalog")
R75 = P / "results/mcwf_shared_mass_catalog_75_20260910T143000Z"
R73 = P / "results/mcwf_shared_mass_conditional_73_20260910T110300Z"
OUT = P / "results/mcwf_r76_compatibility_gate_20260911T000000Z"
STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"
DEPS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
GAMMAS = (0.0, .02, .05, .1, .2, .4, .8)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_dev(dep, seed):
    f = pd.read_parquet(R73 / "tables/PREDICTIONS.parquet")
    return f[(f.deployment == dep) & (f.seed == seed) &
             (f.arm == "FULL-SHARED-NUMERICAL-CONTROL")].copy()

def gate_score(w, d, tau, gamma):
    w = np.asarray(w, float)
    d = np.asarray(d, float)
    out = w.copy()
    scale = max(float(tau), 1.0)
    m = np.isfinite(d) & (d > tau) & np.isfinite(w) & (w > 0)
    out[m] = w[m] * np.exp(-gamma * (d[m] - tau) / scale)
    return out

def logloss(y, s):
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return -(y * np.log(np.clip(p, 1e-12, 1)) +
             (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))

def metrics(f, score, mode):
    f = f.copy()
    f["snew"] = score
    vals = []
    for _, g in f.groupby("idx_i", sort=False):
        y = g.sort_values("snew", ascending=False, kind="mergesort").is_true_pair.to_numpy(bool)
        vals.append([float(y[:k].any()) for k in (1, 5, 10)])
    a = np.asarray(vals, float) if vals else np.zeros((0, 3))
    labels = f.is_true_pair.to_numpy(bool)
    order = np.argsort(-f.snew.to_numpy(float), kind="mergesort")
    yy = labels[order].astype(float)
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    ap = float(np.sum(prec[yy == 1]) / max(1, yy.sum()))
    def false_at(rec):
        ntrue = max(1, int(labels.sum()))
        idx = np.flatnonzero(tp >= rec * ntrue)
        used = idx[0] + 1 if len(idx) else len(yy)
        return float(max(0, used - int(math.ceil(rec * ntrue))))
    return {"mode": mode, "queries": int(len(a)), "true_pairs": int(labels.sum()),
            "R@1": float(a[:, 0].mean()) if len(a) else 0.,
            "R@5": float(a[:, 1].mean()) if len(a) else 0.,
            "R@10": float(a[:, 2].mean()) if len(a) else 0.,
            "AUPRC": ap, "F50": false_at(.5), "F90": false_at(.9)}

def select():
    OUT.mkdir(parents=True, exist_ok=False)
    for d in ("contracts", "configs", "tables", "results", "reports", "figures",
              "manifest", "scripts", "logs"):
        (OUT / d).mkdir()
    contract = {
        "id": "MCWF-R76-COMPATIBILITY-GATE", "status": STATUS, "goal_achieved": False,
        "question": "Can simulation-selected monotone attenuation reduce catastrophic intrinsic-mass false positives without changing time or sky?",
        "frozen": ["R75 encoder", "Z_wf input", "Z_time", "Z_sky", "outer weights",
                   "scope/splits", "all historical outputs"],
        "formula": "w_new=w_old for w_old<=0 or D<=tau; otherwise w_new=w_old*exp(-gamma*(D-tau)/max(tau,1))",
        "deficit": "D_mass from R74 physical measurement, not PE and not a Bayes factor",
        "grid": {"gamma": list(GAMMAS), "tau": "q50/q75/q90 of positive fit-fold D_mass per deployment"},
        "selection": "Per deployment and model seed, minimize fold-1 simulation logloss; real fields are not read during selection.",
        "guard": "Only retain a candidate if fold-1 loss is no worse than identity; attenuation is monotone.",
        "interpretation": "Adaptive development; not independent confirmation. No paper or v9.3 overwrite."
    }
    write(OUT / "contracts/ANALYSIS_CONTRACT.json", contract)
    files = [Path(__file__), R75 / "contracts/FINAL_AUDIT.json",
             R75 / "tables/RETRIEVAL_PER_MODEL.csv", R73 / "tables/PREDICTIONS.parquet",
             R73 / "configs/ALL_CONDITIONAL_MODELS.json"]
    pd.DataFrame([{"path": str(p), "sha256": sha(p)} for p in files]).to_csv(
        OUT / "manifest/INPUT_SHA256.csv", index=False)
    chosen, grid, preds = [], [], []
    for dep in DEPS:
        for seed in SEEDS:
            f = load_dev(dep, seed)
            pos = f[(f.fold == 0) & (f.kind == "true")].D_mass.dropna()
            taus = [float(np.quantile(pos, q)) for q in (.5, .75, .9)]
            best = None
            tune = f.fold == 1
            y = f.loc[tune, "kind"].eq("true").to_numpy(float)
            for ti, tau in enumerate(taus):
                for gamma in GAMMAS:
                    ss = gate_score(f.waveform_score, f.D_full_common_denominator, tau, gamma)
                    loss = float(np.mean(logloss(y, ss[tune])))
                    row = {"deployment": dep, "seed": seed, "tau_index": ti,
                           "tau": tau, "gamma": gamma, "fold1_logloss": loss}
                    grid.append(row)
                    key = (loss, 0 if gamma == 0 else 1, gamma, ti)
                    if best is None or key < best[0]:
                        best = (key, row)
            row = best[1]
            row["selected"] = True
            chosen.append(row)
            ss = gate_score(f.waveform_score, f.D_full_common_denominator,
                            row["tau"], row["gamma"])
            for fold in (0, 1):
                m = f.fold == fold
                z = f.loc[m].copy()
                z["r76_waveform"] = ss[m]
                z["seed"] = seed
                z["deployment"] = dep
                z["fold_eval"] = fold
                preds.append(z[["pair_id", "deployment", "seed", "fold_eval", "kind",
                                "D_mass", "D_full_common_denominator", "waveform_score",
                                "r76_waveform"]])
            print("SELECT", dep, seed, row, flush=True)
    pd.DataFrame(grid).to_csv(OUT / "tables/CALIBRATION_GRID.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / "configs/SELECTED_GATE.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(
        OUT / "tables/SIMULATION_PREDICTIONS.parquet", index=False)
    return chosen

def score_all(chosen):
    rows, real_rows, keyrows = [], [], []
    lookup = {(x["deployment"], int(x["seed"])): x for x in chosen}
    for dep in DEPS:
        for seed in SEEDS:
            cfg = lookup[dep, seed]
            base = R75 / "results/R73-FROZEN-CONDITIONAL-WF" / dep / f"seed_{seed}"
            for p in sorted(base.glob("*/pairs.parquet")):
                f = pd.read_parquet(p)
                if "waveform_score" not in f or "r75_D_full_common_denominator" not in f:
                    continue
                nw = gate_score(f.waveform_score, f.r75_D_full_common_denominator,
                                cfg["tau"], cfg["gamma"])
                oldf = f.waveform_contribution.to_numpy(float)
                oldw = f.waveform_score.to_numpy(float)
                ratio = np.divide(oldf, oldw, out=np.zeros(len(f)), where=np.abs(oldw) > 1e-12)
                nf = f.final_score.to_numpy(float) - oldf + nw * ratio
                zero = np.abs(oldw) <= 1e-12
                nf[zero] = f.final_score.to_numpy(float)[zero]
                rec = metrics(f, nw, "waveform")
                rec.update({"deployment": dep, "seed": seed, "panel": p.parent.name,
                            "method": "R76_GATE", "tau": cfg["tau"], "gamma": cfg["gamma"]})
                rows.append(rec)
                rec = metrics(f, nf, "fusion")
                rec.update({"deployment": dep, "seed": seed, "panel": p.parent.name,
                            "method": "R76_GATE"})
                real_rows.append(rec)
            p = base / "real/fusion_all_pairs.parquet"
            if p.exists():
                f = pd.read_parquet(p)
                nw = gate_score(f.waveform_score, f.r75_D_full_common_denominator,
                                cfg["tau"], cfg["gamma"])
                oldw = f.waveform_score.to_numpy(float)
                oldf = f.waveform_contribution.to_numpy(float)
                ratio = np.divide(oldf, oldw, out=np.zeros(len(f)), where=np.abs(oldw) > 1e-12)
                nf = f.final_score.to_numpy(float) - oldf + nw * ratio
                q = f[f.pair_key == "GW191103_012549--GW191105_143521"]
                if len(q):
                    pos = int(q.index[0])
                    keyrows.append({"deployment": dep, "seed": seed,
                                    "old_waveform": float(oldw[pos]), "new_waveform": float(nw[pos]),
                                    "old_final": float(f.final_score.iloc[pos]), "new_final": float(nf[pos])})
                out = f.assign(r76_waveform=nw, r76_final_score=nf).sort_values(
                    "r76_final_score", ascending=False)
                for rank, (_, r) in enumerate(out.head(50).iterrows(), 1):
                    real_rows.append({"deployment": dep, "seed": seed, "panel": "real_top50",
                                      "method": "R76_GATE", "rank": rank,
                                      "pair_key": r.get("pair_key"), "r76_final_score": r.r76_final_score,
                                      "r76_waveform": r.r76_waveform,
                                      "pe_mc_bc": r.get("pe_mc_bhattacharyya_coefficient"),
                                      "pe_dmax": r.get("pe_dmax_intrinsic"),
                                      "official_frontend": r.get("official_frontend"),
                                      "official_hanabi": r.get("official_hanabi_figure_overlap",
                                                                  r.get("official_fast_golum_table_overlap"))})
    pd.DataFrame(rows).to_csv(OUT / "tables/INJECTION_WAVEFORM_METRICS.csv", index=False)
    pd.DataFrame(real_rows).to_csv(OUT / "tables/INJECTION_FUSION_AND_REAL_AUDIT.csv", index=False)
    pd.DataFrame(keyrows).to_csv(OUT / "tables/KEY_PAIR_AUDIT.csv", index=False)

def report(chosen):
    g = pd.DataFrame(chosen)
    inj = pd.read_csv(OUT / "tables/INJECTION_WAVEFORM_METRICS.csv")
    summary = inj.groupby(["deployment", "method", "mode"])[
        ["R@1", "R@5", "R@10", "AUPRC", "F50", "F90"]].agg(["mean", "std"]).reset_index()
    (OUT / "tables/INJECTION_WAVEFORM_SUMMARY.csv").write_text(
        summary.to_csv(index=False), encoding="utf-8")
    text = ["# R76 兼容性门控探索", "", "状态：" + STATUS, "",
            "本轮只在模拟 development/validation 上选择 tau/gamma；时间、天空、encoder、外层权重没有修改。",
            "门控只衰减正波形证据，不能称为 PE 或 Bayes factor。", "",
            "## 选择结果", g.to_markdown(index=False), "",
            "## 注入波形审计", summary.to_markdown(index=False),
            "", "真实 Top-50 仅作冻结后的审计，不能用于反向调参。",
            "完整真实审计见 tables/INJECTION_FUSION_AND_REAL_AUDIT.csv。", "",
            "本轮不替换 v9.3/R75，不修改论文。", STATUS]
    (OUT / "reports/R76_REPORT_CN.md").write_text("\n".join(text), encoding="utf-8")

def package():
    pkg = P / "packages/mcwf_r76_compatibility_gate_20260911T000000Z.tar.gz"
    if pkg.exists():
        raise RuntimeError("package exists")
    manifest = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.suffix in (".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth"):
            continue
        manifest.append((p, OUT.name + "/" + str(p.relative_to(OUT)), sha(p)))
    with (OUT / "manifest/DELIVERY_SHA256SUMS.txt").open("w") as f:
        for _, a, h in manifest:
            f.write(h + "  " + a + "\n")
    pkg.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pkg, "w:gz") as t:
        for p, a, _ in manifest:
            t.add(p, arcname=a)
    write(str(pkg) + ".sha256", {"sha256": sha(pkg), "bytes": pkg.stat().st_size,
                                 "members": len(manifest)})
    return pkg

if __name__ == "__main__":
    chosen = select()
    score_all(chosen)
    report(chosen)
    print("PACKAGE", package(), flush=True)
