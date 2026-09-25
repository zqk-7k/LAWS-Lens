#!/usr/bin/env python3
"""
Catalog-level retrieval + physical reranking engine.

Mirrors the repository's reranking logic (matchgw/aux_priors):
  - Liao/population-calibrated time-delay likelihood-ratio score
  - observed-sky overlap (step + Gaussian log-overlap), using A90->sigma
  - SNR/amplitude consistency
  - validation-style weighted-sum fusion (row-standardised channels)

Operates purely on observables; no waveform encoder (consistent with the
paper's finding that physical priors carry the catalog-level signal, and with
the fact that real GWTC strain is out-of-distribution for the simulated encoder).

Provides:
  - pairwise feature construction
  - full-catalog retrieval recall R@k
  - false-association / threshold analysis for rarity studies
  - graph construction with abstention threshold + connected components
  - maximum-weight matching baseline
  - parameter-summary kNN baseline
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# ----------------------------------------------------------------------
# physical helpers
# ----------------------------------------------------------------------
def a90_to_sigma_rad(a90_deg2):
    """90% credible AREA (deg^2) -> 1-sigma angular radius (rad).
    For a 2D Gaussian, 90% area A => sigma = sqrt(A / (2 pi ln(10)))?
    Repo uses: 90% contour radius theta_90 with A90 = pi theta_90^2, and
    theta_90 = sigma sqrt(-2 ln(0.1)) = sigma sqrt(2 ln 10). So
    sigma = sqrt(A90/pi) / sqrt(2 ln10)."""
    theta90 = np.sqrt(np.asarray(a90_deg2) / np.pi) * DEG2RAD  # rad
    return theta90 / np.sqrt(2.0 * np.log(10.0))


def angular_sep(ra1, dec1, ra2, dec2):
    """Great-circle separation (rad). Vectorised over pairs."""
    return np.arccos(np.clip(
        np.sin(dec1) * np.sin(dec2) +
        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2), -1, 1))


# ----------------------------------------------------------------------
# time-delay population prior (Liao-style LR)
# ----------------------------------------------------------------------
class TimeDelayPrior:
    """Likelihood ratio between lensed-pair delay distribution and the
    background (unrelated-pair) delay distribution, built empirically from
    the catalog's own true lensed delays vs the uniform-window background."""

    def __init__(self, lensed_delays_s, window_s):
        d = np.asarray(lensed_delays_s)
        d = d[d > 0]
        self.logd = np.log10(d)
        # KDE-ish: store sorted for empirical pdf via histogram
        self.lo, self.hi = self.logd.min() - 0.5, self.logd.max() + 0.5
        self.hist, self.edges = np.histogram(self.logd, bins=40,
                                             range=(self.lo, self.hi), density=True)
        self.window_s = window_s

    def _lensed_pdf(self, dt_s):
        x = np.log10(np.maximum(dt_s, 1e-3))
        idx = np.clip(np.searchsorted(self.edges, x) - 1, 0, len(self.hist) - 1)
        p = self.hist[idx]
        # convert density in log10 to density in linear dt: p(dt)=p(log10 dt)/(dt ln10)
        return np.where((x >= self.lo) & (x <= self.hi),
                        p / (np.maximum(dt_s, 1e-3) * np.log(10)), 1e-30)

    def _bg_pdf(self, dt_s):
        # background: |t_i - t_j| for two uniform draws on [0,T] has triangular pdf
        # f(dt) = 2 (T - dt) / T^2, 0<=dt<=T
        T = self.window_s
        return np.where((dt_s >= 0) & (dt_s <= T), 2 * (T - dt_s) / T**2, 1e-30)

    def lr_score(self, dt_s):
        """log10 likelihood ratio; higher => more lensing-like delay."""
        return np.log10(self._lensed_pdf(dt_s) + 1e-30) - \
               np.log10(self._bg_pdf(dt_s) + 1e-30)


# ----------------------------------------------------------------------
# pairwise feature construction
# ----------------------------------------------------------------------
def build_pair_scores(df, time_prior, include_mass=False):
    """Return dict of full NxN score matrices (higher = more compatible).
    Channels: time_lr, sky_step, sky_logoverlap, snr_ratio (+ optional mass)."""
    n = len(df)
    t = df.geocent_time.values
    ra = df.ra.values; dec = df.dec.values
    sig = a90_to_sigma_rad(df.sky_area_90_deg2.values)  # rad
    snr = df.network_snr.values
    mc = df.chirp_mass.values

    # pairwise dt
    dt = np.abs(t[:, None] - t[None, :])  # NxN seconds
    time_lr = time_prior.lr_score(dt)

    # sky separation
    # vectorised great-circle
    sd = (np.sin(dec)[:, None] * np.sin(dec)[None, :] +
          np.cos(dec)[:, None] * np.cos(dec)[None, :] * np.cos(ra[:, None] - ra[None, :]))
    sep = np.arccos(np.clip(sd, -1, 1))  # rad
    sig2 = sig[:, None]**2 + sig[None, :]**2
    norm_sep = sep / np.sqrt(sig2)
    # step weight: favours small normalised separation
    sky_step = np.where(norm_sep < 1.0, 1.0,
                np.where(norm_sep < 2.0, 0.5,
                np.where(norm_sep < 3.0, 0.2, 0.0)))
    # gaussian log overlap of two 2D gaussians ~ -sep^2/(2 sig2)
    sky_logoverlap = -0.5 * sep**2 / sig2

    # snr ratio consistency (lensing preserves intrinsic; magnification differs)
    snr_ratio = -np.abs(np.log(snr[:, None] / snr[None, :]))

    out = dict(time_lr=time_lr, sky_step=sky_step,
               sky_logoverlap=sky_logoverlap, snr_ratio=snr_ratio)
    if include_mass:
        out['mass'] = -np.abs(np.log(mc[:, None] / mc[None, :]))
    # zero the diagonal influence
    for k in out:
        np.fill_diagonal(out[k], -np.inf if k == 'time_lr' else 0.0)
    return out


def _rowstd(M):
    """Row-wise standardisation, ignoring inf/nan."""
    out = np.array(M, dtype=float)
    for i in range(out.shape[0]):
        row = out[i]
        finite = np.isfinite(row)
        if finite.sum() > 1:
            mu = row[finite].mean(); sd = row[finite].std() + 1e-9
            out[i, finite] = (row[finite] - mu) / sd
            out[i, ~finite] = -10.0
        else:
            out[i, :] = 0.0
    return out


def fuse(scores, weights):
    """Weighted sum of row-standardised channels. weights: dict name->lambda."""
    total = None
    for k, w in weights.items():
        if w == 0 or k not in scores:
            continue
        z = _rowstd(scores[k])
        total = z * w if total is None else total + z * w
    if total is None:
        total = np.zeros_like(next(iter(scores.values())))
    np.fill_diagonal(total, -np.inf)
    return total


# ----------------------------------------------------------------------
# retrieval recall
# ----------------------------------------------------------------------
def true_partner_map(df):
    """source_id -> list of event row-indices (lensed images only)."""
    m = {}
    for i, (sid, kind) in enumerate(zip(df.source_id.values, df.kind.values)):
        if kind in ('SIS', 'PM'):
            m.setdefault(sid, []).append(i)
    return {s: v for s, v in m.items() if len(v) == 2}


def recall_at_k(fused, df, ks=(1, 5, 10), family=None):
    """For each lensed image, rank all others; check if true partner in top-k."""
    pmap = true_partner_map(df)
    kinds = df.kind.values
    queries = []
    for sid, (a, b) in pmap.items():
        if family and kinds[a] != family:
            continue
        queries.append((a, b)); queries.append((b, a))
    if not queries:
        return {f'R@{k}': np.nan for k in ks}
    hits = {k: 0 for k in ks}
    for q, partner in queries:
        order = np.argsort(-fused[q])  # descending
        # rank of partner
        rank = np.where(order == partner)[0]
        if len(rank) == 0:
            continue
        r = rank[0] + 1
        for k in ks:
            if r <= k:
                hits[k] += 1
    return {f'R@{k}': hits[k] / len(queries) for k in ks}


# ----------------------------------------------------------------------
# rarity / false-association analysis
# ----------------------------------------------------------------------
def false_association_analysis(fused, df, thresholds):
    """For each score threshold, count predicted pairs and how many are true.
    Returns DataFrame: threshold, n_pred_pairs, n_true_pairs_recovered,
    n_false_pairs, precision, recall, n_true_total."""
    pmap = true_partner_map(df)
    truth = set()
    for sid, (a, b) in pmap.items():
        truth.add((min(a, b), max(a, b)))
    n_true = len(truth)
    n = fused.shape[0]
    # upper triangle scores
    iu = np.triu_indices(n, k=1)
    s = fused[iu]
    rows = []
    for th in thresholds:
        mask = s >= th
        pred = set(zip(iu[0][mask], iu[1][mask]))
        tp = len(pred & truth)
        fp = len(pred) - tp
        prec = tp / max(len(pred), 1)
        rec = tp / max(n_true, 1)
        rows.append(dict(threshold=th, n_pred_pairs=len(pred),
                         n_true_recovered=tp, n_false_pairs=fp,
                         precision=prec, recall=rec, n_true_total=n_true))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# graph construction + metrics (exact-match)
# ----------------------------------------------------------------------
def build_graph_components(fused, df, topk=1, tau=None):
    """Top-k edges per node retained only if score>=tau. Returns components
    as list of sets of event indices, plus the predicted-singleton set."""
    n = fused.shape[0]
    import networkx as nx
    Gr = nx.Graph()
    Gr.add_nodes_from(range(n))
    for i in range(n):
        order = np.argsort(-fused[i])
        cnt = 0
        for j in order:
            if j == i or not np.isfinite(fused[i, j]):
                continue
            if tau is not None and fused[i, j] < tau:
                break
            Gr.add_edge(i, j, weight=fused[i, j])
            cnt += 1
            if cnt >= topk:
                break
    comps = list(nx.connected_components(Gr))
    singletons = set(c.pop() for c in comps if len(c) == 1) if False else \
                 set().union(*[c for c in comps if len(c) == 1]) if any(len(c)==1 for c in comps) else set()
    return comps, Gr


def graph_metrics_exact(comps, df):
    """Compute exact-match + B-cubed + over-merge/fragmentation + singleton P/R."""
    pmap = true_partner_map(df)
    # true system membership (only lensed)
    true_sys = {}  # event_idx -> source_id
    for sid, (a, b) in pmap.items():
        true_sys[a] = sid; true_sys[b] = sid
    true_systems = {sid: {a, b} for sid, (a, b) in pmap.items()}
    n_true = len(true_systems)

    # map event -> predicted component id
    pred_of = {}
    multi_comps = []
    for ci, comp in enumerate(comps):
        for e in comp:
            pred_of[e] = ci
        if len(comp) >= 2:
            multi_comps.append(comp)

    # exact-match: predicted multi-component that equals exactly a true system
    exact_match = 0
    for comp in multi_comps:
        # is comp exactly a true system?
        lensed_in = {e for e in comp if e in true_sys}
        if len(comp) == 2 and len(lensed_in) == 2:
            sids = {true_sys[e] for e in comp}
            if len(sids) == 1 and true_systems[next(iter(sids))] == set(comp):
                exact_match += 1
    exact_precision = exact_match / max(len(multi_comps), 1)
    exact_recall = exact_match / max(n_true, 1)

    # over-merging: fraction of multi-components containing >1 true system or contaminants
    overmerged = 0
    for comp in multi_comps:
        sids = {true_sys[e] for e in comp if e in true_sys}
        contaminated = any(e not in true_sys for e in comp)
        if len(sids) > 1 or contaminated:
            overmerged += 1
    overmerge_rate = overmerged / max(len(multi_comps), 1)

    # fragmentation: true systems split across >1 predicted component
    fragmented = 0
    for sid, members in true_systems.items():
        pcs = {pred_of.get(e, -1) for e in members}
        if len(pcs) > 1:
            fragmented += 1
    fragmentation_rate = fragmented / max(n_true, 1)

    # B-cubed (over lensed events only) -- vectorised via cluster-size maps
    from collections import Counter, defaultdict
    lensed_events = list(true_sys.keys())
    # predicted cluster sizes (restricted to lensed events) and members
    pred_members = defaultdict(set)
    for e in lensed_events:
        pred_members[pred_of.get(e, -1)].add(e)
    bc_p = []; bc_r = []
    for e in lensed_events:
        pc = pred_of.get(e, -1)
        pred_cluster = pred_members[pc]
        true_cluster = true_systems[true_sys[e]]
        inter = len(pred_cluster & true_cluster)
        bc_p.append(inter / max(len(pred_cluster), 1))
        bc_r.append(inter / max(len(true_cluster), 1))
    bcubed_p = float(np.mean(bc_p)) if bc_p else 0.0
    bcubed_r = float(np.mean(bc_r)) if bc_r else 0.0

    # singleton P/R (unlensed should be singletons)
    unlensed_idx = set(np.where(df.kind.values == 'unlensed')[0])
    pred_singletons = set(e for comp in comps if len(comp) == 1 for e in comp)
    if unlensed_idx:
        sing_recall = len(pred_singletons & unlensed_idx) / len(unlensed_idx)
    else:
        sing_recall = np.nan
    sing_precision = (len(pred_singletons & unlensed_idx) / max(len(pred_singletons), 1)
                      if pred_singletons else np.nan)

    sizes = [len(c) for c in comps]
    return dict(n_pred_components=len(comps),
                n_multi=len(multi_comps),
                exact_precision=exact_precision,
                exact_recall=exact_recall,
                overmerge_rate=overmerge_rate,
                fragmentation_rate=fragmentation_rate,
                bcubed_precision=bcubed_p, bcubed_recall=bcubed_r,
                singleton_recall=sing_recall, singleton_precision=sing_precision,
                mean_component_size=float(np.mean(sizes)),
                max_component_size=int(np.max(sizes)))


def maximum_weight_matching(fused, df, tau=None):
    """Doublet-optimal: non-bipartite max-weight matching with abstention.
    Returns list of matched pairs (sets) + singletons."""
    import networkx as nx
    n = fused.shape[0]
    Gr = nx.Graph()
    Gr.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    s = fused[iu]
    for a, b, w in zip(iu[0], iu[1], s):
        if not np.isfinite(w):
            continue
        if tau is not None and w < tau:
            continue
        Gr.add_edge(int(a), int(b), weight=float(w))
    matching = nx.max_weight_matching(Gr, maxcardinality=False)
    comps = [set(e) for e in matching]
    matched = set().union(*comps) if comps else set()
    for i in range(n):
        if i not in matched:
            comps.append({i})
    return comps


# ----------------------------------------------------------------------
# parameter-summary kNN baseline
# ----------------------------------------------------------------------
def param_knn_scores(df):
    """kNN retrieval directly on parameter summaries (Mc, q, chi_eff~0, dL, sky).
    Returns an NxN similarity (negative standardised Euclidean distance)."""
    feats = np.column_stack([
        df.chirp_mass.values,
        df.mass_ratio.values,
        df.luminosity_distance.values,
        df.ra.values, np.sin(df.dec.values),
    ])
    # standardise
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    D = cdist(feats, feats, 'euclidean')
    np.fill_diagonal(D, np.inf)
    return -D
