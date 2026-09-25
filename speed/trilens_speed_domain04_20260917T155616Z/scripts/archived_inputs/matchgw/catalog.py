from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatalogSystem:
    """A connected component interpreted as one catalog-level lens system candidate."""

    system_id: int
    members: tuple[int, ...]
    edge_count: int
    score_max: float
    score_mean: float
    p_hat_max: float
    p_hat_mean: float
    dominant_pair_id: int
    dominant_count: int
    true_pair_ids: tuple[int, ...]
    unlensed_count: int
    is_true_system: bool
    is_pure_system: bool


def _true_systems(meta: list[dict]) -> dict[int, set[int]]:
    systems: dict[int, set[int]] = defaultdict(set)
    for idx, row in enumerate(meta):
        pair_id = int(row.get("pair_id", -1))
        if pair_id >= 0:
            systems[pair_id].add(int(idx))
    # 至少两张像才构成 catalog-level 强透镜系统。
    return {k: v for k, v in systems.items() if len(v) >= 2}


def _components(n_events: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_events)]
    for i, j in edges:
        if i == j:
            continue
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))

    seen = np.zeros(n_events, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n_events):
        if seen[start] or not adj[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    return comps


def _component_rows(df: pd.DataFrame, members: set[int]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["i"].isin(members) & df["j"].isin(members)]


def catalog_system_report(
    candidates: pd.DataFrame,
    meta: list[dict],
    threshold: float,
    threshold_name: str = "tier12",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate catalog-level system discovery from calibrated pair candidates.

    Pair-level evaluation asks whether an individual edge is correct. This function
    promotes high-confidence edges into a graph, treats connected components as
    candidate lens systems, and evaluates whether true lens systems are recovered
    at the whole-catalog level.
    """
    n_events = len(meta)
    true_system_map = _true_systems(meta)
    total_true_systems = len(true_system_map)

    if candidates.empty:
        empty_cols = [
            "system_id", "members", "size", "edge_count", "score_max", "score_mean",
            "p_hat_max", "p_hat_mean", "dominant_pair_id", "dominant_count",
            "true_pair_ids", "unlensed_count", "is_true_system", "is_pure_system",
        ]
        metrics = _catalog_metrics([], true_system_map, n_events, threshold, threshold_name)
        metrics["catalog_recovered_true_systems"] = 0
        return pd.DataFrame(columns=empty_cols), finalize_catalog_metrics(metrics)

    use = candidates[candidates["p_hat"].astype(float) >= float(threshold)].copy()
    edges = [(int(r.i), int(r.j)) for r in use.itertuples(index=False)]
    comps = _components(n_events, edges)

    systems: list[CatalogSystem] = []
    recovered_true: set[int] = set()
    for system_id, comp in enumerate(comps):
        members = set(comp)
        sub = _component_rows(use, members)
        pair_counts: dict[int, int] = defaultdict(int)
        unlensed_count = 0
        for idx in comp:
            pair_id = int(meta[idx].get("pair_id", -1))
            if pair_id >= 0:
                pair_counts[pair_id] += 1
            else:
                unlensed_count += 1
        true_pair_ids = tuple(sorted(pair_counts))
        complete_true = [pid for pid in true_pair_ids if true_system_map.get(pid, set()).issubset(members)]
        recovered_true.update(complete_true)
        dominant_pair_id = max(pair_counts, key=pair_counts.get) if pair_counts else -1
        dominant_count = int(pair_counts.get(dominant_pair_id, 0)) if dominant_pair_id >= 0 else 0
        is_true_system = bool(complete_true)
        is_pure_system = (
            len(complete_true) == 1
            and len(true_pair_ids) == 1
            and unlensed_count == 0
            and members == true_system_map.get(complete_true[0], set())
        )
        systems.append(CatalogSystem(
            system_id=system_id,
            members=tuple(comp),
            edge_count=int(len(sub)),
            score_max=float(sub["score"].max()) if not sub.empty else 0.0,
            score_mean=float(sub["score"].mean()) if not sub.empty else 0.0,
            p_hat_max=float(sub["p_hat"].max()) if not sub.empty else 0.0,
            p_hat_mean=float(sub["p_hat"].mean()) if not sub.empty else 0.0,
            dominant_pair_id=int(dominant_pair_id),
            dominant_count=dominant_count,
            true_pair_ids=true_pair_ids,
            unlensed_count=int(unlensed_count),
            is_true_system=is_true_system,
            is_pure_system=is_pure_system,
        ))

    rows = []
    for s in systems:
        rows.append({
            "system_id": s.system_id,
            "members": " ".join(map(str, s.members)),
            "size": len(s.members),
            "edge_count": s.edge_count,
            "score_max": s.score_max,
            "score_mean": s.score_mean,
            "p_hat_max": s.p_hat_max,
            "p_hat_mean": s.p_hat_mean,
            "dominant_pair_id": s.dominant_pair_id,
            "dominant_count": s.dominant_count,
            "true_pair_ids": " ".join(map(str, s.true_pair_ids)),
            "unlensed_count": s.unlensed_count,
            "is_true_system": int(s.is_true_system),
            "is_pure_system": int(s.is_pure_system),
        })
    system_df = pd.DataFrame(rows)
    metrics = _catalog_metrics(systems, true_system_map, n_events, threshold, threshold_name)
    metrics["catalog_recovered_true_systems"] = int(len(recovered_true))
    metrics["catalog_system_recall"] = float(len(recovered_true) / max(total_true_systems, 1))
    metrics["catalog_completeness"] = metrics["catalog_system_recall"]
    return system_df, finalize_catalog_metrics(metrics)


def _catalog_metrics(
    systems: list[CatalogSystem],
    true_system_map: dict[int, set[int]],
    n_events: int,
    threshold: float,
    threshold_name: str,
) -> dict[str, float]:
    total_true_systems = len(true_system_map)
    predicted = len(systems)
    true_pred = sum(1 for s in systems if s.is_true_system)
    pure_pred = sum(1 for s in systems if s.is_pure_system)
    false_alarm = predicted - true_pred
    merged_or_contaminated = sum(
        1 for s in systems
        if s.is_true_system and (not s.is_pure_system)
    )
    precision = true_pred / max(predicted, 1)
    purity = pure_pred / max(predicted, 1)
    # Recall is filled by catalog_system_report after complete true systems are counted.
    recall_placeholder = 0.0
    f1_placeholder = 0.0
    sizes = [len(s.members) for s in systems]
    return {
        "catalog_threshold": float(threshold),
        "catalog_threshold_name": threshold_name,
        "catalog_true_systems": int(total_true_systems),
        "catalog_predicted_systems": int(predicted),
        "catalog_true_predicted_systems": int(true_pred),
        "catalog_pure_predicted_systems": int(pure_pred),
        "catalog_false_alarm_systems": int(false_alarm),
        "catalog_merged_or_contaminated_systems": int(merged_or_contaminated),
        "catalog_impure_systems": int(predicted - pure_pred),
        "catalog_system_precision": float(precision),
        "catalog_purity": float(purity),
        "catalog_system_recall": float(recall_placeholder),
        "catalog_system_f1": float(f1_placeholder),
        "catalog_completeness": float(recall_placeholder),
        "catalog_false_alarm_rate_per_event": float(false_alarm / max(n_events, 1)),
        "catalog_mean_system_size": float(np.mean(sizes)) if sizes else 0.0,
        "catalog_max_system_size": int(max(sizes)) if sizes else 0,
    }


def finalize_catalog_metrics(metrics: dict[str, float]) -> dict[str, float]:
    precision = float(metrics.get("catalog_system_precision", 0.0))
    recall = float(metrics.get("catalog_system_recall", 0.0))
    metrics["catalog_system_f1"] = float(2 * precision * recall / max(precision + recall, 1e-12))
    metrics["catalog_completeness"] = recall
    return metrics
