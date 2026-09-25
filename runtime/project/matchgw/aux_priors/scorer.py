from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np


MetricFn = Callable[[np.ndarray], dict[str, dict]]


def _metric_key(metrics: dict[str, dict]) -> tuple[float, float, float, float]:
    overall = metrics["overall"]
    return (
        float(overall["r@10"]),
        float(overall["r@5"]),
        float(overall["r@1"]),
        float(overall.get("top_1pct", 0.0)),
    )


def select_best_weighted_lambdas(
    components: dict[str, np.ndarray],
    prior_keys: list[str],
    grid: list[float],
    evaluate: MetricFn,
    base_key: str = "waveform",
    max_joint_keys: int = 3,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Select weighted-sum correction weights on validation data.

    A small exact grid is used for up to ``max_joint_keys`` priors. Larger feature
    sets fall back to coordinate search so full-catalog validation remains practical.
    """
    best_lams = {key: 0.0 for key in prior_keys}
    best_metrics = evaluate(components[base_key])
    best_key = _metric_key(best_metrics)

    if len(prior_keys) <= max_joint_keys:
        for values in itertools.product(grid, repeat=len(prior_keys)):
            score = components[base_key].copy()
            for key, value in zip(prior_keys, values):
                if value != 0.0:
                    score = score + value * components[key]
            np.fill_diagonal(score, -np.inf)
            metrics = evaluate(score)
            key_tuple = _metric_key(metrics)
            if key_tuple > best_key:
                best_key = key_tuple
                best_lams = dict(zip(prior_keys, values))
                best_metrics = metrics
        return best_lams, best_metrics

    for _ in range(3):
        improved = False
        for prior_key in prior_keys:
            current_lam = best_lams[prior_key]
            current_metrics = best_metrics
            current_key = best_key
            for lam in grid:
                trial = dict(best_lams)
                trial[prior_key] = lam
                score = components[base_key].copy()
                for key, value in trial.items():
                    if value != 0.0:
                        score = score + value * components[key]
                np.fill_diagonal(score, -np.inf)
                metrics = evaluate(score)
                key_tuple = _metric_key(metrics)
                if key_tuple > current_key:
                    current_key = key_tuple
                    current_lam = lam
                    current_metrics = metrics
            if current_lam != best_lams[prior_key]:
                improved = True
            best_lams[prior_key] = current_lam
            best_metrics = current_metrics
            best_key = current_key
        if not improved:
            break
    return best_lams, best_metrics
