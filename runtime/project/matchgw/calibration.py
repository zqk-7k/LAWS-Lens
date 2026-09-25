from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class LogisticCalibrator:
    """Small numpy logistic reranker/calibrator for candidate-pair features."""

    l2: float = 1e-3
    lr: float = 0.05
    max_iter: int = 600
    pos_weight: float | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticCalibrator":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) == 0:
            self.mean_ = np.zeros(x.shape[1], dtype=np.float64)
            self.scale_ = np.ones(x.shape[1], dtype=np.float64)
            self.coef_ = np.zeros(x.shape[1] + 1, dtype=np.float64)
            return self

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.scale_
        xb = np.concatenate([np.ones((len(xs), 1), dtype=np.float64), xs], axis=1)

        pos = float(y.sum())
        neg = float(len(y) - pos)
        pw = self.pos_weight if self.pos_weight is not None else (neg / max(pos, 1.0))
        sample_w = np.where(y > 0.5, pw, 1.0)
        sample_w = sample_w / max(sample_w.mean(), 1e-8)

        w = np.zeros(xb.shape[1], dtype=np.float64)
        for _ in range(self.max_iter):
            p = _sigmoid(xb @ w)
            grad = (xb.T @ ((p - y) * sample_w)) / len(y)
            grad[1:] += self.l2 * w[1:]
            w -= self.lr * grad
        self.coef_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("calibrator is not fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xs = (x - self.mean_) / self.scale_
        xb = np.concatenate([np.ones((len(xs), 1), dtype=np.float64), xs], axis=1)
        return _sigmoid(xb @ self.coef_).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "type": "logistic_pair_calibrator",
            "l2": self.l2,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "pos_weight": self.pos_weight,
            "mean": [] if self.mean_ is None else self.mean_.tolist(),
            "scale": [] if self.scale_ is None else self.scale_.tolist(),
            "coef": [] if self.coef_ is None else self.coef_.tolist(),
        }


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(probs) == 0:
        return {"brier": 0.0, "nll": 0.0, "ece": 0.0}
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    brier = float(np.mean((probs - labels) ** 2))
    nll = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return {"brier": brier, "nll": nll, "ece": float(ece)}
