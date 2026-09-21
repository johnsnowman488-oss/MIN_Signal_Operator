"""Structure-aware nonnegative SOE fitting."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NNLSSOE:
    """Finite SOE k(t) ~= sum_j w_j exp(-gamma_j t), w_j >= 0."""

    weights: np.ndarray
    gammas: np.ndarray
    residual_norm: float

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.sum(
            self.weights[:, None] * np.exp(-self.gammas[:, None] * t[None, :]),
            axis=0,
        )

    @property
    def active_modes(self) -> int:
        scale = max(1.0, float(np.max(self.weights)))
        return int(np.count_nonzero(self.weights > 1e-10 * scale))


def log_rate_grid(gamma_min: float, gamma_max: float, order: int) -> np.ndarray:
    """Return a positive logarithmically spaced decay-rate dictionary."""
    if gamma_min <= 0 or gamma_max <= 0:
        raise ValueError("gamma_min and gamma_max must be positive")
    if gamma_max < gamma_min:
        raise ValueError("gamma_max must be >= gamma_min")
    if order < 1:
        raise ValueError("order must be positive")
    return np.geomspace(gamma_min, gamma_max, order)


def fit_nnls(samples: np.ndarray, t: np.ndarray, gammas: np.ndarray) -> NNLSSOE:
    """Fit nonnegative SOE weights for a supplied positive rate dictionary.

    Decay rates are fixed. Only the nonnegative weights are identified.
    """
    samples = np.asarray(samples, dtype=float)
    t = np.asarray(t, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if samples.ndim != 1 or t.ndim != 1 or gammas.ndim != 1:
        raise ValueError("samples, t, and gammas must be one-dimensional")
    if samples.size != t.size:
        raise ValueError("samples and t must have equal length")
    if samples.size < 2:
        raise ValueError("at least two samples are required")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if gammas.size == 0 or np.any(gammas <= 0):
        raise ValueError("gammas must be nonempty and strictly positive")

    try:
        from scipy.optimize import nnls
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fit_nnls requires scipy>=1.10") from exc

    A = np.exp(-np.outer(t, gammas))
    weights, residual_norm = nnls(A, samples)
    return NNLSSOE(weights=weights, gammas=gammas, residual_norm=float(residual_norm))
