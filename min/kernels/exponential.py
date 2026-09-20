"""Exponential memory kernels."""

from __future__ import annotations

import numpy as np


def exponential_kernel(t: np.ndarray, tau: float, weight: float = 1.0) -> np.ndarray:
    """Return a causal exponential kernel k(t)=weight*exp(-t/tau).

    Values for t < 0 are zero. tau must be positive.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    t = np.asarray(t, dtype=float)
    return weight * np.exp(-np.maximum(t, 0.0) / tau) * (t >= 0)
