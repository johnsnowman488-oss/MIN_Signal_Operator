"""Completely monotone power-law memory kernels."""

from __future__ import annotations

import numpy as np


def power_law_kernel(
    t: np.ndarray,
    alpha: float,
    tau: float = 1.0,
    weight: float = 1.0,
) -> np.ndarray:
    """Return k(t)=weight*(1+t/tau)^(-alpha) for t >= 0.

    For alpha > 0 and tau > 0 this is a completely monotone kernel.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    t = np.asarray(t, dtype=float)
    return weight * (1.0 + np.maximum(t, 0.0) / tau) ** (-alpha) * (t >= 0)
