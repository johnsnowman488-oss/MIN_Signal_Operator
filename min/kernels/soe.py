"""State-space realization of finite sum-of-exponentials memory kernels."""

from __future__ import annotations

import numpy as np


class SOEMemory:
    """Streaming realization of Mx(t)=int_0^t k(t-s)x(s)ds.

    For k(t)=sum_j c_j exp(-gamma_j t), each state obeys
        q'_j = -gamma_j q_j + x(t)
    and the output is sum_j c_j q_j.
    """

    def __init__(self, weights, gammas):
        self.weights = np.asarray(weights)
        self.gammas = np.asarray(gammas)
        if self.weights.ndim != 1 or self.gammas.ndim != 1:
            raise ValueError("weights and gammas must be one-dimensional")
        if self.weights.size != self.gammas.size or self.weights.size == 0:
            raise ValueError("weights and gammas must have equal nonzero length")

    def apply(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x)
        if t.ndim != 1 or x.ndim != 1 or t.size != x.size:
            raise ValueError("t and x must be one-dimensional arrays of equal length")
        if t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing with at least two samples")

        q = np.zeros(self.weights.size, dtype=np.result_type(x, self.weights, float))
        y = np.zeros_like(x, dtype=np.result_type(x, self.weights, float))

        for n in range(1, t.size):
            dt = t[n] - t[n - 1]
            # Exact update for a piecewise-linear input over the interval.
            decay = np.exp(-self.gammas * dt)
            slope = (x[n] - x[n - 1]) / dt
            q = (
                decay * q
                + x[n - 1] * (1.0 - decay) / self.gammas
                + slope * (dt / self.gammas - (1.0 - decay) / self.gammas**2)
            )
            y[n] = np.sum(self.weights * q)

        return y
