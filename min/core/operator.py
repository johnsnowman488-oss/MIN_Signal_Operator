"""Reference numerical implementation of a causal MIN operator."""

from __future__ import annotations

import numpy as np


class MemoryOperator:
    """Discrete causal realization of (Mf)(t)=int_0^t K(t-s)f(s)ds.

    This reference implementation uses the trapezoidal rule. It is intended
    for correctness experiments, not high-performance production use.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        f = np.asarray(f)
        if t.ndim != 1 or f.ndim != 1 or t.size != f.size:
            raise ValueError("t and f must be one-dimensional arrays of equal length")
        if t.size < 2:
            raise ValueError("at least two time samples are required")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")

        # Probe the kernel once so complex-valued kernel models are preserved
        # even when the input signal itself is real.
        kernel_probe = np.asarray(self.kernel(np.array([0.0])))
        y = np.zeros_like(f, dtype=np.result_type(f, kernel_probe, float))
        for i in range(t.size):
            lag = t[i] - t[: i + 1]
            values = self.kernel(lag) * f[: i + 1]
            y[i] = np.trapezoid(values, t[: i + 1])
        return y
