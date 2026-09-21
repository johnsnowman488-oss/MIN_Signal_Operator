"""Numerical metrics for time-domain signal comparisons."""

from __future__ import annotations

import numpy as np


def _pair(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("signals must be 1D and have equal length")
    return x, y


def energy(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sum(np.abs(x) ** 2))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def peak_magnitude(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.max(np.abs(x)))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _pair(x, y)
    return float(np.mean(np.abs(x - y) ** 2))


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    denom = energy(x)
    if denom <= 0:
        raise ValueError("reference signal must have nonzero energy")
    return mse(x, y) * x.size / denom


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _pair(x, y)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("signals must have nonzero norm")
    return float(abs(np.vdot(x, y)) / (nx * ny))


def delay_of_max_correlation(x: np.ndarray, y: np.ndarray) -> int:
    """Return lag in samples maximizing magnitude cross-correlation.

    Positive lag means y is delayed relative to x under the convention used
    by np.correlate(full): y[n] aligns with x[n-lag].
    """
    x, y = _pair(x, y)
    corr = np.correlate(y, x, mode="full")
    return int(np.argmax(np.abs(corr)) - (x.size - 1))
