"""Frequency-response estimators for causal MIN operators."""

from __future__ import annotations

import numpy as np


def complex_tone(t: np.ndarray, omega: float, amplitude: complex = 1.0) -> np.ndarray:
    """Return x(t)=amplitude*exp(i*omega*t)."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if omega < 0:
        raise ValueError("omega must be nonnegative")
    return amplitude * np.exp(1j * omega * t)


def estimate_complex_gain(
    x: np.ndarray,
    y: np.ndarray,
    start: int | None = None,
    tail_fraction: float = 0.25,
) -> complex:
    """Estimate y ~= H*x over a tail segment using least squares.

    The estimator is intentionally a projection rather than a single-sample
    ratio, making it robust to the finite causal startup transient.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be equal-length 1D arrays with >=2 samples")
    if start is None:
        if not (0.0 < tail_fraction < 1.0):
            raise ValueError("tail_fraction must lie in (0,1)")
        start = int(np.floor((1.0 - tail_fraction) * x.size))
    if start < 0 or start >= x.size - 1:
        raise ValueError("start must leave at least two samples")
    xt = x[start:]
    yt = y[start:]
    denom = np.vdot(xt, xt)
    if abs(denom) == 0:
        raise ValueError("input tail has zero energy")
    return np.vdot(xt, yt) / denom


def exponential_frequency_response(omega, tau: float, weight: float = 1.0):
    """Continuous-time H(i*omega)=weight/(1/tau+i*omega)."""
    omega = np.asarray(omega, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")
    return weight / (1.0 / tau + 1j * omega)


def soe_frequency_response(omega, weights, gammas):
    """Continuous-time frequency response of sum_j w_j exp(-gamma_j t)."""
    omega = np.asarray(omega, dtype=float)
    weights = np.asarray(weights, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if weights.ndim != 1 or gammas.ndim != 1 or weights.size != gammas.size:
        raise ValueError("weights and gammas must be equal-length 1D arrays")
    if weights.size == 0 or np.any(gammas <= 0):
        raise ValueError("need nonempty positive decay rates")
    return np.sum(weights[None, :] / (gammas[None, :] + 1j * omega[:, None]), axis=1)


def finite_horizon_kernel_response(t: np.ndarray, kernel, omega):
    """Numerical H_T(omega)=int_0^T K(u) exp(-i omega u) du."""
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be a strictly increasing 1D axis")
    if omega.ndim != 1 or np.any(omega < 0):
        raise ValueError("omega must be a 1D nonnegative array")
    k = np.asarray(kernel(t))
    integrand = k[:, None] * np.exp(-1j * t[:, None] * omega[None, :])
    return np.trapezoid(integrand, t, axis=0)


def group_delay(omega: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Return -d phase/d omega using an unwrapped phase."""
    omega = np.asarray(omega, dtype=float)
    response = np.asarray(response)
    if omega.ndim != 1 or response.ndim != 1 or omega.size != response.size or omega.size < 3:
        raise ValueError("omega and response must be equal-length 1D arrays with >=3 samples")
    phase = np.unwrap(np.angle(response))
    return -np.gradient(phase, omega)
