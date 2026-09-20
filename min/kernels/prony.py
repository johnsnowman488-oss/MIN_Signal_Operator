"""Prony identification of finite sums of decaying exponentials."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PronySOE:
    """Finite exponential representation k(t) ~= sum c_j exp(-gamma_j t)."""

    weights: np.ndarray
    gammas: np.ndarray
    sample_spacing: float

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.sum(
            self.weights[:, None] * np.exp(-self.gammas[:, None] * t[None, :]),
            axis=0,
        )


def fit_prony(samples: np.ndarray, dt: float, order: int) -> PronySOE:
    """Fit a finite exponential sum to uniformly sampled kernel values.

    The reference implementation uses the classical annihilating-polynomial
    construction followed by a linear least-squares amplitude fit.

    For samples k_n = sum_j c_j z_j^n, the roots z_j determine the decay
    rates gamma_j = -log(z_j)/dt. The method is intended for clean or mildly
    perturbed data; order selection and noise-robust variants are separate
    concerns.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if order < 1:
        raise ValueError("order must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if samples.size < 2 * order + 1:
        raise ValueError("at least 2*order+1 samples are required")

    y = samples.astype(np.complex128, copy=False)
    # Sum_{j=0}^p a_j y_{n+j}=0 with a_p=1.
    A = np.vstack([y[n:n + order] for n in range(order)])
    b = -y[order:2 * order]
    coeff = np.linalg.solve(A, b)
    poly = np.r_[coeff, 1.0]
    z = np.roots(poly)

    # Discrete exponentials with |z| close to one correspond to slow memory.
    gammas = -np.log(z) / dt

    n = np.arange(y.size)
    vandermonde = z[None, :] ** n[:, None]
    weights, *_ = np.linalg.lstsq(vandermonde, y, rcond=None)

    # For a real, non-oscillatory kernel, numerical noise may leave tiny
    # imaginary parts. Preserve genuinely complex estimates.
    if np.max(np.abs(weights.imag)) < 1e-10 * max(1.0, np.max(np.abs(weights.real))):
        weights = weights.real
    if np.max(np.abs(gammas.imag)) < 1e-10 * max(1.0, np.max(np.abs(gammas.real))):
        gammas = gammas.real

    return PronySOE(weights=weights, gammas=gammas, sample_spacing=dt)
