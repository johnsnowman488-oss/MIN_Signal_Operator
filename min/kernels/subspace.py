"""Hankel-SVD subspace identification of exponential sums."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SubspaceSOE:
    """Finite exponential representation recovered from sampled data."""

    weights: np.ndarray
    gammas: np.ndarray
    sample_spacing: float
    singular_values: np.ndarray
    method: str

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.sum(
            self.weights[:, None] * np.exp(-self.gammas[:, None] * t[None, :]),
            axis=0,
        )


def _validate(samples: np.ndarray, dt: float, order: int) -> np.ndarray:
    y = np.asarray(samples)
    if y.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if order < 1:
        raise ValueError("order must be positive")
    if y.size < 2 * order + 3:
        raise ValueError("not enough samples for requested subspace order")
    return y.astype(np.complex128, copy=False)


def _hankel_pair(y: np.ndarray, rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = y.size
    m = rows or (n // 2)
    if m < 3 or m >= n - 1:
        raise ValueError("pencil rows must leave at least two columns")
    cols = n - m
    H0 = np.column_stack([y[i:i + m] for i in range(cols)])
    H1 = np.column_stack([y[i + 1:i + m + 1] for i in range(cols)])
    return H0, H1


def _finish(y, z, dt, singular_values, method):
    gammas = -np.log(z) / dt
    n = np.arange(y.size)
    V = z[None, :] ** n[:, None]
    weights, *_ = np.linalg.lstsq(V, y, rcond=None)
    return SubspaceSOE(weights, gammas, dt, singular_values, method)


def fit_matrix_pencil(samples: np.ndarray, dt: float, order: int, rows: int | None = None) -> SubspaceSOE:
    """Recover exponential modes with a truncated Hankel matrix pencil."""
    y = _validate(samples, dt, order)
    H0, H1 = _hankel_pair(y, rows)
    U, s, Vh = np.linalg.svd(H0, full_matrices=False)
    if order > s.size:
        raise ValueError("order exceeds available Hankel subspace dimension")
    Ur, Vr = U[:, :order], Vh.conj().T[:, :order]
    A = Ur.conj().T @ H1 @ Vr @ np.diag(1.0 / s[:order])
    z = np.linalg.eigvals(A)
    if not np.all(np.isfinite(z)) or np.any(np.abs(z) == 0):
        raise FloatingPointError("matrix-pencil produced invalid exponential nodes")
    return _finish(y, z, dt, s, "matrix-pencil")


def fit_esprit(samples: np.ndarray, dt: float, order: int, rows: int | None = None) -> SubspaceSOE:
    """Recover exponential modes with ESPRIT on a truncated Hankel SVD."""
    y = _validate(samples, dt, order)
    H0, _ = _hankel_pair(y, rows)
    U, s, _ = np.linalg.svd(H0, full_matrices=False)
    if order > s.size:
        raise ValueError("order exceeds available Hankel subspace dimension")
    Us = U[:, :order]
    shift = np.linalg.lstsq(Us[:-1, :], Us[1:, :], rcond=None)[0]
    z = np.linalg.eigvals(shift)
    if not np.all(np.isfinite(z)) or np.any(np.abs(z) == 0):
        raise FloatingPointError("ESPRIT produced invalid exponential nodes")
    return _finish(y, z, dt, s, "ESPRIT")
