"""Minimal scalar vector fitting for Laplace-domain kernel models."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VectorFitResult:
    """Pole-residue approximation F(s) ~= sum r_j/(s-p_j)."""

    poles: np.ndarray
    residues: np.ndarray
    iterations: int
    residual_history: tuple[float, ...]

    @property
    def gammas(self) -> np.ndarray:
        return -self.poles

    @property
    def weights(self) -> np.ndarray:
        return self.residues

    def evaluate_laplace(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.complex128)
        return np.sum(self.residues[:, None] / (s[None, :] - self.poles[:, None]), axis=0)

    def evaluate_kernel(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.sum(self.residues[:, None] * np.exp(self.poles[:, None] * t[None, :]), axis=0)


def fit_vector_fitting(
    s: np.ndarray,
    response: np.ndarray,
    order: int,
    initial_gammas: np.ndarray | None = None,
    iterations: int = 8,
) -> VectorFitResult:
    """Fit a zero-direct-term rational Laplace response by pole relocation.

    This compact reference implementation targets smooth memory kernels:
    only residue terms are used, poles are relocated with the sigma-function
    step, and unstable poles are flipped into the left half-plane.
    """
    s = np.asarray(s, dtype=np.complex128)
    f = np.asarray(response, dtype=np.complex128)
    if s.ndim != 1 or f.ndim != 1 or s.size != f.size:
        raise ValueError("s and response must be one-dimensional arrays of equal length")
    if order < 1 or iterations < 1:
        raise ValueError("order and iterations must be positive")
    if s.size < 2 * order + 2:
        raise ValueError("not enough frequency samples for requested order")

    if initial_gammas is None:
        positive_s = np.real(s[np.real(s) > 0])
        if positive_s.size == 0:
            raise ValueError("initial rates require positive real Laplace samples")
        initial_gammas = np.geomspace(positive_s.min(), positive_s.max(), order)
    else:
        initial_gammas = np.asarray(initial_gammas, dtype=float)
        if initial_gammas.ndim != 1 or initial_gammas.size != order or np.any(initial_gammas <= 0):
            raise ValueError("initial_gammas must contain order positive rates")

    poles = -initial_gammas.astype(np.complex128)
    history: list[float] = []

    for _ in range(iterations):
        X = np.column_stack(
            [1.0 / (s - pole) for pole in poles]
            + [-f / (s - pole) for pole in poles]
        )
        solution, *_ = np.linalg.lstsq(X, f, rcond=None)
        reloc = solution[order:]
        relocation_matrix = np.diag(poles) - np.ones((order, 1)) @ reloc[None, :]
        poles = np.linalg.eigvals(relocation_matrix)
        poles = np.where(poles.real > 0, -poles, poles)
        residual = np.linalg.norm(X @ solution - f) / max(np.linalg.norm(f), np.finfo(float).eps)
        history.append(float(residual))
        if not np.all(np.isfinite(poles)):
            raise FloatingPointError("vector fitting produced invalid poles")

    residue_matrix = np.column_stack([1.0 / (s - pole) for pole in poles])
    residues, *_ = np.linalg.lstsq(residue_matrix, f, rcond=None)
    return VectorFitResult(poles, residues, iterations, tuple(history))
