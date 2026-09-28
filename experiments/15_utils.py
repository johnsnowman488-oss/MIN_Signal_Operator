"""Self-contained utilities for the MIN Experiment 15 series.

This module intentionally duplicates/adapts validated 14C numerical primitives
instead of importing 14-series experiment modules. Experiment 15 can therefore
evolve independently while retaining the same SOE state convention.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min import SOEMemory
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
ENVIRONMENTS = ("white_limit", "short", "multiscale", "powerlaw", "squared_exp")
GEOMETRIES = ("clustered", "logspread", "wide")
SEEDS = tuple(range(5))
SNR_DB = (0.0, 10.0, 20.0, 30.0)

NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE
MODES = 16
PCA_DIMENSIONS = tuple(range(1, MODES + 1))
TASK_TOLERANCE = 1.10
READOUT_RIDGE = 1e-10
ENV_NOISE_SEED_OFFSET = 100_003


def environment_envelope(t: np.ndarray, name: str) -> np.ndarray:
    if name == "white_limit":
        return np.exp(-t / 0.005)
    if name == "short":
        return np.exp(-t / 0.05)
    if name == "multiscale":
        return 0.65 * np.exp(-t / 0.03) + 0.35 * np.exp(-t / 0.7)
    if name == "powerlaw":
        return (1.0 + t / 0.2) ** (-0.7)
    if name == "squared_exp":
        return np.exp(-(t / 0.15) ** 2)
    raise ValueError(name)


def rate_pattern(geometry: str) -> np.ndarray:
    if geometry == "clustered":
        return 10.0 * np.exp(np.linspace(-0.05, 0.05, MODES))
    if geometry == "logspread":
        return np.geomspace(2.0, 30.0, MODES)
    if geometry == "wide":
        return np.geomspace(0.5, 100.0, MODES)
    raise ValueError(geometry)


def uniform_weights() -> np.ndarray:
    return np.full(MODES, 1.0 / MODES)


def state_coefficients(gammas: np.ndarray):
    decay = np.exp(-gammas * DT)
    b_curr = 1.0 / gammas - (1.0 - decay) / (DT * gammas**2)
    b_prev = (1.0 - decay) / gammas - b_curr
    return decay, b_curr, b_prev


KERNELS = {
    g: {"gammas": rate_pattern(g), "weights": uniform_weights()}
    for g in GEOMETRIES
}
for _g in GEOMETRIES:
    KERNELS[_g]["coefficients"] = state_coefficients(KERNELS[_g]["gammas"])


def exact_piecewise_linear_states(
    x: np.ndarray,
    gammas: np.ndarray,
    coefficients=None,
) -> np.ndarray:
    samples = np.asarray(x, dtype=complex)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("x must be one-dimensional with at least two samples")
    decay, b_curr, b_prev = coefficients or state_coefficients(gammas)
    out = np.empty((samples.size, gammas.size), dtype=complex)
    for j, (a, bc, bp) in enumerate(zip(decay, b_curr, b_prev)):
        forcing = np.empty(samples.size, dtype=complex)
        forcing[0] = 0.0
        forcing[1:] = bc * samples[1:] + bp * samples[:-1]
        out[:, j] = lfilter([1.0], [1.0, -a], forcing)
    return out


def validate_repository_equivalence() -> float:
    rng = np.random.default_rng(1501)
    x = rng.normal(size=257) + 1j * rng.normal(size=257)
    spec = KERNELS["wide"]
    fast = exact_piecewise_linear_states(x, spec["gammas"], spec["coefficients"])
    reference = SOEMemory(spec["weights"], spec["gammas"]).state_trajectory(
        np.arange(x.size, dtype=float) * DT, x
    )
    rel = float(np.linalg.norm(fast - reference) /
                max(np.linalg.norm(reference), np.finfo(float).tiny))
    if rel > 1e-9:
        raise RuntimeError(f"repository state equivalence failed: {rel:.3e}")
    return rel


def synth_environment_noise(environment: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * DT
    c = environment_envelope(t, environment)
    circ = np.r_[c, c[-2:0:-1]]
    spectrum = np.maximum(np.real(np.fft.rfft(circ)), 0.0)
    z = rng.normal(size=spectrum.size) + 1j * rng.normal(size=spectrum.size)
    z[0] = rng.normal()
    if circ.size % 2 == 0:
        z[-1] = rng.normal()
    noise = np.fft.irfft(np.sqrt(spectrum) * z, n=circ.size)[:n]
    return noise / max(float(np.std(noise)), np.finfo(float).tiny)


def linear_readout_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    gram.flat[:: gram.shape[0] + 1] += READOUT_RIDGE
    return np.linalg.solve(gram, design.conj().T @ y)


def linear_readout_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) ** 2) /
                 max(np.mean(np.abs(y_true) ** 2), np.finfo(float).tiny))


def complex_pca_fit(X: np.ndarray):
    mean = np.mean(X, axis=0)
    _, singular, vh = np.linalg.svd(X - mean, full_matrices=False)
    return mean, singular, vh.conj().T


def pca_project(X: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int):
    return (X - mean) @ components[:, :k]


def pca_energy(singular: np.ndarray, k: int) -> float:
    e = singular**2
    return float(np.sum(e[:k]) / max(np.sum(e), np.finfo(float).tiny))


def spectrum_geometry(X: np.ndarray) -> dict:
    centered = X - np.mean(X, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(X.shape[0] - 1, 1)
    eig = np.sort(np.maximum(np.real(np.linalg.eigvalsh(cov)), 0.0))[::-1]
    total = float(eig.sum())
    if total <= 0:
        return {"state_participation_dimension": 0.0,
                "state_entropy_dimension": 0.0,
                "state_rank90": 0, "state_rank99": 0}
    p = eig / total
    return {
        "state_participation_dimension": float(1.0 / np.sum(p**2)),
        "state_entropy_dimension": float(
            np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))
        ),
        "state_rank90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "state_rank99": int(np.searchsorted(np.cumsum(p), 0.99) + 1),
    }


def kernel_descriptors(gammas: np.ndarray, weights: np.ndarray) -> dict:
    w = weights / float(weights.sum())
    entropy = -float(np.sum(w * np.log(w)))
    scale = float(math.log10(gammas.max() / gammas.min()))
    t = np.linspace(0.0, 0.5, 4000)
    h_t = np.exp(-np.outer(t, gammas)) @ w
    cdf = np.cumsum(np.abs(h_t))
    cdf /= max(float(cdf[-1]), np.finfo(float).tiny)
    return {
        "dictionary_mode_count": int(len(gammas)),
        "weight_entropy_nats": entropy,
        "gfe_entropy_effective_count": float(np.exp(entropy)),
        "gfe_d_eff": float(len(gammas) * np.exp(entropy)),
        "gfe_m_scale_decades": scale,
        "gfe_m_res_modes_per_decade": float(len(gammas) / scale),
        "gfe_m_cap_s": float(np.sum(w / gammas**2) / np.sum(w / gammas)),
        "kernel_t50_s": float(t[np.searchsorted(cdf, 0.50)]),
        "kernel_t90_s": float(t[np.searchsorted(cdf, 0.90)]),
    }


KERNEL_DESCRIPTORS = {
    g: kernel_descriptors(KERNELS[g]["gammas"], KERNELS[g]["weights"])
    for g in GEOMETRIES
}


def alignment_rate_pattern(process: str, role: str) -> np.ndarray:
    """Return a 16-mode rate geometry for the 15B-2 alignment intervention."""
    if process not in {"short", "multiscale", "long", "hidden_mix"}:
        raise ValueError(process)
    if role not in {"matched", "displaced", "broad"}:
        raise ValueError(role)
    def cluster(center, spread=0.15):
        return center * np.exp(np.linspace(-spread, spread, 8))
    if process == "short":
        if role == "matched": return 40.0 * np.exp(np.linspace(-0.15, 0.15, MODES))
        if role == "displaced": return 4.0 * np.exp(np.linspace(-0.15, 0.15, MODES))
        return np.geomspace(0.5, 100.0, MODES)
    if process == "long":
        if role == "matched": return 2.0 * np.exp(np.linspace(-0.15, 0.15, MODES))
        if role == "displaced": return cluster(40.0).repeat(2)
        return np.geomspace(0.5, 100.0, MODES)
    if role == "matched": return np.r_[cluster(40.0), cluster(2.0)]
    if role == "displaced": return np.r_[cluster(4.0), cluster(0.4)]
    return np.geomspace(0.5, 100.0, MODES)

ALIGNMENT_ROLES = ("matched", "displaced", "broad")
ALIGNMENT_KERNELS = {
    process: {role: {"gammas": alignment_rate_pattern(process, role), "weights": uniform_weights()}
              for role in ALIGNMENT_ROLES}
    for process in ("short", "multiscale", "long", "hidden_mix")
}
for _process in ALIGNMENT_KERNELS:
    for _role in ALIGNMENT_ROLES:
        _spec = ALIGNMENT_KERNELS[_process][_role]
        _spec["coefficients"] = state_coefficients(_spec["gammas"])

ALIGNMENT_KERNEL_DESCRIPTORS = {
    process: {role: kernel_descriptors(spec["gammas"], spec["weights"])
              for role, spec in roles.items()}
    for process, roles in ALIGNMENT_KERNELS.items()
}
