"""Receiver-side scalar and linear equalization helpers for MIN experiments."""

from __future__ import annotations

import numpy as np


def complex_gain_fit(reference: np.ndarray, observed: np.ndarray) -> complex:
    """Least-squares complex scalar mapping observed ~= gain*reference."""
    reference = np.asarray(reference)
    observed = np.asarray(observed)
    if reference.ndim != 1 or observed.ndim != 1 or reference.size != observed.size:
        raise ValueError("reference and observed must be 1D arrays of equal length")
    denom = np.vdot(reference, reference)
    if abs(denom) == 0:
        raise ValueError("reference must have nonzero energy")
    return np.vdot(reference, observed) / denom


def apply_scalar_compensation(
    observed: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Remove the best-fit complex scalar from observed."""
    gain = complex_gain_fit(reference, observed)
    if abs(gain) == 0:
        raise ValueError("fitted gain must be nonzero")
    return np.asarray(observed) / gain


def design_fir_equalizer(
    received: np.ndarray,
    desired: np.ndarray,
    taps: int = 7,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Design a causal symbol-spaced complex FIR equalizer.

    The model is desired[n] ~= sum_k taps[k] * received[n-k].
    The first taps-1 samples are excluded so every training row has a
    complete causal history. Ridge regularization stabilizes noisy training.
    """
    received = np.asarray(received)
    desired = np.asarray(desired)
    if received.ndim != 1 or desired.ndim != 1 or received.size != desired.size:
        raise ValueError("received and desired must be 1D arrays of equal length")
    if taps < 1:
        raise ValueError("taps must be positive")
    if received.size <= taps:
        raise ValueError("training sequence is too short for requested taps")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    X = np.column_stack(
        [received[taps - 1 - k : received.size - k] for k in range(taps)]
    )
    d = desired[taps - 1 :]
    gram = X.conj().T @ X
    if ridge:
        gram = gram + ridge * np.eye(taps)
    return np.linalg.solve(gram, X.conj().T @ d)


def apply_fir_equalizer(received: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply a causal symbol-spaced FIR equalizer."""
    received = np.asarray(received)
    coefficients = np.asarray(coefficients)
    if received.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("received and coefficients must be one-dimensional")
    taps = coefficients.size
    if taps < 1:
        raise ValueError("at least one equalizer tap is required")
    output = np.zeros(received.size, dtype=np.result_type(received, coefficients))
    for n in range(taps - 1, received.size):
        output[n] = np.dot(coefficients, received[n - np.arange(taps)])
    return output


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex AWGN at a specified measured signal-power SNR.

    For complex baseband samples, noise real and imaginary components each
    receive half of the total noise variance.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    power = float(np.mean(np.abs(signal) ** 2))
    if power <= 0 or not np.isfinite(power):
        raise ValueError("signal must have finite nonzero average power")
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = (
        rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)
    ) * np.sqrt(noise_power / 2.0)
    return signal + noise


def evm(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return RMS EVM after no additional gain fitting."""
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)
    if reference.ndim != 1 or estimate.ndim != 1 or reference.size != estimate.size:
        raise ValueError("reference and estimate must be 1D arrays of equal length")
    denom = np.mean(np.abs(reference) ** 2)
    if denom <= 0:
        raise ValueError("reference must have nonzero power")
    return float(np.sqrt(np.mean(np.abs(estimate - reference) ** 2) / denom))
