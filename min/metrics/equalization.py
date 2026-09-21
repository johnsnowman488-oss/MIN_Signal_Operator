"""Receiver-side scalar, FIR, and IIR equalization helpers for MIN experiments."""

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


def apply_scalar_compensation(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove the best-fit complex scalar from observed."""
    gain = complex_gain_fit(reference, observed)
    if abs(gain) == 0:
        raise ValueError("fitted gain must be nonzero")
    return np.asarray(observed) / gain


def design_fir_equalizer(received: np.ndarray, desired: np.ndarray, taps: int = 7, ridge: float = 1e-8) -> np.ndarray:
    """Design a causal symbol-spaced complex FIR equalizer."""
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
    X = np.column_stack([received[taps - 1 - k : received.size - k] for k in range(taps)])
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


def design_iir_equalizer(
    received: np.ndarray,
    desired: np.ndarray,
    feedforward: int = 4,
    feedback: int = 4,
    ridge: float = 1e-6,
    pole_radius_limit: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a causal complex ARX/IIR equalizer.

    The model is d[n] ~= sum b[k] r[n-k] + sum a[k] d[n-k-1].
    During inference previous equalizer outputs are fed back recursively.
    """
    received = np.asarray(received)
    desired = np.asarray(desired)
    if received.ndim != 1 or desired.ndim != 1 or received.size != desired.size:
        raise ValueError("received and desired must be 1D arrays of equal length")
    if feedforward < 1 or feedback < 1:
        raise ValueError("feedforward and feedback orders must be positive")
    if received.size <= max(feedforward, feedback):
        raise ValueError("training sequence is too short for requested orders")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if not 0 < pole_radius_limit < 1:
        raise ValueError("pole_radius_limit must lie in (0, 1)")

    order = max(feedforward, feedback)
    rows = []
    target = desired[order:]
    for n in range(order, desired.size):
        rows.append(
            [received[n - k] for k in range(feedforward)]
            + [desired[n - k - 1] for k in range(feedback)]
        )
    X = np.asarray(rows, dtype=complex)
    gram = X.conj().T @ X + ridge * np.eye(X.shape[1])
    coeff = np.linalg.solve(gram, X.conj().T @ target)
    b = coeff[:feedforward]
    a = coeff[feedforward:]

    companion = np.zeros((feedback, feedback), dtype=complex)
    companion[0, :] = a
    if feedback > 1:
        companion[1:, :-1] = np.eye(feedback - 1)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius > pole_radius_limit:
        a = a * (pole_radius_limit / radius)
        companion[0, :] = a
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    return b, a, radius


def apply_iir_equalizer(received: np.ndarray, feedforward: np.ndarray, feedback: np.ndarray) -> np.ndarray:
    """Apply a causal recursive complex IIR equalizer."""
    received = np.asarray(received)
    feedforward = np.asarray(feedforward)
    feedback = np.asarray(feedback)
    if received.ndim != 1 or feedforward.ndim != 1 or feedback.ndim != 1:
        raise ValueError("all inputs must be one-dimensional")
    if feedforward.size < 1 or feedback.size < 1:
        raise ValueError("feedforward and feedback coefficients must be nonempty")
    y = np.zeros(received.size, dtype=np.result_type(received, feedforward, feedback))
    for n in range(received.size):
        value = 0.0j
        for k in range(feedforward.size):
            if n - k >= 0:
                value += feedforward[k] * received[n - k]
        for k in range(feedback.size):
            if n - k - 1 >= 0:
                value += feedback[k] * y[n - k - 1]
        y[n] = value
    return y


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN at a specified measured signal-power SNR."""
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    power = float(np.mean(np.abs(signal) ** 2))
    if power <= 0 or not np.isfinite(power):
        raise ValueError("signal must have finite nonzero average power")
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = (rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)) * np.sqrt(noise_power / 2.0)
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
