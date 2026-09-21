"""Elementary deterministic signal generators."""

from __future__ import annotations

import numpy as np

from .records import SignalRecord, make_time_axis, unit_energy


def _record(
    samples: np.ndarray,
    sample_rate: float,
    signal_type: str,
    parameters: dict,
    seed: int | None = None,
    normalize: bool = False,
) -> SignalRecord:
    samples = unit_energy(samples) if normalize else np.asarray(samples)
    return SignalRecord(samples, sample_rate, signal_type, parameters, seed)


def generate_impulse(
    duration: float,
    sample_rate: float,
    index: int = 0,
    amplitude: complex = 1.0,
    normalize: bool = False,
) -> SignalRecord:
    """Generate a discrete impulse."""
    t = make_time_axis(duration, sample_rate)
    if index < 0 or index >= t.size:
        raise ValueError("index must lie within the sampled signal")
    x = np.zeros(t.size, dtype=np.result_type(amplitude, float))
    x[index] = amplitude
    return _record(x, sample_rate, "impulse", {"index": index, "amplitude": amplitude}, normalize=normalize)


def generate_rectangular_pulse(
    duration: float,
    sample_rate: float,
    start: float,
    width: float,
    amplitude: complex = 1.0,
    normalize: bool = False,
) -> SignalRecord:
    """Generate a rectangular pulse on [start, start+width)."""
    if start < 0 or width <= 0:
        raise ValueError("start must be nonnegative and width must be positive")
    t = make_time_axis(duration, sample_rate)
    x = np.where((t >= start) & (t < start + width), amplitude, 0)
    return _record(
        x,
        sample_rate,
        "rectangular_pulse",
        {"start": start, "width": width, "amplitude": amplitude},
        normalize=normalize,
    )


def generate_gaussian_pulse(
    duration: float,
    sample_rate: float,
    center: float,
    sigma: float,
    amplitude: complex = 1.0,
    normalize: bool = False,
) -> SignalRecord:
    """Generate A*exp(-(t-center)^2/(2 sigma^2))."""
    if sigma <= 0 or center < 0 or center > duration:
        raise ValueError("center must be inside the interval and sigma must be positive")
    t = make_time_axis(duration, sample_rate)
    x = amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return _record(
        x,
        sample_rate,
        "gaussian_pulse",
        {"center": center, "sigma": sigma, "amplitude": amplitude},
        normalize=normalize,
    )


def generate_sine(
    duration: float,
    sample_rate: float,
    frequency: float,
    amplitude: complex = 1.0,
    phase: float = 0.0,
    normalize: bool = False,
) -> SignalRecord:
    """Generate A*cos(2 pi f t + phase)."""
    if frequency < 0:
        raise ValueError("frequency must be nonnegative")
    t = make_time_axis(duration, sample_rate)
    if np.iscomplexobj(amplitude):
        x = amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))
    else:
        x = amplitude * np.cos(2 * np.pi * frequency * t + phase)
    return _record(
        x,
        sample_rate,
        "sine",
        {"frequency": frequency, "amplitude": amplitude, "phase": phase},
        normalize=normalize,
    )


def generate_multitone(
    duration: float,
    sample_rate: float,
    frequencies: np.ndarray | list[float],
    amplitudes: np.ndarray | list[complex] | None = None,
    phases: np.ndarray | list[float] | None = None,
    normalize: bool = False,
) -> SignalRecord:
    """Generate a sum of tones with independently specified frequencies."""
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies < 0):
        raise ValueError("frequencies must be a nonempty 1D nonnegative array")
    if amplitudes is None:
        amplitudes = np.ones(frequencies.size)
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim != 1 or amplitudes.size != frequencies.size:
        raise ValueError("amplitudes must match frequencies")
    if phases is None:
        phases = np.zeros(frequencies.size)
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or phases.size != frequencies.size:
        raise ValueError("phases must match frequencies")
    t = make_time_axis(duration, sample_rate)
    x = np.zeros(t.size, dtype=np.result_type(amplitudes, float))
    for f, a, p in zip(frequencies, amplitudes, phases):
        x = x + a * np.cos(2 * np.pi * f * t + p)
    return _record(
        x,
        sample_rate,
        "multitone",
        {"frequencies": frequencies.tolist(), "amplitudes": amplitudes.tolist(), "phases": phases.tolist()},
        normalize=normalize,
    )


def generate_linear_chirp(
    duration: float,
    sample_rate: float,
    f0: float,
    f1: float,
    amplitude: complex = 1.0,
    phase: float = 0.0,
    normalize: bool = False,
) -> SignalRecord:
    """Generate a linear-frequency chirp from f0 to f1 over duration."""
    if f0 < 0 or f1 < 0:
        raise ValueError("f0 and f1 must be nonnegative")
    if duration <= 0:
        raise ValueError("duration must be positive")
    t = make_time_axis(duration, sample_rate)
    k = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2) + phase
    if np.iscomplexobj(amplitude):
        x = amplitude * np.exp(1j * phi)
    else:
        x = amplitude * np.cos(phi)
    return _record(
        x,
        sample_rate,
        "linear_chirp",
        {"f0": f0, "f1": f1, "amplitude": amplitude, "phase": phase},
        normalize=normalize,
    )
