"""Signal containers and common time-axis/normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class SignalRecord:
    """Reproducible signal plus the metadata needed to regenerate it."""

    samples: np.ndarray
    sample_rate: float
    signal_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "samples", samples)

    @property
    def duration(self) -> float:
        return (self.samples.size - 1) / self.sample_rate

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.samples)

    @property
    def energy(self) -> float:
        return float(np.sum(np.abs(self.samples) ** 2))

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.abs(self.samples) ** 2)))


def make_time_axis(duration: float, sample_rate: float) -> np.ndarray:
    """Return a uniform axis including both endpoints when representable."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    n = int(round(duration * sample_rate))
    if n < 1:
        raise ValueError("duration*sample_rate must be at least one")
    return np.arange(n + 1, dtype=float) / sample_rate


def unit_energy(samples: np.ndarray) -> np.ndarray:
    """Normalize a signal to unit discrete l2 energy."""
    x = np.asarray(samples)
    energy = np.sum(np.abs(x) ** 2)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("signal must have finite nonzero energy")
    return x / np.sqrt(energy)
