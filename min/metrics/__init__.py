"""Signal-processing metrics used by MIN experiments."""

from .signal import (
    energy,
    rms,
    peak_magnitude,
    mse,
    nmse,
    correlation,
    delay_of_max_correlation,
)

__all__ = [
    "energy",
    "rms",
    "peak_magnitude",
    "mse",
    "nmse",
    "correlation",
    "delay_of_max_correlation",
]
