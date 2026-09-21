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
from .frequency import (
    complex_tone,
    estimate_complex_gain,
    exponential_frequency_response,
    soe_frequency_response,
    finite_horizon_kernel_response,
    group_delay,
)

__all__ = [
    "energy",
    "rms",
    "peak_magnitude",
    "mse",
    "nmse",
    "correlation",
    "delay_of_max_correlation",
    "complex_tone",
    "estimate_complex_gain",
    "exponential_frequency_response",
    "soe_frequency_response",
    "finite_horizon_kernel_response",
    "group_delay",
]
