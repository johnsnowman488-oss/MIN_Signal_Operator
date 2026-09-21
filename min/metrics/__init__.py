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
from .equalization import (
    add_awgn,
    apply_fir_equalizer,
    apply_scalar_compensation,
    complex_gain_fit,
    design_fir_equalizer,
    evm,
)

__all__ = [
    "energy", "rms", "peak_magnitude", "mse", "nmse", "correlation",
    "delay_of_max_correlation", "complex_tone", "estimate_complex_gain",
    "exponential_frequency_response", "soe_frequency_response",
    "finite_horizon_kernel_response", "group_delay", "add_awgn",
    "apply_fir_equalizer", "apply_scalar_compensation", "complex_gain_fit",
    "design_fir_equalizer", "evm",
]
