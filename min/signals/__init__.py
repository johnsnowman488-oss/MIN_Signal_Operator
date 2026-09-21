"""Synthetic signal generators and signal records for MIN experiments."""

from .records import SignalRecord, make_time_axis, unit_energy
from .elementary import (
    generate_impulse,
    generate_rectangular_pulse,
    generate_gaussian_pulse,
    generate_sine,
    generate_multitone,
    generate_linear_chirp,
)

__all__ = [
    "SignalRecord",
    "make_time_axis",
    "unit_energy",
    "generate_impulse",
    "generate_rectangular_pulse",
    "generate_gaussian_pulse",
    "generate_sine",
    "generate_multitone",
    "generate_linear_chirp",
]
