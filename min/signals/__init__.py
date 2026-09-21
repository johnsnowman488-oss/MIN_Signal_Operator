"""Synthetic and digital signal generators."""

from .digital import DigitalSignal, generate_16qam, generate_bpsk, generate_qpsk
from .elementary import (
    generate_gaussian_pulse,
    generate_impulse,
    generate_linear_chirp,
    generate_multitone,
    generate_rectangular_pulse,
    generate_sine,
)
from .records import SignalRecord, make_time_axis, unit_energy

__all__ = [
    "DigitalSignal",
    "SignalRecord",
    "generate_16qam",
    "generate_bpsk",
    "generate_gaussian_pulse",
    "generate_impulse",
    "generate_linear_chirp",
    "generate_multitone",
    "generate_qpsk",
    "generate_rectangular_pulse",
    "generate_sine",
    "make_time_axis",
    "unit_energy",
]
