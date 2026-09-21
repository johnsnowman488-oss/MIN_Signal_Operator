"""Controlled digital communication signal generators for MIN experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .records import SignalRecord, unit_energy


@dataclass(frozen=True)
class DigitalSignal:
    """Complex baseband waveform plus transmitted symbols."""

    record: SignalRecord
    symbols: np.ndarray
    samples_per_symbol: int
    symbol_indices: np.ndarray


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _waveform(
    symbols: np.ndarray,
    samples_per_symbol: int,
    sample_rate: float,
    seed: int | None,
    signal_type: str,
    normalize: bool,
) -> DigitalSignal:
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be at least 2")
    waveform = np.repeat(symbols, samples_per_symbol).astype(complex)
    if normalize:
        waveform = unit_energy(waveform)
    record = SignalRecord(
        waveform,
        sample_rate,
        signal_type,
        {
            "num_symbols": int(symbols.size),
            "samples_per_symbol": samples_per_symbol,
            "normalize": normalize,
        },
        seed,
    )
    return DigitalSignal(
        record=record,
        symbols=np.asarray(symbols, dtype=complex),
        samples_per_symbol=samples_per_symbol,
        symbol_indices=np.arange(symbols.size) * samples_per_symbol
        + samples_per_symbol // 2,
    )


def generate_bpsk(
    num_symbols: int,
    samples_per_symbol: int = 16,
    symbol_rate: float = 100.0,
    seed: int | None = 0,
    normalize: bool = False,
) -> DigitalSignal:
    """Generate random BPSK rectangular-pulse baseband."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be positive")
    bits = _rng(seed).integers(0, 2, size=num_symbols)
    symbols = (2 * bits - 1).astype(float)
    return _waveform(
        symbols, samples_per_symbol, symbol_rate * samples_per_symbol,
        seed, "bpsk", normalize,
    )


def generate_qpsk(
    num_symbols: int,
    samples_per_symbol: int = 16,
    symbol_rate: float = 100.0,
    seed: int | None = 0,
    normalize: bool = False,
) -> DigitalSignal:
    """Generate random unit-energy QPSK symbols with Gray-style quadrant mapping."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be positive")
    rng = _rng(seed)
    values = rng.integers(0, 4, size=num_symbols)
    phase = np.pi / 4 + values * np.pi / 2
    symbols = np.exp(1j * phase)
    return _waveform(
        symbols, samples_per_symbol, symbol_rate * samples_per_symbol,
        seed, "qpsk", normalize,
    )


def generate_16qam(
    num_symbols: int,
    samples_per_symbol: int = 16,
    symbol_rate: float = 100.0,
    seed: int | None = 0,
    normalize: bool = False,
) -> DigitalSignal:
    """Generate random normalized square 16-QAM symbols."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be positive")
    rng = _rng(seed)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    i = levels[rng.integers(0, 4, size=num_symbols)]
    q = levels[rng.integers(0, 4, size=num_symbols)]
    symbols = (i + 1j * q) / np.sqrt(10.0)
    return _waveform(
        symbols, samples_per_symbol, symbol_rate * samples_per_symbol,
        seed, "16qam", normalize,
    )
