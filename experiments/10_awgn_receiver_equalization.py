"""Receiver-compensated AWGN experiment for MIN digital signals.

Experiment 10 asks whether temporal distortion introduced by MIN memory can be
compensated at the receiver, and how residual EVM/BER scales with SNR.

The equalizer is trained on the first 128 transmitted symbols and evaluated on
the remaining held-out symbols. AWGN is injected before symbol sampling.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min import MemoryOperator
from min.kernels import exponential_kernel, power_law_kernel
from min.metrics.equalization import (
    add_awgn,
    apply_fir_equalizer,
    apply_scalar_compensation,
    design_fir_equalizer,
    evm,
)
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
KERNELS = (
    "identity",
    "exponential_tau_0.10",
    "soe_two_scale",
    "powerlaw_alpha_0.70",
)
RECEIVERS = ("raw", "scalar", "fir7")
SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
NUM_SYMBOLS = 512
TRAIN_SYMBOLS = 128
SPS = 16
SYMBOL_RATE = 100.0
FIR_TAPS = 7
SEEDS = tuple(range(5))


def _memory_kernel(name: str, lag: np.ndarray) -> np.ndarray:
    if name == "exponential_tau_0.10":
        return exponential_kernel(lag, 0.10)
    if name == "soe_two_scale":
        return 0.7 * np.exp(-2.0 * lag) + 0.3 * np.exp(-30.0 * lag)
    if name == "powerlaw_alpha_0.70":
        return power_law_kernel(lag, alpha=0.70, tau=0.50)
    raise ValueError(name)


def apply_min_uniform(name: str, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply the same causal trapezoidal MIN rule as MemoryOperator efficiently."""
    if name == "identity":
        return x.copy()
    t = np.asarray(t, dtype=float)
    x = np.asarray(x)
    if t.size != x.size or t.size < 2 or not np.allclose(np.diff(t), np.diff(t)[0]):
        raise ValueError("t must be uniform and match x")
    dt = float(t[1] - t[0])
    lags = np.arange(t.size, dtype=float) * dt
    k = _memory_kernel(name, lags)
    n = x.size
    size = 1 << (2 * n - 1).bit_length()
    conv = np.fft.ifft(np.fft.fft(k, size) * np.fft.fft(x, size))[:n]
    y = dt * conv
    # Trapezoidal endpoint corrections relative to ordinary discrete convolution.
    y -= 0.5 * dt * (k[:n] * x[0] + k[0] * x)
    return y


def _bits_from_symbols(signal_name: str, symbols: np.ndarray) -> np.ndarray:
    """Return deterministic binary labels matching the generator constellation indexing."""
    symbols = np.asarray(symbols)
    if signal_name == "BPSK":
        values = (np.real(symbols) < 0).astype(np.uint8)
        return values[:, None]
    if signal_name == "QPSK":
        phase = np.mod(np.angle(symbols) - np.pi / 4.0, 2.0 * np.pi)
        idx = np.mod(np.rint(phase / (np.pi / 2.0)).astype(int), 4)
        return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)
    if signal_name == "16QAM":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        ii = np.argmin(np.abs(np.real(symbols)[:, None] - levels[None, :]), axis=1)
        qq = np.argmin(np.abs(np.imag(symbols)[:, None] - levels[None, :]), axis=1)
        return np.column_stack(
            ((ii >> 1) & 1, ii & 1, (qq >> 1) & 1, qq & 1)
        ).astype(np.uint8)
    raise ValueError(signal_name)


def _nearest_symbols(signal_name: str, estimate: np.ndarray) -> np.ndarray:
    if signal_name == "BPSK":
        return np.where(np.real(estimate) >= 0, 1.0, -1.0).astype(complex)
    if signal_name == "QPSK":
        constellation = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    elif signal_name == "16QAM":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        constellation = (levels[:, None] + 1j * levels[None, :]).reshape(-1)
    else:
        raise ValueError(signal_name)
    estimate = np.asarray(estimate)
    return constellation[np.argmin(np.abs(estimate[:, None] - constellation[None, :]), axis=1)]


def _ber(signal_name: str, transmitted: np.ndarray, estimate: np.ndarray) -> float:
    tx_bits = _bits_from_symbols(signal_name, transmitted)
    decided = _nearest_symbols(signal_name, estimate)
    rx_bits = _bits_from_symbols(signal_name, decided)
    return float(np.mean(tx_bits != rx_bits))


def _run_case(signal_name: str, generator, kernel: str, seed: int, snr_db: float):
    signal = generator(
        NUM_SYMBOLS,
        samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE,
        seed=seed,
    )
    x = signal.record.samples
    t = np.arange(x.size) / signal.record.sample_rate
    clean = apply_min_uniform(kernel, t, x)
    rng = np.random.default_rng(seed * 1000 + int(snr_db * 10) + 17)
    noisy = add_awgn(clean, snr_db, rng)
    observed = noisy[signal.symbol_indices]
    transmitted = signal.symbols
    train = slice(0, TRAIN_SYMBOLS)
    test = slice(TRAIN_SYMBOLS, NUM_SYMBOLS)

    scalar_gain = np.vdot(transmitted[train], observed[train]) / np.vdot(
        transmitted[train], transmitted[train]
    )
    scalar_test = observed[test] / scalar_gain
    coeff = design_fir_equalizer(
        observed[train],
        transmitted[train],
        taps=FIR_TAPS,
        ridge=1e-6,
    )
    equalized_all = apply_fir_equalizer(observed, coeff)
    fir_test = equalized_all[test]

    estimates = {
        "raw": observed[test],
        "scalar": scalar_test,
        "fir7": fir_test,
    }
    rows = []
    for receiver, estimate in estimates.items():
        rows.append(
            {
                "experiment": "10_awgn_receiver_equalization",
                "signal": signal_name,
                "kernel": kernel,
                "receiver": receiver,
                "snr_db": snr_db,
                "seed": seed,
                "num_symbols": NUM_SYMBOLS,
                "train_symbols": TRAIN_SYMBOLS,
                "test_symbols": NUM_SYMBOLS - TRAIN_SYMBOLS,
                "fir_taps": FIR_TAPS if receiver == "fir7" else 0,
                "received_power": float(np.mean(np.abs(observed) ** 2)),
                "evm_percent": 100.0 * evm(transmitted[test], estimate),
                "ber": _ber(signal_name, transmitted[test], estimate),
            }
        )
    return rows


def main() -> None:
    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for kernel in KERNELS:
                for snr_db in SNR_DB:
                    rows.extend(_run_case(signal_name, generator, kernel, seed, snr_db))

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "10_awgn_receiver_equalization_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for signal_name in SIGNALS:
        summary[signal_name] = {}
        for kernel in KERNELS:
            summary[signal_name][kernel] = {}
            for receiver in RECEIVERS:
                for snr_db in SNR_DB:
                    values = [
                        r for r in rows
                        if r["signal"] == signal_name
                        and r["kernel"] == kernel
                        and r["receiver"] == receiver
                        and r["snr_db"] == snr_db
                    ]
                    key = f"{snr_db:g}dB"
                    summary[signal_name][kernel].setdefault(receiver, {})[key] = {
                        "mean_evm_percent": float(np.mean([r["evm_percent"] for r in values])),
                        "std_evm_percent": float(np.std([r["evm_percent"] for r in values], ddof=1)),
                        "mean_ber": float(np.mean([r["ber"] for r in values])),
                        "std_ber": float(np.std([r["ber"] for r in values], ddof=1)),
                    }

    summary_path = out / "10_awgn_receiver_equalization_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "10_AWGN_Receiver_Equalization",
                "date": "2026-09-21",
                "seeds": list(SEEDS),
                "signals": list(SIGNALS),
                "kernels": list(KERNELS),
                "receivers": list(RECEIVERS),
                "snr_db": list(SNR_DB),
                "equalizer": {
                    "type": "causal symbol-spaced complex FIR",
                    "taps": FIR_TAPS,
                    "training_symbols": TRAIN_SYMBOLS,
                    "ridge": 1e-6,
                },
                "noise_definition": "complex AWGN scaled to measured clean waveform average power at requested SNR",
                "evaluation": "held-out symbols only",
                "note": "Results test receiver compensation; no claim of intrinsic MIN communication gain is made.",
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
