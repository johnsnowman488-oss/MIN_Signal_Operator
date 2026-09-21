"""Experiment 12A: receiver recoverability ladder.

Each receiver sees identical signal, memory, channel, noise, training prefix,
and held-out symbols:

raw -> FIR-7 -> FIR-15 -> FIR-31 -> IIR-2 -> IIR-4 -> IIR-8.

The IIR receivers are linear recursive ARX baselines, not claimed optimal
state-space realizations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.metrics.equalization import (
    add_awgn, apply_fir_equalizer, apply_iir_equalizer,
    design_fir_equalizer, design_iir_equalizer, evm,
)
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
MEMORIES = ("identity", "exponential_tau_0.10", "soe_two_scale", "powerlaw_alpha_0.70")
CHANNELS = ("identity", "flat_rayleigh", "multipath_3tap")
RECEIVERS = ("raw", "fir7", "fir15", "fir31", "iir2", "iir4", "iir8")
SNR_DB = (0.0, 10.0, 20.0, 30.0)
NUM_SYMBOLS = 512
TRAIN_SYMBOLS = 128
SPS = 16
SYMBOL_RATE = 100.0
SEEDS = tuple(range(5))


def _bits(signal_name, symbols):
    symbols = np.asarray(symbols)
    if signal_name == "BPSK":
        return (np.real(symbols) < 0).astype(np.uint8)[:, None]
    if signal_name == "QPSK":
        phase = np.mod(np.angle(symbols) - np.pi / 4, 2 * np.pi)
        idx = np.mod(np.rint(phase / (np.pi / 2)).astype(int), 4)
        return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10)
    ii = np.argmin(abs(np.real(symbols)[:, None] - levels), axis=1)
    qq = np.argmin(abs(np.imag(symbols)[:, None] - levels), axis=1)
    return np.column_stack(((ii >> 1) & 1, ii & 1, (qq >> 1) & 1, (qq & 1))).astype(np.uint8)


def _decide(signal_name, estimate):
    if signal_name == "BPSK":
        return np.where(np.real(estimate) >= 0, 1, -1).astype(complex)
    if signal_name == "QPSK":
        constellation = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    else:
        levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10)
        constellation = (levels[:, None] + 1j * levels[None, :]).reshape(-1)
    return constellation[np.argmin(abs(np.asarray(estimate)[:, None] - constellation[None, :]), axis=1)]


def _ber(signal_name, tx, estimate):
    return float(np.mean(_bits(signal_name, tx) != _bits(signal_name, _decide(signal_name, estimate))))


def _memory_filter(x, name):
    if name == "identity":
        return x
    count = 32 if name != "powerlaw_alpha_0.70" else 64
    lags = np.arange(count * SPS) / (SYMBOL_RATE * SPS)
    if name == "exponential_tau_0.10":
        k = np.exp(-lags / 0.10)
    elif name == "soe_two_scale":
        k = 0.7 * np.exp(-2 * lags) + 0.3 * np.exp(-30 * lags)
    elif name == "powerlaw_alpha_0.70":
        k = (1 + lags / 0.5) ** -0.7
    else:
        raise ValueError(name)
    k /= np.sum(k)
    return np.convolve(x, k, mode="full")[:x.size]


def _channel(x, name, rng):
    if name == "identity":
        return x, np.array([1.0 + 0.0j])
    if name == "flat_rayleigh":
        h = (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
        h /= max(abs(h), 1e-12)
        return h * x, h
    if name == "multipath_3tap":
        phases = rng.uniform(0, 2 * np.pi, 2)
        h = np.array([1.0, 0.45 * np.exp(1j * phases[0]), 0.25 * np.exp(1j * phases[1])], dtype=complex)
        h /= np.sqrt(np.sum(abs(h) ** 2))
        return np.convolve(x, h, mode="full")[:x.size], h
    raise ValueError(name)


def _complexity(receiver):
    if receiver == "raw":
        return 0, 0, 0, 0
    if receiver.startswith("fir"):
        taps = int(receiver[3:])
        return taps, max(taps - 1, 0), taps, taps - 1
    order = int(receiver[3:])
    return 2 * order, order, 2 * order, order


def _run_case(signal_name, generator, memory, channel, seed, snr_db):
    signal = generator(NUM_SYMBOLS, samples_per_symbol=SPS, symbol_rate=SYMBOL_RATE, seed=seed)
    memory_out = _memory_filter(signal.record.samples, memory)
    channel_rng = np.random.default_rng(10000 + seed)
    channel_out, h = _channel(memory_out, channel, channel_rng)
    noise_rng = np.random.default_rng(20000 + seed * 100 + int(snr_db))
    observed = add_awgn(channel_out, snr_db, noise_rng)[signal.symbol_indices]
    tx = signal.symbols
    train = slice(0, TRAIN_SYMBOLS)
    test = slice(TRAIN_SYMBOLS, NUM_SYMBOLS)
    rows = []

    for receiver in RECEIVERS:
        pole_radius = np.nan
        if receiver == "raw":
            estimate = observed[test]
        elif receiver.startswith("fir"):
            taps = int(receiver[3:])
            coeff = design_fir_equalizer(observed[train], tx[train], taps=taps, ridge=1e-5)
            estimate = apply_fir_equalizer(observed, coeff)[test]
        else:
            order = int(receiver[3:])
            b, a, pole_radius = design_iir_equalizer(
                observed[train], tx[train], feedforward=order, feedback=order,
                ridge=1e-5, pole_radius_limit=0.98,
            )
            estimate = apply_iir_equalizer(observed, b, a)[test]

        params, states, macs, effective_memory = _complexity(receiver)
        rows.append({
            "experiment": "12A_receiver_recoverability",
            "signal": signal_name, "memory": memory,
            "channel": channel, "receiver": receiver, "snr_db": snr_db, "seed": seed,
            "channel_energy": float(np.sum(abs(h) ** 2)),
            "evm_percent": 100 * evm(tx[test], estimate),
            "ber": _ber(signal_name, tx[test], estimate),
            "heldout_mse": float(np.mean(abs(estimate - tx[test]) ** 2)),
            "parameter_count": params, "state_dimension": states,
            "macs_per_sample": macs, "effective_memory_samples": effective_memory,
            "pole_radius": pole_radius,
        })
    return rows


def main():
    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for memory_name in MEMORIES:
                for channel in CHANNELS:
                    for snr in SNR_DB:
                        case = _run_case(signal_name, generator, memory_name, channel, seed, snr)
                        for row in case:
                            row["memory"] = memory_name
                        rows.extend(case)

    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "12_receiver_recoverability_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "12A_Receiver_Recoverability",
        "date": "2026-09-21",
        "signals": list(SIGNALS), "memories": list(MEMORIES),
        "channels": list(CHANNELS), "receivers": list(RECEIVERS),
        "snr_db": list(SNR_DB), "seeds": list(SEEDS),
        "rows": len(rows),
        "evaluation": "384 held-out symbols after 128-symbol training prefix",
        "receiver_note": "IIR receivers are recursive linear ARX baselines with pole-radius stabilization; they are not claimed optimal state-space realizations.",
        "fairness_note": "Within each realization all receivers use identical signal, channel, noise, training prefix, and held-out symbols.",
    }
    (out / "12_receiver_recoverability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
