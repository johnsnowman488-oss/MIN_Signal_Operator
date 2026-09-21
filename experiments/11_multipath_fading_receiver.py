"""Experiment 11: controlled multipath and fading with receiver equalization.

This experiment separates three effects:
1. deterministic MIN memory,
2. a controlled communication channel (multipath or flat fading),
3. receiver-side compensation.

The receiver is trained on a prefix and evaluated only on held-out symbols.
No claim of communication benefit is assumed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.metrics.equalization import add_awgn, apply_fir_equalizer, design_fir_equalizer, evm
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
MEMORIES = ("identity", "exponential_tau_0.10", "soe_two_scale", "powerlaw_alpha_0.70")
CHANNELS = ("flat_rayleigh", "multipath_3tap")
RECEIVERS = ("raw", "fir7", "fir15")
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
    return np.column_stack(((ii >> 1) & 1, ii & 1, (qq >> 1) & 1, qq & 1)).astype(np.uint8)


def _decide(signal_name, estimate):
    if signal_name == "BPSK":
        return np.where(np.real(estimate) >= 0, 1, -1).astype(complex)
    if signal_name == "QPSK":
        c = np.exp(1j * (np.pi/4 + np.arange(4)*np.pi/2))
    else:
        l = np.array([-3., -1., 1., 3.]) / np.sqrt(10)
        c = (l[:, None] + 1j*l[None, :]).reshape(-1)
    return c[np.argmin(abs(np.asarray(estimate)[:, None] - c[None, :]), axis=1)]


def _ber(signal_name, tx, estimate):
    return float(np.mean(_bits(signal_name, tx) != _bits(signal_name, _decide(signal_name, estimate))))


def _memory_filter(x, name, sps):
    if name == "identity":
        return x
    # Symbol-spaced FIR approximations of the same causal memory regimes.
    if name == "exponential_tau_0.10":
        tau = 0.10
        lags = np.arange(32*sps) / (SYMBOL_RATE*SPS)
        k = np.exp(-lags/tau)
    elif name == "soe_two_scale":
        lags = np.arange(32*sps) / (SYMBOL_RATE*SPS)
        k = .7*np.exp(-2*lags) + .3*np.exp(-30*lags)
    elif name == "powerlaw_alpha_0.70":
        lags = np.arange(64*sps) / (SYMBOL_RATE*SPS)
        k = (1 + lags/.5)**-.7
    else:
        raise ValueError(name)
    # Normalize the discrete kernel so memory changes temporal shape without
    # introducing an arbitrary power gain.
    k = k / np.sum(k)
    return np.convolve(x, k, mode="full")[:x.size]


def _channel(x, channel_name, rng):
    if channel_name == "flat_rayleigh":
        h = (rng.normal() + 1j*rng.normal()) / np.sqrt(2)
        h = h / max(abs(h), 1e-12)
        return h*x, h
    if channel_name == "multipath_3tap":
        phases = rng.uniform(0, 2*np.pi, 2)
        h = np.array([
            1.0,
            .45*np.exp(1j*phases[0]),
            .25*np.exp(1j*phases[1]),
        ], dtype=complex)
        h /= np.sqrt(np.sum(abs(h)**2))
        return np.convolve(x, h, mode="full")[:x.size], h
    raise ValueError(channel_name)


def _run_case(signal_name, generator, memory, channel, seed, snr_db):
    signal = generator(NUM_SYMBOLS, samples_per_symbol=SPS, symbol_rate=SYMBOL_RATE, seed=seed)
    x = signal.record.samples
    memory_out = _memory_filter(x, memory, SPS)
    channel_rng = np.random.default_rng(10000 + seed)
    channel_out, h = _channel(memory_out, channel, channel_rng)
    noise_rng = np.random.default_rng(20000 + seed*100 + int(snr_db))
    noisy = add_awgn(channel_out, snr_db, noise_rng)
    observed = noisy[signal.symbol_indices]
    tx = signal.symbols
    train = slice(0, TRAIN_SYMBOLS)
    test = slice(TRAIN_SYMBOLS, NUM_SYMBOLS)

    rows = []
    for receiver in RECEIVERS:
        taps = {"raw": 1, "fir7": 7, "fir15": 15}[receiver]
        if receiver == "raw":
            estimate = observed[test]
        else:
            coeff = design_fir_equalizer(observed[train], tx[train], taps=taps, ridge=1e-5)
            estimate = apply_fir_equalizer(observed, coeff)[test]
        rows.append({
            "experiment": "11_multipath_fading_receiver",
            "signal": signal_name,
            "memory": memory,
            "channel": channel,
            "receiver": receiver,
            "snr_db": snr_db,
            "seed": seed,
            "channel_energy": float(np.sum(abs(h)**2)),
            "evm_percent": 100*evm(tx[test], estimate),
            "ber": _ber(signal_name, tx[test], estimate),
        })
    return rows


def main():
    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for memory in MEMORIES:
                for channel in CHANNELS:
                    for snr in SNR_DB:
                        rows.extend(_run_case(signal_name, generator, memory, channel, seed, snr))

    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "11_multipath_fading_receiver_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader(); w.writerows(rows)

    summary = {}
    for memory in MEMORIES:
        summary[memory] = {}
        for channel in CHANNELS:
            summary[memory][channel] = {}
            for receiver in RECEIVERS:
                summary[memory][channel][receiver] = {}
                for snr in SNR_DB:
                    v = [r for r in rows if r["memory"] == memory and r["channel"] == channel
                         and r["receiver"] == receiver and r["snr_db"] == snr]
                    summary[memory][channel][receiver][f"{snr:g}dB"] = {
                        "mean_evm_percent": float(np.mean([r["evm_percent"] for r in v])),
                        "mean_ber": float(np.mean([r["ber"] for r in v])),
                    }

    (out / "11_multipath_fading_receiver_summary.json").write_text(json.dumps({
        "experiment": "11_Multipath_Fading_Receiver",
        "date": "2026-09-21",
        "signals": list(SIGNALS),
        "memories": list(MEMORIES),
        "channels": list(CHANNELS),
        "receivers": list(RECEIVERS),
        "snr_db": list(SNR_DB),
        "seeds": list(SEEDS),
        "rows": len(rows),
        "evaluation": "384 held-out symbols after 128-symbol training prefix",
        "note": "This is a controlled channel/receiver experiment; it does not establish an intrinsic MIN communication advantage."
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
