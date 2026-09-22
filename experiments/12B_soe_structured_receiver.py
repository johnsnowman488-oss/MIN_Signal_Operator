"""Experiment 12B: SOE/MIN-structured receiver.

The receiver fixes its recursive denominator from the known sampled SOE
weights/gammas and learns only the feedforward numerator on the training
prefix. This is deliberately different from a generic fitted IIR.
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
    add_awgn,
    apply_fir_equalizer,
    apply_iir_equalizer,
    design_fir_equalizer,
    design_iir_equalizer,
    design_fixed_denominator_equalizer,
    evm,
)
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
SOE_MEMORIES = {
    "exponential_tau_0.10": (np.array([1.0]), np.array([10.0])),
    "soe_two_scale": (np.array([0.7, 0.3]), np.array([2.0, 30.0])),
}
CHANNELS = ("identity", "flat_rayleigh", "multipath_3tap")
RECEIVERS = ("raw", "fir7", "iir2", "soe_exact", "soe_stable")
SNR_DB = (0.0, 10.0, 20.0, 30.0)
NUM_SYMBOLS = 512
TRAIN_SYMBOLS = 128
SPS = 16
SYMBOL_RATE = 100.0
SEEDS = tuple(range(5))
DT = 1.0 / SYMBOL_RATE
SAMPLE_DT = 1.0 / (SYMBOL_RATE * SPS)


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
        constellation = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    else:
        levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10)
        constellation = (levels[:, None] + 1j * levels[None, :]).reshape(-1)
    return constellation[np.argmin(abs(np.asarray(estimate)[:, None] - constellation[None, :]), axis=1)]


def _ber(signal_name, tx, estimate):
    return float(np.mean(_bits(signal_name, tx) != _bits(signal_name, _decide(signal_name, estimate))))


def _memory_filter(x, name):
    weights, gammas = SOE_MEMORIES[name]
    count = 32
    lags = np.arange(count * SPS) * SAMPLE_DT
    k = sum(w * np.exp(-g * lags) for w, g in zip(weights, gammas))
    k /= np.sum(k)
    return np.convolve(x, k, mode="full")[:x.size]

def _soe_inverse_feedback(memory):
    """Derive the inverse denominator from the actual symbol-rate SOE realization."""
    weights, gammas = SOE_MEMORIES[memory]
    x = np.zeros(512, dtype=float)
    x[:SPS] = 1.0
    lags = np.arange(32 * SPS) * SAMPLE_DT
    k = sum(w * np.exp(-g * lags) for w, g in zip(weights, gammas))
    k /= np.sum(k)
    y = np.convolve(x, k, mode="full")[:x.size]
    h = y[np.arange(SPS // 2, x.size, SPS)]
    poles = np.exp(-gammas * DT)
    denominator = np.array([1.0], dtype=complex)
    for pole in poles:
        denominator = np.polynomial.polynomial.polymul(denominator, [1.0, -pole])
    forward_numerator = np.convolve(h, denominator)[:denominator.size]
    forward_numerator /= forward_numerator[0]

    exact_feedback = -forward_numerator[1:]
    roots = np.roots(forward_numerator[::-1]) if forward_numerator.size > 1 else np.array([])
    stable_roots = np.array([
        1.0 / np.conj(root) if abs(root) > 1.0 else root for root in roots
    ])
    if stable_roots.size:
        stable_denominator = np.array([1.0], dtype=complex)
        for root in stable_roots:
            stable_denominator = np.polynomial.polynomial.polymul(
                stable_denominator, [1.0, -root]
            )
        stable_feedback = -stable_denominator[1:]
    else:
        stable_feedback = np.array([], dtype=complex)
    return exact_feedback, stable_feedback, float(np.max(np.abs(roots))) if roots.size else 0.0


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


def _complexity(receiver, memory):
    if receiver == "raw":
        return 0, 0, 0
    if receiver == "fir7":
        return 7, 6, 7
    if receiver == "iir2":
        return 4, 2, 4
    weights, _ = SOE_MEMORIES[memory]
    m = len(weights)
    # The sampled rectangular-pulse realization contributes m+1 learned
    # numerator taps and m recursive states in the inverse denominator.
    return 2 * m + 1, m, 2 * m + 1


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

    weights, gammas = SOE_MEMORIES[memory]
    for receiver in RECEIVERS:
        pole_radius = np.nan
        inverse_zero_radius = np.nan
        if receiver == "raw":
            estimate = observed[test]
        elif receiver == "fir7":
            coeff = design_fir_equalizer(observed[train], tx[train], taps=7, ridge=1e-5)
            estimate = apply_fir_equalizer(observed, coeff)[test]
        elif receiver == "iir2":
            b, a, pole_radius = design_iir_equalizer(
                observed[train], tx[train],
                feedforward=2, feedback=2, ridge=1e-5, pole_radius_limit=0.98,
            )
            estimate = apply_iir_equalizer(observed, b, a)[test]
        else:
            exact_feedback, stable_feedback, inverse_zero_radius = _soe_inverse_feedback(memory)
            feedback = exact_feedback if receiver == "soe_exact" else stable_feedback
            b = design_fixed_denominator_equalizer(
                observed[train], tx[train], feedback,
                numerator_order=len(weights) + 1, ridge=1e-5,
            )
            estimate = apply_iir_equalizer(observed, b, feedback)[test]
            pole_radius = float(np.max(np.abs(np.roots(
                np.r_[1.0, -feedback]
            )))) if feedback.size else 0.0

        params, states, macs = _complexity(receiver, memory)
        rows.append({
            "experiment": "12B_SOE_structured_receiver",
            "signal": signal_name,
            "memory": memory,
            "channel": channel,
            "receiver": receiver,
            "snr_db": snr_db,
            "seed": seed,
            "channel_energy": float(np.sum(abs(h) ** 2)),
            "evm_percent": 100 * evm(tx[test], estimate),
            "ber": _ber(signal_name, tx[test], estimate),
            "heldout_mse": float(np.mean(abs(estimate - tx[test]) ** 2)),
            "parameter_count": params,
            "state_dimension": states,
            "macs_per_sample": macs,
            "effective_memory_samples": states,
            "pole_radius": pole_radius,
            "inverse_zero_radius": inverse_zero_radius,
        })
    return rows


def main():
    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for memory in SOE_MEMORIES:
                for channel in CHANNELS:
                    for snr in SNR_DB:
                        rows.extend(_run_case(signal_name, generator, memory, channel, seed, snr))

    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "12B_soe_structured_receiver_results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "12B_SOE_structured_receiver",
        "date": "2026-09-22",
        "rows": len(rows),
        "memories": list(SOE_MEMORIES),
        "channels": list(CHANNELS),
        "receivers": list(RECEIVERS),
        "snr_db": list(SNR_DB),
        "seeds": list(SEEDS),
        "evaluation": "384 held-out symbols after 128-symbol training prefix",
        "structure": "Symbol-rate SOE poles are fixed from known weights/gammas and the sampled rectangular-pulse realization; only the feedforward numerator is learned. soe_exact uses the resulting causal inverse denominator; soe_stable reflects unstable inverse poles inside the unit circle.",
        "boundary": "This tests the sampled SOE realization, not continuous MIN invertibility.",
    }
    (out / "12B_soe_structured_receiver_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
