"""Experiment 09: BPSK/QPSK/16-QAM through controlled MIN memory regimes.

This experiment measures how a memory transformation changes sampled
constellations. It deliberately does not claim BER improvement.
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
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
KERNELS = ("identity", "exponential_tau_0.10", "soe_two_scale", "powerlaw_alpha_0.70")


def apply_kernel(name: str, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    if name == "identity":
        return x.copy()
    if name == "exponential_tau_0.10":
        return MemoryOperator(lambda lag: exponential_kernel(lag, 0.10)).apply(t, x)
    if name == "soe_two_scale":
        w = np.array([0.7, 0.3]); g = np.array([2.0, 30.0])
        return MemoryOperator(lambda lag: sum(wi*np.exp(-gi*np.asarray(lag)) for wi,gi in zip(w,g))).apply(t,x)
    if name == "powerlaw_alpha_0.70":
        return MemoryOperator(lambda lag: power_law_kernel(lag, alpha=0.70, tau=0.50)).apply(t,x)
    raise ValueError(name)


def complex_gain_fit(reference: np.ndarray, observed: np.ndarray) -> complex:
    return np.vdot(reference, observed) / np.vdot(reference, reference)


def evm_after_gain(reference: np.ndarray, observed: np.ndarray) -> float:
    gain = complex_gain_fit(reference, observed)
    residual = observed - gain * reference
    return float(np.sqrt(np.mean(np.abs(residual)**2) / np.mean(np.abs(gain*reference)**2)))


def run_one(signal_name: str, generator, seed: int, sps: int = 16) -> list[dict]:
    signal = generator(512, samples_per_symbol=sps, symbol_rate=100.0, seed=seed)
    x = signal.record.samples
    t = np.arange(x.size) / signal.record.sample_rate
    rows = []
    for kernel_name in KERNELS:
        y = apply_kernel(kernel_name, t, x)
        observed = y[signal.symbol_indices]
        gain = complex_gain_fit(signal.symbols, observed)
        rows.append({
            "signal": signal_name, "kernel": kernel_name, "seed": seed,
            "num_symbols": 512, "samples_per_symbol": sps,
            "gain_real": float(gain.real), "gain_imag": float(gain.imag),
            "gain_magnitude": float(abs(gain)),
            "gain_phase_deg": float(np.degrees(np.angle(gain))),
            "evm_percent": float(100.0 * evm_after_gain(signal.symbols, observed)),
        })
    return rows


def main() -> None:
    rows = []
    for seed in range(5):
        for name, generator in SIGNALS.items():
            rows.extend(run_one(name, generator, seed))
    out = ROOT / "experiments" / "results"; out.mkdir(parents=True, exist_ok=True)
    with (out / "09_digital_memory_transform_results.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = {}
    for signal in SIGNALS:
        summary[signal] = {}
        for kernel in KERNELS:
            vals = [r["evm_percent"] for r in rows if r["signal"] == signal and r["kernel"] == kernel]
            summary[signal][kernel] = {"mean_evm_percent":float(np.mean(vals)),
                "std_evm_percent":float(np.std(vals,ddof=1)),
                "min_evm_percent":float(np.min(vals)), "max_evm_percent":float(np.max(vals))}
    (out / "09_digital_memory_transform_summary.json").write_text(json.dumps({
        "experiment":"09_MIN_Digital_Memory_Transform","date":"2026-09-21",
        "seeds":list(range(5)),"signals":list(SIGNALS),"kernels":list(KERNELS),
        "metric":"EVM after complex scalar gain fitting",
        "note":"No noise, channel, equalizer, or BER claim is included.",
        "summary":summary},indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
