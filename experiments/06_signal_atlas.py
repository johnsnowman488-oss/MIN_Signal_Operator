"""First MIN synthetic-signal atlas.

The atlas characterizes the transformation induced by several memory kernels
on elementary deterministic signals. It is deliberately not a task benchmark:
there is no claim that a transformed signal should be closer to the input.
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
from min.kernels import exponential_kernel
from min.signals import (
    SignalRecord,
    generate_gaussian_pulse,
    generate_impulse,
    generate_linear_chirp,
    generate_multitone,
    generate_rectangular_pulse,
    generate_sine,
)
from min.metrics import correlation, delay_of_max_correlation, energy, peak_magnitude


def build_signals(sample_rate: float = 1000.0, duration: float = 2.0) -> list[SignalRecord]:
    return [
        generate_impulse(duration, sample_rate, index=20),
        generate_rectangular_pulse(duration, sample_rate, start=0.4, width=0.2),
        generate_gaussian_pulse(duration, sample_rate, center=0.8, sigma=0.08),
        generate_sine(duration, sample_rate, frequency=5.0),
        generate_multitone(duration, sample_rate, frequencies=[5.0, 17.0], amplitudes=[1.0, 0.35]),
        generate_linear_chirp(duration, sample_rate, f0=3.0, f1=80.0),
    ]


def identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


def apply_memory(record: SignalRecord, kernel_name: str) -> np.ndarray:
    t = np.arange(record.samples.size) / record.sample_rate
    x = record.samples
    if kernel_name == "identity":
        return identity(x)
    if kernel_name == "exponential_tau_0.10":
        return MemoryOperator(lambda lag: exponential_kernel(lag, 0.10)).apply(t, x)
    if kernel_name == "soe_two_scale":
        weights = np.array([0.7, 0.3])
        gammas = np.array([2.0, 30.0])
        kernel = lambda lag: sum(
            w * np.exp(-g * np.asarray(lag))
            for w, g in zip(weights, gammas)
        )
        return MemoryOperator(kernel).apply(t, x)
    if kernel_name == "powerlaw":
        kernel = lambda lag: (1.0 + np.maximum(lag, 0.0) / 0.5) ** (-0.7)
        return MemoryOperator(kernel).apply(t, x)
    raise ValueError(f"unknown kernel: {kernel_name}")


def main() -> None:
    signals = build_signals()
    kernel_names = ["identity", "exponential_tau_0.10", "soe_two_scale", "powerlaw"]
    rows: list[dict] = []

    for signal in signals:
        x = signal.samples
        for kernel_name in kernel_names:
            y = apply_memory(signal, kernel_name)
            rows.append({
                "signal": signal.signal_type,
                "kernel": kernel_name,
                "samples": int(x.size),
                "input_energy": energy(x),
                "output_energy": energy(y),
                "energy_ratio": energy(y) / max(energy(x), np.finfo(float).eps),
                "input_peak": peak_magnitude(x),
                "output_peak": peak_magnitude(y),
                "peak_ratio": peak_magnitude(y) / max(peak_magnitude(x), np.finfo(float).eps),
                "correlation": correlation(x, y),
                "delay_samples": delay_of_max_correlation(x, y),
            })

    out_dir = ROOT / "experiments"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results" / "06_signal_atlas.json"
    csv_path = out_dir / "results" / "06_signal_atlas.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['signal']:20s} {row['kernel']:22s} "
            f"E_ratio={row['energy_ratio']:.4e} "
            f"peak_ratio={row['peak_ratio']:.4e} "
            f"corr={row['correlation']:.5f} "
            f"delay={row['delay_samples']:4d}"
        )


if __name__ == "__main__":
    main()
