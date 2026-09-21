"""Experiment 08: finite-horizon memory scaling.

Measure how quickly the finite-history response H_T(omega) approaches the
infinite-horizon response H_inf(omega) as the observation horizon T grows.

The experiment is deliberately frequency-resolved: a long-memory kernel can
have a substantial finite-horizon effect even when its nonzero-frequency
infinite-horizon transform exists.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from min import exponential_kernel, power_law_kernel

ROOT = Path(__file__).resolve().parents[1]
DT = 0.002
HORIZONS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
OMEGAS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kind: str
    tau: float = 0.5
    alpha: float = 0.7


KERNELS = [
    KernelSpec("exponential_tau_0.10", "exponential", tau=0.10),
    KernelSpec("powerlaw_alpha_0.50_tau_0.50", "powerlaw", tau=0.50, alpha=0.50),
    KernelSpec("powerlaw_alpha_0.70_tau_0.50", "powerlaw", tau=0.50, alpha=0.70),
    KernelSpec("powerlaw_alpha_0.90_tau_0.50", "powerlaw", tau=0.50, alpha=0.90),
]


def kernel_value(spec: KernelSpec, t: np.ndarray) -> np.ndarray:
    if spec.kind == "exponential":
        return exponential_kernel(t, spec.tau)
    if spec.kind == "powerlaw":
        return power_law_kernel(t, alpha=spec.alpha, tau=spec.tau)
    raise ValueError(spec.kind)


def h_infinite(spec: KernelSpec, omega: float) -> complex:
    if spec.kind == "exponential":
        return spec.tau / (1.0 + 1j * omega * spec.tau)

    envelope = lambda s: float((1.0 + s / spec.tau) ** (-spec.alpha))
    real = quad(
        envelope, 0.0, np.inf, weight="cos", wvar=omega,
        epsabs=1e-9, epsrel=1e-9, limit=600,
    )[0]
    imag = -quad(
        envelope, 0.0, np.inf, weight="sin", wvar=omega,
        epsabs=1e-9, epsrel=1e-9, limit=600,
    )[0]
    return complex(real, imag)


def h_finite(spec: KernelSpec, omega: float, horizon: float) -> complex:
    if spec.kind == "exponential":
        # Exact finite-window transform removes a numerical-discretization
        # floor from the horizon-scaling measurement.
        a = 1.0 / spec.tau + 1j * omega
        return complex((1.0 - np.exp(-a * horizon)) / a)

    envelope = lambda s: float((1.0 + s / spec.tau) ** (-spec.alpha))
    real = quad(
        envelope, 0.0, horizon, weight="cos", wvar=omega,
        epsabs=1e-10, epsrel=1e-10, limit=500,
    )[0]
    imag = -quad(
        envelope, 0.0, horizon, weight="sin", wvar=omega,
        epsabs=1e-10, epsrel=1e-10, limit=500,
    )[0]
    return complex(real, imag)


def phase_error_deg(a: complex, b: complex) -> float:
    d = np.angle(a) - np.angle(b)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(d))


def rows_for_kernel(spec: KernelSpec) -> list[dict]:
    rows: list[dict] = []
    for omega_value in OMEGAS:
        omega = float(omega_value)
        h_inf = h_infinite(spec, omega)
        denom = max(abs(h_inf), np.finfo(float).eps)
        for horizon_value in HORIZONS:
            horizon = float(horizon_value)
            h_t = h_finite(spec, omega, horizon)
            error = abs(h_t - h_inf) / denom
            rows.append(
                {
                    "kernel": spec.name,
                    "kind": spec.kind,
                    "alpha": spec.alpha if spec.kind == "powerlaw" else np.nan,
                    "tau_s": spec.tau,
                    "omega_rad_s": omega,
                    "horizon_s": horizon,
                    "H_inf_magnitude": float(abs(h_inf)),
                    "H_inf_phase_rad": float(np.angle(h_inf)),
                    "H_T_magnitude": float(abs(h_t)),
                    "H_T_phase_rad": float(np.angle(h_t)),
                    "relative_complex_error": float(error),
                    "magnitude_relative_error": float(abs(abs(h_t) - abs(h_inf)) / denom),
                    "phase_error_deg": phase_error_deg(h_t, h_inf),
                    "relative_tail_error": float(error),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for spec in KERNELS:
        subset = [r for r in rows if r["kernel"] == spec.name]
        thresholds: dict[str, dict] = {}
        for omega in OMEGAS:
            at_freq = sorted(
                [r for r in subset if r["omega_rad_s"] == float(omega)],
                key=lambda r: r["horizon_s"],
            )
            for threshold in (0.10, 0.05, 0.01):
                hits = [
                    r["horizon_s"] for r in at_freq
                    if r["relative_complex_error"] <= threshold
                ]
                thresholds.setdefault(str(float(omega)), {})[
                    f"T_for_{int(threshold * 100)}pct_or_better_s"
                ] = min(hits) if hits else None

        summary[spec.name] = {
            "points": len(subset),
            "max_error_at_shortest_horizon": float(
                max(r["relative_complex_error"] for r in subset if r["horizon_s"] == HORIZONS[0])
            ),
            "median_error_at_8s": float(
                np.median([r["relative_complex_error"] for r in subset if r["horizon_s"] == 8.0])
            ),
            "max_error_at_64s": float(
                max(r["relative_complex_error"] for r in subset if r["horizon_s"] == 64.0)
            ),
            "frequency_specific_horizon_thresholds": thresholds,
        }
    return summary


def main() -> None:
    rows: list[dict] = []
    for spec in KERNELS:
        rows.extend(rows_for_kernel(spec))

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "08_finite_horizon_memory_scaling_results.csv"
    summary_path = out_dir / "08_finite_horizon_memory_scaling_summary.json"

    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "08_MIN_Finite_Horizon_Memory_Scaling",
                "date": "2026-09-21",
                "settings": {
                    "dt_s": DT,
                    "horizons_s": HORIZONS.tolist(),
                    "omega_rad_s": OMEGAS.tolist(),
                    "error_definition": "|H_T-H_inf|/|H_inf|",
                    "powerlaw_dc_note": "alpha <= 1 has divergent DC integral; only nonzero-frequency oscillatory responses are compared.",
                    "finite_window_method": "exact for exponential; weighted finite-interval oscillatory quadrature for power-law",
                },
                "summary": summary,
            },
            indent=2,
        ) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
