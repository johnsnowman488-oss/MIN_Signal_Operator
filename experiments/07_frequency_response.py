"""Experiment 07: MIN frequency-response characterization.

The experiment compares three levels of description:
1. a reference frequency response H_ref,
2. a finite-observation kernel transform H_window,
3. causal estimates from MIN{exp(j*omega*t)}:
   - H_endpoint at the observation endpoint,
   - H_tail from demodulation over the final observation interval.

This separates operator discretization, finite-horizon truncation, and residual
startup/nonstationarity effects. The power-law kernel is treated specially:
for alpha=0.70 its DC integral diverges, while nonzero-frequency response is
computed with weighted oscillatory quadrature.
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

DT = 0.001
T = 12.0
OMEGAS = np.geomspace(0.5, 40.0, 12)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kind: str
    tail_start: float


KERNELS = [
    KernelSpec("identity", "identity", 0.0),
    KernelSpec("exponential_tau_0.10", "exponential", 1.0),
    KernelSpec("soe_two_scale", "soe", 4.0),
    KernelSpec("powerlaw_alpha_0.70_tau_0.50", "powerlaw", 6.0),
]


def kernel_value(spec: KernelSpec, t: np.ndarray) -> np.ndarray:
    if spec.kind == "identity":
        raise ValueError("identity has no ordinary sampled kernel")
    if spec.kind == "exponential":
        return exponential_kernel(t, 0.10)
    if spec.kind == "soe":
        return 0.7 * np.exp(-2.0 * np.maximum(t, 0.0)) + 0.3 * np.exp(
            -30.0 * np.maximum(t, 0.0)
        )
    if spec.kind == "powerlaw":
        return power_law_kernel(t, alpha=0.70, tau=0.50)
    raise ValueError(spec.kind)


def analytic_reference(spec: KernelSpec, omega: float) -> complex | None:
    if spec.kind == "identity":
        return 1.0 + 0.0j
    if spec.kind == "exponential":
        tau = 0.10
        return tau / (1.0 + 1j * omega * tau)
    if spec.kind == "soe":
        return 0.7 / (2.0 + 1j * omega) + 0.3 / (30.0 + 1j * omega)
    return None


def frequency_response_reference(spec: KernelSpec, omega: float) -> tuple[complex, str]:
    analytic = analytic_reference(spec, omega)
    if analytic is not None:
        return analytic, "analytic"
    envelope = lambda s: float((1.0 + s / 0.50) ** (-0.70))
    real = quad(
        envelope,
        0.0,
        np.inf,
        weight="cos",
        wvar=omega,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500,
    )[0]
    imag = -quad(
        envelope,
        0.0,
        np.inf,
        weight="sin",
        wvar=omega,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500,
    )[0]
    return complex(real, imag), "scipy_weighted_quadrature"


def trapezoid_transform(k: np.ndarray, t: np.ndarray, omega: float) -> complex:
    return complex(np.trapezoid(k * np.exp(-1j * omega * t), t))


def apply_uniform_trapezoid_kernel(k: np.ndarray, x: np.ndarray, dt: float) -> np.ndarray:
    """Discrete causal convolution matching the trapezoid MIN reference rule."""
    y = np.convolve(k, x)[: x.size] * dt
    if x.size:
        y -= 0.5 * dt * k[: x.size] * x[0]
        y -= 0.5 * dt * k[0] * x
        y[0] = 0.0
    return y


def direct_min(spec: KernelSpec, t: np.ndarray, omega: float) -> np.ndarray:
    x = np.exp(1j * omega * t)
    if spec.kind == "identity":
        return x
    return apply_uniform_trapezoid_kernel(kernel_value(spec, t), x, t[1] - t[0])


def wrap_phase(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def build_rows() -> tuple[list[dict], dict]:
    t = np.arange(0.0, T + DT / 2.0, DT)
    rows: list[dict] = []

    for spec in KERNELS:
        for omega_value in OMEGAS:
            omega = float(omega_value)
            h_ref, reference_kind = frequency_response_reference(spec, omega)
            if spec.kind == "identity":
                h_window = h_ref
            else:
                h_window = trapezoid_transform(kernel_value(spec, t), t, omega)

            y = direct_min(spec, t, omega)
            tail_mask = t >= spec.tail_start
            x_tail = np.exp(1j * omega * t[tail_mask])
            y_tail = y[tail_mask]
            demod = y_tail * np.exp(-1j * omega * t[tail_mask])
            h_tail = complex(np.mean(demod))
            h_endpoint = complex(y[-1] * np.exp(-1j * omega * t[-1]))

            ref_mag = max(abs(h_ref), np.finfo(float).eps)
            window_mag = max(abs(h_window), np.finfo(float).eps)
            tail_model = h_ref * x_tail
            tail_model_norm = max(np.linalg.norm(tail_model), np.finfo(float).eps)

            rows.append(
                {
                    "kernel": spec.name,
                    "omega_rad_s": omega,
                    "dt_s": DT,
                    "T_s": T,
                    "tail_start_s": spec.tail_start,
                    "reference_kind": reference_kind,
                    "dc_behavior": "finite" if spec.kind in {"identity", "exponential", "soe"} else "diverges",
                    "H_ref_real": float(np.real(h_ref)),
                    "H_ref_imag": float(np.imag(h_ref)),
                    "H_ref_magnitude": float(abs(h_ref)),
                    "H_ref_phase_rad": float(np.angle(h_ref)),
                    "H_window_real": float(np.real(h_window)),
                    "H_window_imag": float(np.imag(h_window)),
                    "H_window_magnitude": float(abs(h_window)),
                    "H_window_phase_rad": float(np.angle(h_window)),
                    "H_endpoint_real": float(np.real(h_endpoint)),
                    "H_endpoint_imag": float(np.imag(h_endpoint)),
                    "H_endpoint_magnitude": float(abs(h_endpoint)),
                    "H_endpoint_phase_rad": float(np.angle(h_endpoint)),
                    "H_tail_real": float(np.real(h_tail)),
                    "H_tail_imag": float(np.imag(h_tail)),
                    "H_tail_magnitude": float(abs(h_tail)),
                    "H_tail_phase_rad": float(np.angle(h_tail)),
                    "operator_vs_window_rel_error": float(abs(h_endpoint - h_window) / window_mag),
                    "tail_estimator_vs_window_rel_error": float(abs(h_tail - h_window) / window_mag),
                    "window_vs_reference_rel_error": float(abs(h_window - h_ref) / ref_mag),
                    "operator_vs_reference_rel_error": float(abs(h_endpoint - h_ref) / ref_mag),
                    "tail_vs_reference_rel_error": float(abs(h_tail - h_ref) / ref_mag),
                    "tail_demod_cv": float(np.std(np.abs(demod)) / max(abs(h_tail), np.finfo(float).eps)),
                    "tail_model_residual_rel": float(np.linalg.norm(y_tail - tail_model) / tail_model_norm),
                    "endpoint_phase_error_deg": float(np.degrees(wrap_phase(np.angle(h_endpoint) - np.angle(h_ref)))),
                    "tail_phase_error_deg": float(np.degrees(wrap_phase(np.angle(h_tail) - np.angle(h_ref)))),
                }
            )

    for spec in KERNELS:
        subset = [row for row in rows if row["kernel"] == spec.name]
        subset.sort(key=lambda row: row["omega_rad_s"])
        omega = np.array([row["omega_rad_s"] for row in subset])
        ref_phase = np.unwrap([row["H_ref_phase_rad"] for row in subset])
        tail_phase = np.unwrap([row["H_tail_phase_rad"] for row in subset])
        ref_gd = -np.gradient(ref_phase, omega)
        tail_gd = -np.gradient(tail_phase, omega)
        for row, ref_delay, tail_delay in zip(subset, ref_gd, tail_gd):
            row["reference_group_delay_s"] = float(ref_delay)
            row["tail_estimated_group_delay_s"] = float(tail_delay)

    summary = {}
    for spec in KERNELS:
        subset = [row for row in rows if row["kernel"] == spec.name]
        summary[spec.name] = {
            "points": len(subset),
            "median_operator_vs_reference_rel_error": float(np.median([row["operator_vs_reference_rel_error"] for row in subset])),
            "max_operator_vs_reference_rel_error": float(np.max([row["operator_vs_reference_rel_error"] for row in subset])),
            "max_window_vs_reference_rel_error": float(np.max([row["window_vs_reference_rel_error"] for row in subset])),
            "max_tail_model_residual_rel": float(np.max([row["tail_model_residual_rel"] for row in subset])),
            "max_operator_vs_window_rel_error": float(np.max([row["operator_vs_window_rel_error"] for row in subset])),
            "median_tail_estimator_vs_window_rel_error": float(np.median([row["tail_estimator_vs_window_rel_error"] for row in subset])),
            "note": "Power-law uses weighted oscillatory quadrature at nonzero frequency; its DC gain diverges for alpha=0.70 because the kernel is not integrable.",
        }
    return rows, summary


def main() -> None:
    rows, summary = build_rows()
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "07_frequency_response_results.csv"
    summary_path = out_dir / "07_frequency_response_summary.json"

    result_fields = [
        "kernel",
        "omega_rad_s",
        "H_ref_magnitude",
        "H_ref_phase_rad",
        "H_window_magnitude",
        "H_endpoint_magnitude",
        "H_tail_magnitude",
        "operator_vs_window_rel_error",
        "tail_estimator_vs_window_rel_error",
        "window_vs_reference_rel_error",
        "operator_vs_reference_rel_error",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in result_fields})

    diagnostics = {}
    for spec in KERNELS:
        subset = sorted(
            (row for row in rows if row["kernel"] == spec.name),
            key=lambda row: row["omega_rad_s"],
        )
        selected = [subset[0], subset[len(subset) // 2], subset[-1]]
        diagnostics[spec.name] = [
            {
                "omega_rad_s": row["omega_rad_s"],
                "window_vs_reference_rel_error": row["window_vs_reference_rel_error"],
                "tail_estimator_vs_window_rel_error": row["tail_estimator_vs_window_rel_error"],
                "operator_vs_reference_rel_error": row["operator_vs_reference_rel_error"],
            }
            for row in selected
        ]

    summary_path.write_text(
        json.dumps(
            {
                "experiment": "07_MIN_Frequency_Response",
                "date": "2026-09-21",
                "settings": {
                    "dt_s": DT,
                    "T_s": T,
                    "omega_rad_s": OMEGAS.tolist(),
                    "reference_method": "analytic for identity/exponential/SOE; scipy weighted oscillatory quadrature for power-law",
                    "dc_behavior": {
                        "identity": "finite",
                        "exponential_tau_0.10": "finite, gain=0.10",
                        "soe_two_scale": "finite, gain=0.365",
                        "powerlaw_alpha_0.70_tau_0.50": "diverges",
                    },
                },
                "summary": summary,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
