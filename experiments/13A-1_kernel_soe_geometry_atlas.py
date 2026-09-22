"""Experiment 13A-1: controlled Kernel-SOE Geometry Atlas.

This first 13A variant studies the forward temporal representation created by
finite sums of exponential memory modes. It deliberately excludes channels,
noise, receivers, and downstream task models.

For each kernel
    K(t) = sum_j a_j exp(-gamma_j t)
we construct causal exponential states
    q_j(t) = integral_0^t exp(-gamma_j(t-s)) x(s) ds
using an exact piecewise-constant recurrence over the sampled waveform.

The experiment records geometry of the resulting temporal state/feature space
and separately records scalar convolution behavior. "Effective dimension" is
used descriptively; no manifold claim is made.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.signals import generate_16qam, generate_bpsk, generate_qpsk


SIGNALS = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "16QAM": generate_16qam,
}

MODE_COUNTS = (1, 2, 4, 8, 16)
RATE_GEOMETRIES = ("clustered", "logspread", "wide")
WEIGHT_PATTERNS = ("uniform", "slow_dominant", "fast_dominant")
SEEDS = tuple(range(5))

NUM_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE


def rate_pattern(mode_count: int, geometry: str) -> np.ndarray:
    """Return ordered decay rates from slow to fast, in inverse seconds."""
    if geometry == "clustered":
        return 10.0 * np.exp(np.linspace(-0.05, 0.05, mode_count))
    if geometry == "logspread":
        return np.geomspace(2.0, 30.0, mode_count)
    if geometry == "wide":
        return np.geomspace(0.5, 100.0, mode_count)
    raise ValueError(f"unknown rate geometry: {geometry}")


def weight_pattern(mode_count: int, pattern: str) -> np.ndarray:
    """Return positive weights summing to one."""
    if mode_count == 1:
        return np.ones(1)
    if pattern == "uniform":
        return np.full(mode_count, 1.0 / mode_count)

    w = np.full(mode_count, 0.3 / (mode_count - 1))
    if pattern == "slow_dominant":
        w[0] = 0.7
    elif pattern == "fast_dominant":
        w[-1] = 0.7
    else:
        raise ValueError(f"unknown weight pattern: {pattern}")
    return w


def exponential_states(x: np.ndarray, gammas: np.ndarray, dt: float) -> np.ndarray:
    """Sample q_j(t)=integral exp(-gamma_j(t-s))x(s) ds.

    With piecewise-constant x over each sample interval:
        q[n] = exp(-gamma*dt) q[n-1]
             + (1-exp(-gamma*dt))/gamma * x[n].
    """
    x = np.asarray(x, dtype=complex)
    gammas = np.asarray(gammas, dtype=float)
    poles = np.exp(-gammas * dt)
    increments = -np.expm1(-gammas * dt) / gammas

    q = np.zeros((x.size, gammas.size), dtype=complex)
    state = np.zeros(gammas.size, dtype=complex)
    for n, sample in enumerate(x):
        state = poles * state + increments * sample
        q[n] = state
    return q


def scalar_output(q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return q @ weights


def impulse_metrics(gammas: np.ndarray, weights: np.ndarray, dt: float) -> dict:
    """Measure temporal spread of the aggregate causal kernel."""
    horizon = max(8192, int(np.ceil(12.0 / (np.min(gammas) * dt))))
    horizon = min(horizon, 65536)

    impulse = np.zeros(horizon, dtype=complex)
    impulse[0] = 1.0
    q = exponential_states(impulse, gammas, dt)
    h = q @ weights
    magnitude = np.abs(h)
    mass = float(np.sum(magnitude))

    if mass <= 0:
        t50 = t90 = centroid = 0.0
    else:
        cdf = np.cumsum(magnitude) / mass
        t50 = float(np.searchsorted(cdf, 0.50) * dt)
        t90 = float(np.searchsorted(cdf, 0.90) * dt)
        centroid = float(
            np.sum(np.arange(horizon) * dt * magnitude) / mass
        )

    n_fft = 16384
    h_fft = np.zeros(n_fft, dtype=complex)
    h_fft[: min(h.size, n_fft)] = h[:n_fft]
    freq = np.fft.rfftfreq(n_fft, d=dt)
    response = np.fft.fft(h_fft)[: n_fft // 2 + 1]
    mag = np.abs(response)

    return {
        "kernel_l1_sampled": mass,
        "kernel_l2_sampled": float(np.linalg.norm(h)),
        "kernel_t50_s": t50,
        "kernel_t90_s": t90,
        "kernel_centroid_s": centroid,
        "kernel_dc_gain": float(mag[0]),
        "kernel_nyquist_gain": float(mag[-1]),
        "kernel_frequency_dynamic_range": float(
            mag.max() / max(mag.min(), np.finfo(float).tiny)
        ),
        "kernel_fft_frequency_max_hz": float(freq[np.argmax(mag)]),
    }


def state_geometry(q: np.ndarray, weights: np.ndarray) -> dict:
    """Compute descriptive geometry of the complex temporal state trajectory."""
    centered = q - np.mean(q, axis=0, keepdims=True)
    gram = centered.conj().T @ centered / max(centered.shape[0] - 1, 1)
    eig = np.maximum(np.linalg.eigvalsh(gram).real, 0.0)
    eig = np.sort(eig)[::-1]

    total = float(np.sum(eig))
    participation = float(
        total**2 / max(np.sum(eig**2), np.finfo(float).tiny)
    )

    if total > 0:
        cumulative = np.cumsum(eig) / total
        rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
        rank99 = int(np.searchsorted(cumulative, 0.99) + 1)
    else:
        rank90 = rank99 = 0

    normalized = np.sqrt(np.maximum(np.real(np.diag(gram)), 0.0))
    corr = gram / np.outer(
        np.maximum(normalized, np.finfo(float).tiny),
        np.maximum(normalized, np.finfo(float).tiny),
    )
    offdiag = np.abs(corr - np.diag(np.diag(corr)))
    max_collinearity = float(np.max(offdiag)) if offdiag.size else 0.0

    trajectory_length = float(
        np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))
    )
    radius = float(
        np.sqrt(np.mean(np.sum(np.abs(centered) ** 2, axis=1)))
    )
    component_energy = np.mean(np.abs(q) ** 2, axis=0)

    qw = q * np.sqrt(weights)[None, :]
    cw = qw - np.mean(qw, axis=0, keepdims=True)
    gw = cw.conj().T @ cw / max(cw.shape[0] - 1, 1)
    ew = np.maximum(np.linalg.eigvalsh(gw).real, 0.0)
    ew = np.sort(ew)[::-1]
    tw = float(np.sum(ew))
    pw = float(tw**2 / max(np.sum(ew**2), np.finfo(float).tiny))
    cw_cum = np.cumsum(ew) / max(tw, np.finfo(float).tiny)
    rw99 = int(np.searchsorted(cw_cum, 0.99) + 1) if tw > 0 else 0

    return {
        "state_energy": float(np.sum(component_energy)),
        "state_rms_radius": radius,
        "state_trajectory_length": trajectory_length,
        "state_participation_dimension": participation,
        "state_rank_90": rank90,
        "state_rank_99": rank99,
        "state_max_component_collinearity": max_collinearity,
        "state_weighted_participation_dimension": pw,
        "state_weighted_rank_99": rw99,
        "state_max_component_energy": float(np.max(component_energy)),
        "state_min_component_energy": float(np.min(component_energy)),
    }


def scalar_geometry(y: np.ndarray) -> dict:
    centered = y - np.mean(y)
    rms = float(np.sqrt(np.mean(np.abs(centered) ** 2)))
    length = float(np.sum(np.abs(np.diff(y))))
    energy = float(np.mean(np.abs(y) ** 2))
    return {
        "scalar_output_energy": energy,
        "scalar_output_rms": rms,
        "scalar_output_trajectory_length": length,
        "scalar_output_effective_dimension": 1.0 if energy > 0 else 0.0,
    }


def run_case(
    signal_name: str,
    generator,
    mode_count: int,
    geometry: str,
    weight_name: str,
    seed: int,
) -> dict:
    signal = generator(
        NUM_SYMBOLS,
        samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE,
        seed=seed,
    )
    x = np.asarray(signal.record.samples, dtype=complex)

    gammas = rate_pattern(mode_count, geometry)
    weights = weight_pattern(mode_count, weight_name)
    q = exponential_states(x, gammas, DT)
    y = scalar_output(q, weights)

    row = {
        "experiment": "13A-1_kernel_soe_geometry_atlas",
        "signal": signal_name,
        "seed": seed,
        "mode_count": mode_count,
        "rate_geometry": geometry,
        "weight_pattern": weight_name,
        "gamma_min": float(np.min(gammas)),
        "gamma_max": float(np.max(gammas)),
        "gamma_geometric_span": float(np.max(gammas) / np.min(gammas)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "sample_rate_hz": SAMPLE_RATE,
        "dt_s": DT,
    }
    row.update(impulse_metrics(gammas, weights, DT))
    row.update(state_geometry(q, weights))
    row.update(scalar_geometry(y))
    return row


def main() -> None:
    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for mode_count in MODE_COUNTS:
                for geometry in RATE_GEOMETRIES:
                    for weight_name in WEIGHT_PATTERNS:
                        rows.append(
                            run_case(
                                signal_name,
                                generator,
                                mode_count,
                                geometry,
                                weight_name,
                                seed,
                            )
                        )

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "13A-1_kernel_soe_geometry_atlas_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    numeric = [
        "state_participation_dimension",
        "state_weighted_participation_dimension",
        "state_rank_99",
        "state_weighted_rank_99",
        "state_max_component_collinearity",
        "state_rms_radius",
        "state_trajectory_length",
        "kernel_t90_s",
        "kernel_centroid_s",
        "scalar_output_energy",
    ]
    aggregate = {}
    for row in rows:
        key = (row["mode_count"], row["rate_geometry"], row["weight_pattern"])
        bucket = aggregate.setdefault(key, {name: [] for name in numeric})
        for name in numeric:
            bucket[name].append(float(row[name]))

    summary_rows = []
    for key, bucket in sorted(aggregate.items(), key=lambda item: item[0]):
        mode_count, geometry, weight_name = key
        summary_rows.append(
            {
                "mode_count": mode_count,
                "rate_geometry": geometry,
                "weight_pattern": weight_name,
                **{
                    f"{name}_mean": float(np.mean(values))
                    for name, values in bucket.items()
                },
                **{
                    f"{name}_std": float(np.std(values, ddof=1))
                    for name, values in bucket.items()
                },
            }
        )

    summary = {
        "experiment": "13A-1_kernel_soe_geometry_atlas",
        "variant": "controlled SOE mode/rate/weight geometry",
        "rows": len(rows),
        "grid": {
            "signals": list(SIGNALS),
            "mode_counts": list(MODE_COUNTS),
            "rate_geometries": list(RATE_GEOMETRIES),
            "weight_patterns": list(WEIGHT_PATTERNS),
            "seeds": list(SEEDS),
        },
        "total_kernel_conditions": (
            len(MODE_COUNTS) * len(RATE_GEOMETRIES) * len(WEIGHT_PATTERNS)
        ),
        "definitions": {
            "state": "q_j(t)=integral exp(-gamma_j(t-s))*x(s) ds",
            "weighted_state": "sqrt(a_j)*q_j used only as an explicit alternative coordinate metric",
            "effective_dimension": "participation ratio of centered complex state covariance",
            "rank_99": "smallest covariance rank explaining at least 99 percent of variance",
            "collinearity": "maximum absolute off-diagonal normalized complex state covariance",
            "scalar_output": "sum_j a_j q_j",
        },
        "interpretation_boundary": [
            "This variant characterizes forward temporal representation only.",
            "It does not test classification, prediction, propagation robustness, or communication advantage.",
            "Effective dimension is descriptive and does not establish a manifold.",
            "Weighted and unweighted state metrics are both retained because coordinate scaling changes Euclidean geometry.",
        ],
        "primary_metrics": numeric,
    }
    (out / "13A-1_kernel_soe_geometry_atlas_summary.json").write_text(
        json.dumps(summary, indent=2) + "\\n"
    )

    summary_csv = out / "13A-1_kernel_soe_geometry_atlas_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps(summary, indent=2))
    for row in summary_rows:
        if (
            row["mode_count"] in (1, 4, 16)
            and row["rate_geometry"] == "logspread"
            and row["weight_pattern"] == "uniform"
        ):
            print(
                "atlas",
                row["mode_count"],
                row["rate_geometry"],
                row["weight_pattern"],
                "participation_mean=",
                row["state_participation_dimension_mean"],
                "rank99_mean=",
                row["state_rank_99_mean"],
                "t90_s=",
                row["kernel_t90_s_mean"],
                "collinearity_mean=",
                row["state_max_component_collinearity_mean"],
            )


if __name__ == "__main__":
    main()

# Workflow trigger: validated atlas implementation is ready for CI execution.
