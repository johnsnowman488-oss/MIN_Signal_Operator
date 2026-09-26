"""Experiment 14C: controlled SOE geometry -> task relevance.

The task environment and noisy observation are held fixed while three 16-mode
positive SOE/MIN rate geometries from 13A/13B are applied to the same observed
digital waveform: clustered, logspread, and wide.

For each geometry, the full state is evaluated through complex PCA dimensions
1..16 and the scalar MIN projection z = w^T q is evaluated separately.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min import SOEMemory
from min.signals import generate_16qam, generate_bpsk, generate_qpsk


SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
ENVIRONMENTS = ("white_limit", "short", "multiscale", "powerlaw", "squared_exp")
GEOMETRIES = ("clustered", "logspread", "wide")
SEEDS = tuple(range(5))
SNR_DB = (0.0, 10.0, 20.0, 30.0)

NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE
MODES = 16
PCA_DIMENSIONS = tuple(range(1, MODES + 1))
TASK_TOLERANCE = 1.10
READOUT_RIDGE = 1e-10
ENV_NOISE_SEED_OFFSET = 100_003


def environment_envelope(t: np.ndarray, name: str) -> np.ndarray:
    if name == "white_limit":
        return np.exp(-t / 0.005)
    if name == "short":
        return np.exp(-t / 0.05)
    if name == "multiscale":
        return 0.65 * np.exp(-t / 0.03) + 0.35 * np.exp(-t / 0.7)
    if name == "powerlaw":
        return (1.0 + t / 0.2) ** (-0.7)
    if name == "squared_exp":
        return np.exp(-(t / 0.15) ** 2)
    raise ValueError(name)


def rate_pattern(geometry: str) -> np.ndarray:
    if geometry == "clustered":
        return 10.0 * np.exp(np.linspace(-0.05, 0.05, MODES))
    if geometry == "logspread":
        return np.geomspace(2.0, 30.0, MODES)
    if geometry == "wide":
        return np.geomspace(0.5, 100.0, MODES)
    raise ValueError(geometry)


def uniform_weights() -> np.ndarray:
    return np.full(MODES, 1.0 / MODES)


def synth_environment_noise(environment: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * DT
    c = environment_envelope(t, environment)
    circ = np.r_[c, c[-2:0:-1]]
    spectrum = np.maximum(np.real(np.fft.rfft(circ)), 0.0)
    z = rng.normal(size=spectrum.size) + 1j * rng.normal(size=spectrum.size)
    z[0] = rng.normal()
    if circ.size % 2 == 0:
        z[-1] = rng.normal()
    noise = np.fft.irfft(np.sqrt(spectrum) * z, n=circ.size)[:n]
    return noise / max(float(np.std(noise)), np.finfo(float).tiny)


def state_coefficients(gammas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decay = np.exp(-gammas * DT)
    b_curr = 1.0 / gammas - (1.0 - decay) / (DT * gammas**2)
    b_prev = (1.0 - decay) / gammas - b_curr
    return decay, b_curr, b_prev


KERNELS = {}
for _geometry in GEOMETRIES:
    _gammas = rate_pattern(_geometry)
    _weights = uniform_weights()
    KERNELS[_geometry] = {
        "gammas": _gammas,
        "weights": _weights,
        "coefficients": state_coefficients(_gammas),
    }


def exact_piecewise_linear_states(
    x: np.ndarray,
    gammas: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Fast uniform-grid recurrence matching repository SOEMemory exactly."""
    samples = np.asarray(x, dtype=complex)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("x must be one-dimensional with at least two samples")
    decay, b_curr, b_prev = coefficients or state_coefficients(gammas)
    out = np.empty((samples.size, gammas.size), dtype=complex)
    for j, (a, bc, bp) in enumerate(zip(decay, b_curr, b_prev)):
        forcing = np.empty(samples.size, dtype=complex)
        forcing[0] = 0.0
        forcing[1:] = bc * samples[1:] + bp * samples[:-1]
        out[:, j] = lfilter([1.0], [1.0, -a], forcing)
    return out


def validate_repository_equivalence() -> float:
    rng = np.random.default_rng(1401)
    x = rng.normal(size=257) + 1j * rng.normal(size=257)
    spec = KERNELS["wide"]
    fast = exact_piecewise_linear_states(x, spec["gammas"], spec["coefficients"])
    reference = SOEMemory(spec["weights"], spec["gammas"]).state_trajectory(
        np.arange(x.size, dtype=float) * DT, x
    )
    rel = float(np.linalg.norm(fast - reference) / max(
        np.linalg.norm(reference), np.finfo(float).tiny
    ))
    if rel > 1e-9:
        raise RuntimeError(f"repository state equivalence failed: {rel:.3e}")
    return rel


def linear_readout_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    gram.flat[:: gram.shape[0] + 1] += READOUT_RIDGE
    return np.linalg.solve(gram, design.conj().T @ y)


def linear_readout_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) ** 2) /
                 max(np.mean(np.abs(y_true) ** 2), np.finfo(float).tiny))


def complex_pca_fit(X: np.ndarray):
    mean = np.mean(X, axis=0)
    _, singular, vh = np.linalg.svd(X - mean, full_matrices=False)
    return mean, singular, vh.conj().T


def pca_project(X, mean, components, k):
    return (X - mean) @ components[:, :k]


def pca_energy(singular: np.ndarray, k: int) -> float:
    e = singular**2
    return float(np.sum(e[:k]) / max(np.sum(e), np.finfo(float).tiny))


def spectrum_geometry(X: np.ndarray) -> dict:
    centered = X - np.mean(X, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(X.shape[0] - 1, 1)
    eig = np.sort(np.maximum(np.real(np.linalg.eigvalsh(cov)), 0.0))[::-1]
    total = float(eig.sum())
    if total <= 0:
        return {"center_state_participation_dimension": 0.0,
                "center_state_entropy_dimension": 0.0,
                "center_state_rank90": 0, "center_state_rank99": 0}
    p = eig / total
    return {
        "center_state_participation_dimension": float(1.0 / np.sum(p**2)),
        "center_state_entropy_dimension": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        "center_state_rank90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "center_state_rank99": int(np.searchsorted(np.cumsum(p), 0.99) + 1),
    }


def kernel_descriptors(gammas: np.ndarray, weights: np.ndarray) -> dict:
    w = weights / float(weights.sum())
    h = -float(np.sum(w * np.log(w)))
    scale = float(math.log10(gammas.max() / gammas.min()))
    t = np.linspace(0.0, 0.5, 4000)
    h_t = np.exp(-np.outer(t, gammas)) @ w
    cdf = np.cumsum(np.abs(h_t))
    cdf /= max(float(cdf[-1]), np.finfo(float).tiny)
    return {
        "dictionary_mode_count": int(len(gammas)),
        "weight_entropy_nats": h,
        "gfe_entropy_effective_count": float(np.exp(h)),
        "gfe_d_eff": float(len(gammas) * np.exp(h)),
        "gfe_m_scale_decades": scale,
        "gfe_m_res_modes_per_decade": float(len(gammas) / scale),
        "gfe_m_cap_s": float(np.sum(w / gammas**2) / np.sum(w / gammas)),
        "kernel_t50_s": float(t[np.searchsorted(cdf, 0.50)]),
        "kernel_t90_s": float(t[np.searchsorted(cdf, 0.90)]),
    }


KERNEL_DESCRIPTORS = {
    g: kernel_descriptors(KERNELS[g]["gammas"], KERNELS[g]["weights"])
    for g in GEOMETRIES
}


def run_case(signal_name, generator, environment, snr_db, seed):
    train = generator(NUM_TRAIN_SYMBOLS, samples_per_symbol=SPS,
                      symbol_rate=SYMBOL_RATE, seed=seed, normalize=False)
    test = generator(NUM_TEST_SYMBOLS, samples_per_symbol=SPS,
                     symbol_rate=SYMBOL_RATE, seed=seed + 10_000, normalize=False)
    x_train = np.asarray(train.record.samples, dtype=complex)
    x_test = np.asarray(test.record.samples, dtype=complex)
    signal_power = float(np.mean(np.abs(x_train) ** 2))
    scale = math.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    y_train = x_train + scale * synth_environment_noise(
        environment, len(x_train), seed + ENV_NOISE_SEED_OFFSET
    )
    y_test = x_test + scale * synth_environment_noise(
        environment, len(x_test), seed + ENV_NOISE_SEED_OFFSET + 50_000
    )

    centers_train = np.arange(NUM_TRAIN_SYMBOLS) * SPS + SPS // 2
    centers_test = np.arange(NUM_TEST_SYMBOLS) * SPS + SPS // 2
    raw_train = y_train[centers_train]
    raw_test = y_test[centers_test]
    raw_beta = linear_readout_fit(raw_train[:, None], train.symbols)
    raw_nmse = nmse(test.symbols, linear_readout_predict(raw_test[:, None], raw_beta))

    base = {
        "experiment": "14C_geometry_to_task",
        "signal": signal_name, "environment": environment,
        "snr_db": float(snr_db), "seed": int(seed),
        "raw_center_sample_nmse": raw_nmse,
    }
    curve_rows, scalar_rows = [], []
    case = dict(base)

    for geometry in GEOMETRIES:
        spec = KERNELS[geometry]
        q_train = exact_piecewise_linear_states(
            y_train, spec["gammas"], spec["coefficients"]
        )
        q_test = exact_piecewise_linear_states(
            y_test, spec["gammas"], spec["coefficients"]
        )
        X_train, X_test = q_train[centers_train], q_test[centers_test]
        mean, singular, components = complex_pca_fit(X_train)
        curve = {}
        for k in PCA_DIMENSIONS:
            beta = linear_readout_fit(
                pca_project(X_train, mean, components, k), train.symbols
            )
            value = nmse(
                test.symbols,
                linear_readout_predict(
                    pca_project(X_test, mean, components, k), beta
                ),
            )
            curve[k] = value
            curve_rows.append({
                **base, "geometry": geometry,
                "retained_pca_dimension": int(k),
                "heldout_symbol_nmse": value,
                "pca_explained_energy": pca_energy(singular, k),
                "full_state_reference": int(k == MODES),
                **KERNEL_DESCRIPTORS[geometry],
            })

        full_nmse = curve[MODES]
        threshold = TASK_TOLERANCE * full_nmse
        d_task = next((k for k in PCA_DIMENSIONS if curve[k] <= threshold), None)
        z_train = q_train[centers_train] @ spec["weights"]
        z_test = q_test[centers_test] @ spec["weights"]
        z_beta = linear_readout_fit(z_train[:, None], train.symbols)
        scalar_nmse = nmse(
            test.symbols, linear_readout_predict(z_test[:, None], z_beta)
        )
        geom = spectrum_geometry(X_train)
        scalar_rows.append({
            **base, "geometry": geometry, "condition": "scalar_min",
            "heldout_symbol_nmse": float(scalar_nmse),
            "nmse_delta_vs_raw": float(scalar_nmse - raw_nmse),
            "nmse_ratio_to_raw": float(scalar_nmse / max(raw_nmse, np.finfo(float).tiny)),
            "full_state_nmse": float(full_nmse),
            "d_task_10pct": d_task,
            "pca_energy_at_d_task": pca_energy(singular, d_task) if d_task else float("nan"),
            **KERNEL_DESCRIPTORS[geometry], **geom,
        })
        case.update({
            f"{geometry}_full_state_nmse": float(full_nmse),
            f"{geometry}_scalar_nmse": float(scalar_nmse),
            f"{geometry}_d_task_10pct": d_task,
            f"{geometry}_pca_energy_at_d_task": pca_energy(singular, d_task) if d_task else float("nan"),
            f"{geometry}_center_state_participation": geom["center_state_participation_dimension"],
        })

    return curve_rows, scalar_rows, case


def main():
    equivalence_error = validate_repository_equivalence()
    print(f"repository_state_equivalence_relative_error={equivalence_error:.3e}")

    curve_rows, scalar_rows, cases = [], [], []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for environment in ENVIRONMENTS:
                for snr_db in SNR_DB:
                    a, b, c = run_case(signal_name, generator, environment, snr_db, seed)
                    curve_rows.extend(a)
                    scalar_rows.extend(b)
                    cases.append(c)

    expected_cases = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS) * len(SNR_DB)
    assert len(cases) == expected_cases
    assert len(curve_rows) == expected_cases * len(GEOMETRIES) * len(PCA_DIMENSIONS)
    assert len(scalar_rows) == expected_cases * len(GEOMETRIES)

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pca": out / "14C_geometry_to_task_pca_results.csv",
        "scalar": out / "14C_geometry_to_task_scalar_results.csv",
        "cases": out / "14C_geometry_to_task_cases.csv",
        "summary": out / "14C_geometry_to_task_summary.csv",
        "metadata": out / "14C_geometry_to_task_summary.json",
    }
    for path, data in (
        (outputs["pca"], curve_rows),
        (outputs["scalar"], scalar_rows),
        (outputs["cases"], cases),
    ):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    grouped = {}
    for r in scalar_rows:
        grouped.setdefault((r["geometry"], r["environment"], r["snr_db"]), []).append(r)
    summary_rows = []
    for (geometry, environment, snr_db), bucket in sorted(grouped.items()):
        dvals = [r["d_task_10pct"] for r in bucket if r["d_task_10pct"] is not None]
        summary_rows.append({
            "geometry": geometry, "environment": environment,
            "snr_db": float(snr_db), "n": len(bucket),
            "mean_scalar_nmse": float(np.mean([r["heldout_symbol_nmse"] for r in bucket])),
            "std_scalar_nmse": float(np.std([r["heldout_symbol_nmse"] for r in bucket], ddof=1)),
            "mean_full_state_nmse": float(np.mean([r["full_state_nmse"] for r in bucket])),
            "std_full_state_nmse": float(np.std([r["full_state_nmse"] for r in bucket], ddof=1)),
            "mean_d_task": float(np.mean(dvals)) if dvals else float("nan"),
            "mean_center_state_participation": float(
                np.mean([r["center_state_participation_dimension"] for r in bucket])
            ),
            "mean_nmse_ratio_to_raw": float(
                np.mean([r["nmse_ratio_to_raw"] for r in bucket])
            ),
        })
    with outputs["summary"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "experiment": "14C_geometry_to_task",
        "purpose": "Isolate temporal rate geometry effects on task-relevant MIN structure under fixed task environments.",
        "grid": {
            "signals": list(SIGNALS), "environments": list(ENVIRONMENTS),
            "snr_db": list(SNR_DB), "geometries": list(GEOMETRIES),
            "seeds": list(SEEDS), "nominal_modes": MODES,
            "pca_dimensions": list(PCA_DIMENSIONS),
        },
        "cases": expected_cases,
        "pca_rows": len(curve_rows),
        "scalar_rows": len(scalar_rows),
        "geometry_definitions": {
            "clustered": "10*exp(linspace(-0.05,0.05,16)) s^-1",
            "logspread": "geomspace(2,30,16) s^-1",
            "wide": "geomspace(0.5,100,16) s^-1",
        },
        "weight_control": "uniform positive weights in every geometry; H_mem and D_eff are identical across the three geometry conditions.",
        "observation_control": "Each geometry is applied to the exact same noisy y_train/y_test realization within each task case.",
        "task_dimension": "smallest retained PCA dimension with held-out NMSE <= 1.10 times that geometry's full-state NMSE.",
        "interpretation_boundary": [
            "This is a geometry-to-task experiment, not a search for a universally best geometry.",
            "The environment is fixed within each paired case; representation geometry is the intervention.",
            "L and D_eff are matched across geometries, while rate support and M_scale differ.",
            "D_task is operational and task-specific, not an information-theoretic or physical state dimension.",
            "Scalar-MIN results are secondary because projection can conflate geometry with one-dimensional compression.",
        ],
        "repository_equivalence_relative_error": equivalence_error,
        "outputs": [p.name for p in outputs.values()],
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=True) + "\n")
    print(json.dumps(metadata, indent=2, allow_nan=True))
    print(f"rows_pca={len(curve_rows)} rows_scalar={len(scalar_rows)} cases={len(cases)}")


if __name__ == "__main__":
    main()
