"""Experiment 14E: causal one-step-ahead prediction from MIN/SOE states.

The target is the noisy observed waveform at the next symbol center. Symbols
are IID in the controlled generators, so next-symbol prediction would contain
no legitimate signal-side predictability. Predicting the next observed sample
instead tests whether the memory state retains temporally useful information
about the colored environment/noise.

Within each case, the observation, environment, signal, SNR, and geometry are
fixed. The predictor at center n only sees samples through center n and predicts
the observation at center n+1. No future samples enter state construction,
PCA fitting, or readout fitting.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

C_PATH = ROOT / "experiments" / "14C_geometry_to_task.py"
spec = importlib.util.spec_from_file_location("experiment14c", C_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

SIGNALS = mod.SIGNALS
ENVIRONMENTS = mod.ENVIRONMENTS
GEOMETRIES = mod.GEOMETRIES
SEEDS = mod.SEEDS
SNR_DB = mod.SNR_DB
PCA_DIMENSIONS = (1, 2, 4, 8, 12, 16)
RIDGE_LAMBDAS = (0.0, 1e-8, 1e-6)
COND_THRESHOLD = 1e12
TASK_TOLERANCE = 1.10


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    scale = max(float(np.trace(gram).real) / gram.shape[0], np.finfo(float).tiny)
    alpha = float(lam) * scale
    reg = np.eye(gram.shape[0], dtype=complex)
    reg[0, 0] = 0.0
    if lam == 0.0:
        return np.linalg.lstsq(design, y, rcond=None)[0], 0.0
    beta = np.linalg.solve(gram + alpha * reg, design.conj().T @ y)
    return beta, alpha


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) ** 2) /
                 max(np.mean(np.abs(y_true) ** 2), np.finfo(float).tiny))


def conditioning(X: np.ndarray) -> dict:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    singular = np.linalg.svd(design, compute_uv=False)
    smax = float(singular[0])
    positive = singular[singular > smax * np.finfo(float).eps]
    smin = float(positive[-1]) if positive.size else 0.0
    condition = float(smax / smin) if smin > 0 else float("inf")
    rank = int(np.sum(singular > smax * np.finfo(float).eps))
    return {
        "readout_sigma_max": smax,
        "readout_sigma_min": smin,
        "readout_condition_number": condition,
        "readout_numerical_rank": rank,
        "readout_ill_conditioned": int(condition > COND_THRESHOLD),
    }


def prepare_case(generator, environment, snr_db, seed):
    train = generator(mod.NUM_TRAIN_SYMBOLS, samples_per_symbol=mod.SPS,
                      symbol_rate=mod.SYMBOL_RATE, seed=seed, normalize=False)
    test = generator(mod.NUM_TEST_SYMBOLS, samples_per_symbol=mod.SPS,
                     symbol_rate=mod.SYMBOL_RATE, seed=seed + 10_000,
                     normalize=False)
    x_train = np.asarray(train.record.samples, dtype=complex)
    x_test = np.asarray(test.record.samples, dtype=complex)
    signal_power = float(np.mean(np.abs(x_train) ** 2))
    scale = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    y_train = x_train + scale * mod.synth_environment_noise(
        environment, len(x_train), seed + mod.ENV_NOISE_SEED_OFFSET
    )
    y_test = x_test + scale * mod.synth_environment_noise(
        environment, len(x_test), seed + mod.ENV_NOISE_SEED_OFFSET + 50_000
    )
    centers_train = np.arange(mod.NUM_TRAIN_SYMBOLS) * mod.SPS + mod.SPS // 2
    centers_test = np.arange(mod.NUM_TEST_SYMBOLS) * mod.SPS + mod.SPS // 2
    return train, test, y_train, y_test, centers_train, centers_test


def run_case(signal_name, generator, environment, snr_db, seed):
    train, test, y_train, y_test, centers_train, centers_test = prepare_case(
        generator, environment, snr_db, seed
    )

    # Causal pairs: state/observation at n -> observation at n+1.
    past_train = centers_train[:-1]
    target_train = centers_train[1:]
    past_test = centers_test[:-1]
    target_test = centers_test[1:]

    raw_train = y_train[past_train]
    raw_target_train = y_train[target_train]
    raw_test = y_test[past_test]
    raw_target_test = y_test[target_test]
    raw_beta, _ = ridge_fit(raw_train[:, None], raw_target_train, 1e-6)
    raw_nmse = nmse(raw_target_test, predict(raw_test[:, None], raw_beta))

    rows = []
    for geometry in GEOMETRIES:
        kernel = mod.KERNELS[geometry]
        q_train_all = mod.exact_piecewise_linear_states(
            y_train, kernel["gammas"], kernel["coefficients"]
        )
        q_test_all = mod.exact_piecewise_linear_states(
            y_test, kernel["gammas"], kernel["coefficients"]
        )
        q_train = q_train_all[past_train]
        q_test = q_test_all[past_test]

        cond = conditioning(q_train)
        mean = np.mean(q_train, axis=0)
        _, singular, components = mod.complex_pca_fit(q_train)

        curve = {}
        for d in PCA_DIMENSIONS:
            X_train = (q_train - mean) @ components[:, :d]
            X_test = (q_test - mean) @ components[:, :d]
            for lam in RIDGE_LAMBDAS:
                beta, alpha = ridge_fit(X_train, raw_target_train, lam)
                value = nmse(raw_target_test, predict(X_test, beta))
                curve[(d, lam)] = value
                rows.append({
                    "experiment": "14E_causal_prediction",
                    "signal": signal_name,
                    "environment": environment,
                    "snr_db": float(snr_db),
                    "seed": int(seed),
                    "geometry": geometry,
                    "retained_dimension": int(d),
                    "ridge_lambda": float(lam),
                    "ridge_alpha": float(alpha),
                    "heldout_next_observation_nmse": value,
                    "raw_current_observation_nmse": raw_nmse,
                    "nmse_delta_vs_raw": value - raw_nmse,
                    "pca_explained_energy": mod.pca_energy(singular, d),
                    "representation": "full_state",
                    "d_task_10pct": None,
                    **cond,
                    **mod.KERNEL_DESCRIPTORS[geometry],
                })

        reference = curve[(16, 1e-6)]
        threshold = TASK_TOLERANCE * reference
        d_task = next(
            (d for d in PCA_DIMENSIONS if curve[(d, 1e-6)] <= threshold),
            None,
        )

        z_train = q_train @ kernel["weights"]
        z_test = q_test @ kernel["weights"]
        for lam in RIDGE_LAMBDAS:
            beta, alpha = ridge_fit(z_train[:, None], raw_target_train, lam)
            value = nmse(raw_target_test, predict(z_test[:, None], beta))
            rows.append({
                "experiment": "14E_causal_prediction",
                "signal": signal_name,
                "environment": environment,
                "snr_db": float(snr_db),
                "seed": int(seed),
                "geometry": geometry,
                "retained_dimension": 1,
                "ridge_lambda": float(lam),
                "ridge_alpha": float(alpha),
                "heldout_next_observation_nmse": value,
                "raw_current_observation_nmse": raw_nmse,
                "nmse_delta_vs_raw": value - raw_nmse,
                "pca_explained_energy": float("nan"),
                "representation": "scalar_min",
                "d_task_10pct": d_task,
                **cond,
                **mod.KERNEL_DESCRIPTORS[geometry],
            })
    return rows


def summarize(rows):
    grouped = {}
    for r in rows:
        if r["representation"] != "full_state":
            continue
        key = (r["geometry"], r["retained_dimension"], r["ridge_lambda"], r["snr_db"])
        grouped.setdefault(key, []).append(r)
    out = []
    for key, bucket in sorted(grouped.items()):
        geometry, d, lam, snr = key
        vals = [r["heldout_next_observation_nmse"] for r in bucket]
        out.append({
            "geometry": geometry,
            "retained_dimension": d,
            "ridge_lambda": lam,
            "snr_db": snr,
            "n": len(bucket),
            "mean_nmse": float(np.mean(vals)),
            "median_nmse": float(np.median(vals)),
            "std_nmse": float(np.std(vals, ddof=1)),
            "mean_raw_nmse": float(np.mean([r["raw_current_observation_nmse"] for r in bucket])),
            "mean_condition_number": float(np.mean([r["readout_condition_number"] for r in bucket])),
            "mean_pca_energy": float(np.mean([r["pca_explained_energy"] for r in bucket])),
        })
    return out


def main():
    eq = mod.validate_repository_equivalence()
    all_rows = []
    cases = 0
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for environment in ENVIRONMENTS:
                for snr_db in SNR_DB:
                    all_rows.extend(run_case(signal_name, generator, environment, snr_db, seed))
                    cases += 1

    expected_cases = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS) * len(SNR_DB)
    expected_full = expected_cases * len(GEOMETRIES) * len(PCA_DIMENSIONS) * len(RIDGE_LAMBDAS)
    expected_scalar = expected_cases * len(GEOMETRIES) * len(RIDGE_LAMBDAS)
    assert cases == expected_cases
    assert len(all_rows) == expected_full + expected_scalar

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    detail = out / "14E_causal_prediction_results.csv"
    summary = out / "14E_causal_prediction_summary.csv"
    metadata = out / "14E_causal_prediction_summary.json"

    with detail.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = summarize(all_rows)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "experiment": "14E_causal_prediction",
        "purpose": "Test whether environment-informed MIN/SOE states retain temporally useful information for one-symbol-ahead prediction of the observed waveform.",
        "target": "noisy observed waveform at the next symbol center",
        "cases": cases,
        "full_state_rows": expected_full,
        "scalar_rows": expected_scalar,
        "geometries": list(GEOMETRIES),
        "dimensions": list(PCA_DIMENSIONS),
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "task_tolerance": TASK_TOLERANCE,
        "causality": "predictor at center n uses only samples through center n; target is center n+1.",
        "baseline": "current noisy center sample predicting the next noisy center sample with lambda=1e-6.",
        "target_rationale": "The controlled symbol generators are IID, so next-symbol prediction would not test temporal memory. The next observed sample retains colored-environment temporal dependence while preserving unpredictable future symbol content.",
        "repository_equivalence_relative_error": eq,
        "paired_observation_control": "same noisy y_train/y_test is reused across geometries within each task case.",
        "heldout_protocol": "kernel, PCA basis, and readout are fit only from causal training pairs; held-out future observations are used only for evaluation.",
        "interpretation_boundary": [
            "This is a temporal-information diagnostic, not a universal MIN performance benchmark.",
            "Prediction error includes an irreducible component from the independent future symbol.",
            "Improvement over the current-observation baseline is the key accessibility comparison.",
            "D_task is operational and task-specific, not a physical or information-theoretic state dimension.",
        ],
    }
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"rows={len(all_rows)} cases={cases} full_state_rows={expected_full} scalar_rows={expected_scalar}")


if __name__ == "__main__":
    main()
