"""Experiment 14D: representation accessibility and decoder conditioning.

14D holds the observed waveform and MIN/SOE geometry fixed within each paired
case, then varies only the downstream readout. It asks whether the geometry
effects seen in 14C survive controlled changes in decoder conditioning,
regularization, and retained state dimension.

The primary state is the full 16-mode MIN/SOE realization. Scalar MIN remains
a secondary diagnostic. Ridge strength is normalized by the training design
Gram trace so that lambda values are comparable across geometries/cases.
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
RIDGE_LAMBDAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1)
COND_THRESHOLD = 1e12


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    scale = max(float(np.trace(gram).real) / gram.shape[0], np.finfo(float).tiny)
    alpha = float(lam) * scale
    reg = np.eye(gram.shape[0], dtype=complex)
    reg[0, 0] = 0.0
    if lam == 0.0:
        # Use SVD least squares for the true unregularized baseline; a direct
        # Gram solve is unstable for the highly redundant clustered states.
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
    raw_train = y_train[centers_train]
    raw_test = y_test[centers_test]
    raw_beta, _ = ridge_fit(raw_train[:, None], train.symbols, 1e-6)
    raw_nmse = nmse(test.symbols, predict(raw_test[:, None], raw_beta))

    rows = []
    for geometry in GEOMETRIES:
        kernel = mod.KERNELS[geometry]
        q_train = mod.exact_piecewise_linear_states(
            y_train, kernel["gammas"], kernel["coefficients"]
        )[centers_train]
        q_test = mod.exact_piecewise_linear_states(
            y_test, kernel["gammas"], kernel["coefficients"]
        )[centers_test]

        cond = conditioning(q_train)
        mean = np.mean(q_train, axis=0)
        _, singular, components = mod.complex_pca_fit(q_train)

        for d in PCA_DIMENSIONS:
            X_train = (q_train - mean) @ components[:, :d]
            X_test = (q_test - mean) @ components[:, :d]
            for lam in RIDGE_LAMBDAS:
                beta, alpha = ridge_fit(X_train, train.symbols, lam)
                value = nmse(test.symbols, predict(X_test, beta))
                rows.append({
                    "experiment": "14D_readout_stability",
                    "signal": signal_name,
                    "environment": environment,
                    "snr_db": float(snr_db),
                    "seed": int(seed),
                    "geometry": geometry,
                    "retained_dimension": int(d),
                    "ridge_lambda": float(lam),
                    "ridge_alpha": float(alpha),
                    "heldout_symbol_nmse": value,
                    "raw_center_sample_nmse": raw_nmse,
                    "nmse_delta_vs_raw": value - raw_nmse,
                    "pca_explained_energy": mod.pca_energy(singular, d),
                    "representation": "full_state",
                    **cond,
                    **mod.KERNEL_DESCRIPTORS[geometry],
                })

        z_train = q_train @ kernel["weights"]
        z_test = q_test @ kernel["weights"]
        beta, alpha = ridge_fit(z_train[:, None], train.symbols, 1e-6)
        scalar_nmse = nmse(test.symbols, predict(z_test[:, None], beta))
        rows.append({
            "experiment": "14D_readout_stability",
            "signal": signal_name,
            "environment": environment,
            "snr_db": float(snr_db),
            "seed": int(seed),
            "geometry": geometry,
            "retained_dimension": 1,
            "ridge_lambda": 1e-6,
            "ridge_alpha": alpha,
            "heldout_symbol_nmse": scalar_nmse,
            "raw_center_sample_nmse": raw_nmse,
            "nmse_delta_vs_raw": scalar_nmse - raw_nmse,
            "pca_explained_energy": float("nan"),
            "representation": "scalar_min",
            **cond,
            **mod.KERNEL_DESCRIPTORS[geometry],
        })
    return rows


def summarize(rows):
    state = [r for r in rows if r.get("representation", "full_state") == "full_state"]
    grouped = {}
    for r in state:
        key = (r["geometry"], r["retained_dimension"], r["ridge_lambda"], r["snr_db"])
        grouped.setdefault(key, []).append(r)
    out = []
    for key, bucket in sorted(grouped.items()):
        geometry, d, lam, snr = key
        vals = [r["heldout_symbol_nmse"] for r in bucket]
        out.append({
            "geometry": geometry,
            "retained_dimension": d,
            "ridge_lambda": lam,
            "snr_db": snr,
            "n": len(bucket),
            "mean_nmse": float(np.mean(vals)),
            "median_nmse": float(np.median(vals)),
            "std_nmse": float(np.std(vals, ddof=1)),
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
    expected_scalar = expected_cases * len(GEOMETRIES)
    assert cases == expected_cases
    assert len(all_rows) == expected_full + expected_scalar

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    detail = out / "14D_readout_stability_results.csv"
    summary = out / "14D_readout_stability_summary.csv"
    metadata = out / "14D_readout_stability_summary.json"

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
        "experiment": "14D_readout_stability",
        "purpose": "Determine whether 14C geometry effects persist under controlled readout conditioning, ridge regularization, and retained state dimension.",
        "cases": cases,
        "full_state_rows": expected_full,
        "scalar_rows": expected_scalar,
        "geometries": list(GEOMETRIES),
        "dimensions": list(PCA_DIMENSIONS),
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "conditioning_threshold": COND_THRESHOLD,
        "ridge_normalization": "lambda multiplied by mean diagonal scale of the augmented training Gram matrix; intercept is not regularized.",
        "repository_equivalence_relative_error": eq,
        "paired_observation_control": "same noisy y_train/y_test is reused across geometries within each task case.",
        "heldout_protocol": "kernel, PCA basis, and readout are fit only from training symbols; held-out symbols are used only for evaluation.",
        "interpretation_boundary": [
            "This experiment tests decoder accessibility/stability, not a universal geometry ranking.",
            "D is the retained PCA state dimension and is distinct from D_eff, D_state, and D_task.",
            "Condition number is a readout diagnostic, not itself evidence of information content.",
            "If regularization changes geometry gaps, that gap has a decoder-conditioning component; if gaps persist, the evidence for representation-level geometry effects strengthens.",
        ],
    }
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"rows={len(all_rows)} cases={cases} full_state_rows={expected_full} scalar_rows={expected_scalar}")


if __name__ == "__main__":
    main()
