"""Experiment 15A: memory geometry x task generalization.

15A keeps the controlled 14C geometry intervention but evaluates three
downstream tasks on the same noisy signal cases:
  1. denoising: recover the clean current waveform sample;
  2. prediction: predict the next clean waveform sample causally;
  3. symbol recovery: recover the transmitted symbol at the symbol center.

The three 16-mode geometries use identical positive uniform weights and hence
matched nominal mode count and D_eff. Only the decay-rate geometry changes.

15A is intentionally self-contained: it imports numerical utilities only from
15_utils.py and does not import any 14-series experiment module.
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
from importlib.util import module_from_spec, spec_from_file_location

UTIL_PATH = ROOT / "experiments" / "15_utils.py"
UTIL_SPEC = spec_from_file_location("min_exp15_utils", UTIL_PATH)
UTIL = module_from_spec(UTIL_SPEC)
sys.modules[UTIL_SPEC.name] = UTIL
UTIL_SPEC.loader.exec_module(UTIL)

SIGNALS = UTIL.SIGNALS
ENVIRONMENTS = UTIL.ENVIRONMENTS
GEOMETRIES = UTIL.GEOMETRIES
SEEDS = UTIL.SEEDS
SNR_DB = UTIL.SNR_DB
SPS = UTIL.SPS
MODES = UTIL.MODES
PCA_DIMENSIONS = UTIL.PCA_DIMENSIONS
TASK_TOLERANCE = UTIL.TASK_TOLERANCE
ENV_NOISE_SEED_OFFSET = UTIL.ENV_NOISE_SEED_OFFSET
NUM_TRAIN_SYMBOLS = UTIL.NUM_TRAIN_SYMBOLS
NUM_TEST_SYMBOLS = UTIL.NUM_TEST_SYMBOLS
SYMBOL_RATE = UTIL.SYMBOL_RATE
DT = UTIL.DT
SAMPLE_RATE = UTIL.SAMPLE_RATE

TASKS = ("denoise", "prediction", "symbol_recovery")


def task_arrays(
    x_clean: np.ndarray,
    y_noisy: np.ndarray,
    symbols: np.ndarray,
    task: str,
    train: bool,
):
    if task == "denoise":
        # Same-time causal readout: q[n] contains y[0:n].
        return y_noisy, x_clean
    if task == "prediction":
        # Strictly causal one-step prediction: q[n] -> x[n+1].
        return y_noisy[:-1], x_clean[1:]
    if task == "symbol_recovery":
        centers = np.arange(len(symbols)) * SPS + SPS // 2
        return y_noisy[centers], symbols
    raise ValueError(task)


def task_baseline(
    x_clean: np.ndarray,
    y_noisy: np.ndarray,
    symbols: np.ndarray,
    task: str,
):
    X, target = task_arrays(x_clean, y_noisy, symbols, task, train=True)
    beta = UTIL.linear_readout_fit(X[:, None], target)
    pred = UTIL.linear_readout_predict(X[:, None], beta)
    return UTIL.nmse(target, pred)


def run_case(signal_name, generator, environment, snr_db, seed):
    train = generator(
        NUM_TRAIN_SYMBOLS, samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE, seed=seed, normalize=False
    )
    test = generator(
        NUM_TEST_SYMBOLS, samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE, seed=seed + 10_000, normalize=False
    )
    x_train = np.asarray(train.record.samples, dtype=complex)
    x_test = np.asarray(test.record.samples, dtype=complex)

    signal_power = float(np.mean(np.abs(x_train) ** 2))
    scale = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    y_train = x_train + scale * UTIL.synth_environment_noise(
        environment, len(x_train), seed + ENV_NOISE_SEED_OFFSET
    )
    y_test = x_test + scale * UTIL.synth_environment_noise(
        environment, len(x_test), seed + ENV_NOISE_SEED_OFFSET + 50_000
    )

    base = {
        "experiment": "15A_geometry_task_generalization",
        "signal": signal_name,
        "environment": environment,
        "snr_db": float(snr_db),
        "seed": int(seed),
    }

    task_rows = []
    curve_rows = []

    for task in TASKS:
        raw_nmse = task_baseline(x_train, y_train, train.symbols, task)
        raw_test_X, raw_test_target = task_arrays(
            x_test, y_test, test.symbols, task, train=False
        )
        raw_train_X, raw_train_target = task_arrays(
            x_train, y_train, train.symbols, task, train=True
        )
        raw_beta = UTIL.linear_readout_fit(raw_train_X[:, None], raw_train_target)
        raw_nmse = UTIL.nmse(
            raw_test_target,
            UTIL.linear_readout_predict(raw_test_X[:, None], raw_beta),
        )

        for geometry in GEOMETRIES:
            spec = UTIL.KERNELS[geometry]
            q_train = UTIL.exact_piecewise_linear_states(
                y_train, spec["gammas"], spec["coefficients"]
            )
            q_test = UTIL.exact_piecewise_linear_states(
                y_test, spec["gammas"], spec["coefficients"]
            )

            q_train_task, target_train = task_arrays(
                x_train, y_train, train.symbols, task, train=True
            )
            q_test_task, target_test = task_arrays(
                x_test, y_test, test.symbols, task, train=False
            )

            # q_*_task is only used to determine alignment; use the same
            # indexing on the corresponding full MIN state.
            if task == "prediction":
                X_train = q_train[:-1]
                X_test = q_test[:-1]
            elif task == "symbol_recovery":
                centers_train = np.arange(len(train.symbols)) * SPS + SPS // 2
                centers_test = np.arange(len(test.symbols)) * SPS + SPS // 2
                X_train = q_train[centers_train]
                X_test = q_test[centers_test]
            else:
                X_train = q_train
                X_test = q_test

            mean, singular, components = UTIL.complex_pca_fit(X_train)
            state_geom = UTIL.spectrum_geometry(X_train)

            curve = {}
            for k in PCA_DIMENSIONS:
                beta = UTIL.linear_readout_fit(
                    UTIL.pca_project(X_train, mean, components, k),
                    target_train,
                )
                value = UTIL.nmse(
                    target_test,
                    UTIL.linear_readout_predict(
                        UTIL.pca_project(X_test, mean, components, k), beta
                    ),
                )
                curve[k] = value
                curve_rows.append({
                    **base,
                    "task": task,
                    "geometry": geometry,
                    "retained_pca_dimension": int(k),
                    "heldout_nmse": float(value),
                    "raw_baseline_nmse": float(raw_nmse),
                    "nmse_delta_vs_raw": float(value - raw_nmse),
                    "nmse_ratio_to_raw": float(
                        value / max(raw_nmse, np.finfo(float).tiny)
                    ),
                    "pca_explained_energy": UTIL.pca_energy(singular, k),
                    "full_state_reference": int(k == MODES),
                    **UTIL.KERNEL_DESCRIPTORS[geometry],
                    **state_geom,
                })

            full_nmse = curve[MODES]
            threshold = TASK_TOLERANCE * full_nmse
            d_task = next(
                (k for k in PCA_DIMENSIONS if curve[k] <= threshold), None
            )

            task_rows.append({
                **base,
                "task": task,
                "geometry": geometry,
                "raw_baseline_nmse": float(raw_nmse),
                "full_state_nmse": float(full_nmse),
                "d_task_10pct": d_task,
                "pca_energy_at_d_task": (
                    UTIL.pca_energy(singular, d_task)
                    if d_task is not None else float("nan")
                ),
                "nmse_gain_vs_raw": float(raw_nmse - full_nmse),
                "nmse_ratio_to_raw": float(
                    full_nmse / max(raw_nmse, np.finfo(float).tiny)
                ),
                **UTIL.KERNEL_DESCRIPTORS[geometry],
                **state_geom,
            })

    return curve_rows, task_rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["task"], row["geometry"], row["environment"], row["snr_db"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, bucket in sorted(grouped.items()):
        task, geometry, environment, snr_db = key
        dvals = [r["d_task_10pct"] for r in bucket
                 if r["d_task_10pct"] is not None]
        summary.append({
            "task": task,
            "geometry": geometry,
            "environment": environment,
            "snr_db": float(snr_db),
            "n": len(bucket),
            "mean_full_state_nmse": float(
                np.mean([r["full_state_nmse"] for r in bucket])
            ),
            "std_full_state_nmse": float(
                np.std([r["full_state_nmse"] for r in bucket], ddof=1)
            ),
            "mean_raw_baseline_nmse": float(
                np.mean([r["raw_baseline_nmse"] for r in bucket])
            ),
            "mean_nmse_ratio_to_raw": float(
                np.mean([r["nmse_ratio_to_raw"] for r in bucket])
            ),
            "mean_d_task": float(np.mean(dvals)) if dvals else float("nan"),
            "std_d_task": float(np.std(dvals, ddof=1)) if len(dvals) > 1 else 0.0,
            "mean_state_participation_dimension": float(
                np.mean([r["state_participation_dimension"] for r in bucket])
            ),
        })
    return summary


def main():
    equivalence_error = UTIL.validate_repository_equivalence()
    print(f"repository_state_equivalence_relative_error={equivalence_error:.3e}")

    curve_rows, task_rows = [], []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for environment in ENVIRONMENTS:
                for snr_db in SNR_DB:
                    curves, tasks = run_case(
                        signal_name, generator, environment, snr_db, seed
                    )
                    curve_rows.extend(curves)
                    task_rows.extend(tasks)

    expected_cases = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS) * len(SNR_DB)
    expected_task_rows = expected_cases * len(TASKS) * len(GEOMETRIES)
    expected_curve_rows = expected_task_rows * len(PCA_DIMENSIONS)
    assert len(task_rows) == expected_task_rows
    assert len(curve_rows) == expected_curve_rows

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "15A_geometry_task_generalization_pca_results.csv"
    task_path = out / "15A_geometry_task_generalization_cases.csv"
    summary_path = out / "15A_geometry_task_generalization_summary.csv"
    metadata_path = out / "15A_geometry_task_generalization_summary.json"

    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    with task_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    summary_rows = summarize(task_rows)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "experiment": "15A_geometry_task_generalization",
        "purpose": "Test whether the controlled memory-rate-geometry effect observed in 14C persists across distinct downstream tasks.",
        "tasks": {
            "denoise": "causal readout from q[n] to clean x[n]",
            "prediction": "strictly causal readout from q[n] to clean x[n+1]",
            "symbol_recovery": "readout from q at symbol-center samples to transmitted symbol",
        },
        "grid": {
            "signals": list(SIGNALS),
            "environments": list(ENVIRONMENTS),
            "snr_db": list(SNR_DB),
            "geometries": list(GEOMETRIES),
            "seeds": list(SEEDS),
            "nominal_modes": MODES,
            "pca_dimensions": list(PCA_DIMENSIONS),
        },
        "cases": expected_cases,
        "task_rows": expected_task_rows,
        "pca_rows": expected_curve_rows,
        "geometry_control": "All geometries use 16 positive uniform weights, so nominal mode count, weight entropy, and D_eff are matched; only rate geometry changes.",
        "observation_control": "Within each signal/environment/SNR/seed case, every geometry sees the exact same noisy realization.",
        "readout_control": "The same linear complex ridge readout and PCA protocol is used for every geometry/task pairing.",
        "task_dimension": "Smallest retained PCA dimension whose held-out NMSE is within 10% of that geometry/task full 16-state NMSE.",
        "negative_result_policy": "No geometry is assumed to be superior. Cross-task disagreement, null effects, or raw/history advantages are retained as results.",
        "independence": "15A imports only 15_utils.py; it does not import 14-series experiment modules.",
        "repository_state_equivalence_relative_error": equivalence_error,
        "outputs": [
            curve_path.name, task_path.name, summary_path.name, metadata_path.name
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(
        f"rows_pca={len(curve_rows)} rows_tasks={len(task_rows)} "
        f"cases={expected_cases}"
    )


if __name__ == "__main__":
    main()
