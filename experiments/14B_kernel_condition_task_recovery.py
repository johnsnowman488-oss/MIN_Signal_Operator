"""Experiment 14B: task performance under environment-kernel conditions.

14B tests the H2 question from the Experiment 14 roadmap:

    Does alignment between the true environment and the MIN/SOE kernel
    materially affect a downstream digital-symbol recovery task?

Each task case is evaluated with the same observed signal and held-out
symbols under four representation conditions:

    raw          noisy symbol-center observation; no memory representation
    informed     oracle environment-derived 16-mode MIN/SOE state
    estimated    kernel estimated from an independent finite/noisy
                 environment-only observation
    mismatched   oracle kernel belonging to a different environment

The downstream model is deliberately unchanged from 14A: a complex linear
readout trained on 512 symbols and evaluated on a separate 512-symbol
sequence. 14B uses the full 16-state representation rather than PCA so that
the experiment isolates kernel-condition effects rather than task
dimensionality.

The primary task metric is held-out normalized symbol reconstruction MSE
(NMSE). Kernel fidelity and state-geometry descriptors are recorded so that
task changes can be related back to the representation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.kernels import SOEMemory
from min.signals import generate_16qam, generate_bpsk, generate_qpsk


SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
ENVIRONMENTS = ("white_limit", "short", "multiscale", "powerlaw", "squared_exp")
SEEDS = tuple(range(5))
SNR_DB = (0.0, 10.0, 20.0, 30.0)

NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE

FIT_HORIZON_S = 0.08
ESTIMATION_OBSERVATION_LENGTH = 2048
DICTIONARY_GAMMAS = np.geomspace(0.5, 100.0, 16)

ENV_NOISE_SEED_OFFSET = 100_003
ESTIMATION_SEED_OFFSET = 2_000_003
READOUT_RIDGE = 1e-10

CONDITIONS = ("informed", "estimated", "mismatched", "raw_center")
MISMATCH_MAP = {
    environment: ENVIRONMENTS[(index + 1) % len(ENVIRONMENTS)]
    for index, environment in enumerate(ENVIRONMENTS)
}


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


def fit_kernel_from_samples(
    lag_times: np.ndarray,
    covariance: np.ndarray,
    gammas: np.ndarray = DICTIONARY_GAMMAS,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.maximum(np.asarray(covariance, dtype=float), 0.0)
    A = np.exp(-np.outer(lag_times, gammas))
    weights, _ = nnls(A, target)
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("NNLS returned zero total kernel weight")
    return gammas.copy(), weights / total


def oracle_kernel(environment: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, FIT_HORIZON_S, 2048)
    target = environment_envelope(t, environment)
    return fit_kernel_from_samples(t, target)


def synth_environment_noise(
    environment: str,
    n: int,
    seed: int,
) -> np.ndarray:
    """Generate a normalized real Gaussian process with the target covariance."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * DT
    c = environment_envelope(t, environment)
    circ = np.r_[c, c[-2:0:-1]]
    spectrum = np.maximum(np.real(np.fft.rfft(circ)), 0.0)

    z = rng.normal(size=spectrum.size) + 1j * rng.normal(size=spectrum.size)
    z[0] = rng.normal()
    if circ.size % 2 == 0:
        z[-1] = rng.normal()

    noise = np.fft.irfft(
        np.sqrt(spectrum) * z,
        n=circ.size,
    )[:n]
    return noise / max(float(np.std(noise)), np.finfo(float).tiny)


def estimate_environment_kernel(
    environment: str,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate an environment kernel from a separate noisy environment probe."""
    latent = synth_environment_noise(
        environment,
        ESTIMATION_OBSERVATION_LENGTH,
        seed,
    )
    rng = np.random.default_rng(seed + 777_001)
    white_sigma = math.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))
    observed = latent + rng.normal(scale=white_sigma, size=latent.size)

    centered = observed - np.mean(observed)
    n = centered.size
    max_lag = int(round(FIT_HORIZON_S / DT))
    lags = np.arange(max_lag + 1, dtype=int)
    covariance = np.array(
        [
            np.dot(centered[: n - lag], centered[lag:]) / max(n - lag, 1)
            for lag in lags
        ],
        dtype=float,
    )
    covariance /= max(float(covariance[0]), np.finfo(float).tiny)

    lag_times = lags * DT
    gammas, weights = fit_kernel_from_samples(lag_times, covariance)
    fitted = np.exp(-np.outer(lag_times, gammas)) @ weights
    true = environment_envelope(lag_times, environment)

    kernel_fit_error = float(
        np.linalg.norm(fitted - true)
        / max(np.linalg.norm(true), np.finfo(float).tiny)
    )
    return gammas, weights, {
        "estimation_kernel_fit_relative_l2": kernel_fit_error,
        "estimation_covariance_relative_l2": float(
            np.linalg.norm(covariance - true)
            / max(np.linalg.norm(true), np.finfo(float).tiny)
        ),
    }


def exact_piecewise_linear_states(
    x: np.ndarray,
    gammas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fast implementation of the same piecewise-linear recurrence as SOEMemory."""
    samples = np.asarray(x, dtype=complex)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("x must be a one-dimensional array with at least two samples")

    dt = DT
    decay = np.exp(-gammas * dt)
    b_curr = 1.0 / gammas - (1.0 - decay) / (dt * gammas**2)
    b_prev = (1.0 - decay) / gammas - b_curr

    # q[n] = decay*q[n-1] + b_curr*x[n] + b_prev*x[n-1].
    # The first sample remains at zero, matching SOEMemory.state_trajectory.
    filtered = np.empty((samples.size, gammas.size), dtype=complex)
    for j, (a, bc, bp) in enumerate(zip(decay, b_curr, b_prev)):
        forcing = np.empty(samples.size, dtype=complex)
        forcing[0] = 0.0
        forcing[1:] = bc * samples[1:] + bp * samples[:-1]
        filtered[:, j] = lfilter(
            [1.0],
            [1.0, -a],
            forcing,
        )

    return filtered


def validate_accelerated_state_equivalence() -> float:
    """Check the fast uniform-grid realization against repository SOEMemory once."""
    rng = np.random.default_rng(41)
    x = rng.normal(size=257) + 1j * rng.normal(size=257)
    gammas = DICTIONARY_GAMMAS.copy()
    weights = np.full(gammas.size, 1.0 / gammas.size)
    fast = exact_piecewise_linear_states(x, gammas, weights)
    reference = SOEMemory(weights, gammas).state_trajectory(
        np.arange(x.size, dtype=float) * DT,
        x,
    )
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    rel_error = float(np.linalg.norm(fast - reference) / scale)
    if rel_error > 1e-9:
        raise RuntimeError(
            f"Accelerated MIN state path differs from repository reference: {rel_error:.3e}"
        )
    return rel_error


def min_state_trajectory(
    x: np.ndarray,
    gammas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Use the exact repository-equivalent fast realization on the uniform grid."""
    return exact_piecewise_linear_states(x, gammas, weights)


def linear_readout_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    gram.flat[:: gram.shape[0] + 1] += READOUT_RIDGE
    rhs = design.conj().T @ y
    return np.linalg.solve(gram, rhs)


def linear_readout_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    return design @ beta


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.mean(np.abs(y_pred - y_true) ** 2)
    power = np.mean(np.abs(y_true) ** 2)
    return float(err / max(power, np.finfo(float).tiny))


def raw_center_samples(observed: np.ndarray, symbol_count: int, sps: int) -> np.ndarray:
    return np.asarray(observed).reshape(symbol_count, sps)[:, sps // 2]


def state_features(
    state: np.ndarray,
    symbols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.arange(len(symbols), dtype=int) * SPS + SPS // 2
    return state[centers], np.asarray(symbols, dtype=complex)


def kernel_descriptors(
    gammas: np.ndarray,
    weights: np.ndarray,
) -> dict:
    w = weights / max(float(weights.sum()), np.finfo(float).tiny)
    active = w > 1e-12
    h = -float(np.sum(w[active] * np.log(w[active])))
    scale = (
        math.log10(float(gammas[active].max() / gammas[active].min()))
        if np.count_nonzero(active) > 1
        else 0.0
    )
    return {
        "dictionary_mode_count": int(len(gammas)),
        "nonzero_kernel_modes": int(np.count_nonzero(active)),
        "gfe_h_mem_nats": h,
        "gfe_entropy_effective_count": float(np.exp(h)),
        "gfe_d_eff": float(len(gammas) * np.exp(h)),
        "gfe_m_scale_decades": float(scale),
        "gfe_m_res_modes_per_decade": (
            float(np.count_nonzero(active) / scale) if scale > 0 else float("nan")
        ),
        "gfe_m_cap_s": float(
            np.sum(w / gammas**2)
            / max(np.sum(w / gammas), np.finfo(float).tiny)
        ),
    }


def min_scalar_trajectory(
    x: np.ndarray,
    gammas: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the scalar MIN/SOE output z[n] = sum_j w_j q_j[n]."""
    q = exact_piecewise_linear_states(x, gammas, weights)
    return q @ weights


def state_geometry(q: np.ndarray) -> dict:
    centered = q - np.mean(q, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(q.shape[0] - 1, 1)
    eig = np.sort(np.maximum(np.real(np.linalg.eigvalsh(cov)), 0.0))[::-1]
    total = float(eig.sum())
    if total <= 0:
        return {
            "state_participation_dimension": 0.0,
            "state_entropy_dimension": 0.0,
            "state_rank90": 0,
            "state_rank99": 0,
        }
    p = eig / total
    return {
        "state_participation_dimension": float(1.0 / np.sum(p**2)),
        "state_entropy_dimension": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        "state_rank90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "state_rank99": int(np.searchsorted(np.cumsum(p), 0.99) + 1),
    }


def run_case(
    signal_name: str,
    generator,
    environment: str,
    snr_db: float,
    seed: int,
    oracle_kernels: dict,
    estimated_kernels: dict,
):
    train = generator(
        NUM_TRAIN_SYMBOLS,
        samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE,
        seed=seed,
        normalize=False,
    )
    test = generator(
        NUM_TEST_SYMBOLS,
        samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE,
        seed=seed + 10_000,
        normalize=False,
    )

    x_train = np.asarray(train.record.samples, dtype=complex)
    x_test = np.asarray(test.record.samples, dtype=complex)
    signal_power = float(np.mean(np.abs(x_train) ** 2))

    noise_train = synth_environment_noise(
        environment,
        len(x_train),
        seed + ENV_NOISE_SEED_OFFSET,
    )
    noise_test = synth_environment_noise(
        environment,
        len(x_test),
        seed + ENV_NOISE_SEED_OFFSET + 50_000,
    )
    noise_scale = math.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    y_train = x_train + noise_train * noise_scale
    y_test = x_test + noise_test * noise_scale

    raw_train = raw_center_samples(y_train, NUM_TRAIN_SYMBOLS, SPS)
    raw_test = raw_center_samples(y_test, NUM_TEST_SYMBOLS, SPS)
    raw_beta = linear_readout_fit(raw_train[:, None], train.symbols)
    raw_nmse = nmse(
        test.symbols,
        linear_readout_predict(raw_test[:, None], raw_beta),
    )

    rows = []
    for condition in ("informed", "estimated", "mismatched"):
        if condition == "informed":
            condition_environment = environment
            gammas, weights = oracle_kernels[environment]
        elif condition == "estimated":
            condition_environment = environment
            gammas, weights, estimate_meta = estimated_kernels[(environment, snr_db, seed)]
        else:
            condition_environment = MISMATCH_MAP[environment]
            gammas, weights = oracle_kernels[condition_environment]
            estimate_meta = {}

        z_train = min_scalar_trajectory(y_train, gammas, weights)
        z_test = min_scalar_trajectory(y_test, gammas, weights)
        centers_train = np.arange(len(train.symbols), dtype=int) * SPS + SPS // 2
        centers_test = np.arange(len(test.symbols), dtype=int) * SPS + SPS // 2
        X_train = z_train[centers_train, None]
        X_test = z_test[centers_test, None]
        symbols_train = np.asarray(train.symbols, dtype=complex)
        symbols_test = np.asarray(test.symbols, dtype=complex)

        beta = linear_readout_fit(X_train, symbols_train)
        pred = linear_readout_predict(X_test, beta)
        condition_nmse = nmse(symbols_test, pred)

        oracle_weights = oracle_kernels[environment][1]
        ref_fit = np.exp(-np.outer(
            np.linspace(0.0, FIT_HORIZON_S, 2048),
            gammas,
        )) @ weights
        true_env = environment_envelope(
            np.linspace(0.0, FIT_HORIZON_S, 2048),
            environment,
        )

        common = {
            "experiment": "14B_kernel_condition_task_recovery",
            "signal": signal_name,
            "environment": environment,
            "snr_db": float(snr_db),
            "seed": int(seed),
            "condition": condition,
            "condition_environment": condition_environment,
            "mismatch_environment": MISMATCH_MAP[environment] if condition == "mismatched" else "",
            "modal_state_dimension": int(len(gammas)),
            "representation_dimension": 1,
            "heldout_symbol_nmse": float(condition_nmse),
            "raw_center_sample_nmse": float(raw_nmse),
            "nmse_delta_vs_raw": float(condition_nmse - raw_nmse),
            "nmse_ratio_to_informed": float("nan"),
            "kernel_fit_relative_l2_to_true_environment": float(
                np.linalg.norm(ref_fit - true_env)
                / max(np.linalg.norm(true_env), np.finfo(float).tiny)
            ),
            "kernel_weight_relative_l2_to_oracle": float(
                np.linalg.norm(weights - oracle_weights)
                / max(np.linalg.norm(oracle_weights), np.finfo(float).tiny)
            ),
            "estimation_kernel_fit_relative_l2": float("nan"),
            "estimation_covariance_relative_l2": float("nan"),
        }
        common.update(kernel_descriptors(gammas, weights))
        common.update(geom)
        if condition == "estimated":
            common.update(estimate_meta)

        rows.append(common)

    informed_nmse = rows[0]["heldout_symbol_nmse"]
    for row in rows:
        row["nmse_ratio_to_informed"] = float(
            row["heldout_symbol_nmse"]
            / max(informed_nmse, np.finfo(float).tiny)
        )
    raw_row = {
        "experiment": "14B_kernel_condition_task_recovery",
        "signal": signal_name,
        "environment": environment,
        "snr_db": float(snr_db),
        "seed": int(seed),
        "condition": "raw_center",
        "condition_environment": "",
        "mismatch_environment": "",
        "full_state_dimension": 1,
        "heldout_symbol_nmse": float(raw_nmse),
        "raw_center_sample_nmse": float(raw_nmse),
        "nmse_delta_vs_raw": 0.0,
        "nmse_ratio_to_informed": float(raw_nmse / max(informed_nmse, np.finfo(float).tiny)),
        "kernel_fit_relative_l2_to_true_environment": float("nan"),
        "kernel_weight_relative_l2_to_oracle": float("nan"),
    }
    rows.append(raw_row)
    return rows


def main() -> None:
    equivalence_error = validate_accelerated_state_equivalence()
    print(f"accelerated_state_equivalence_relative_error={equivalence_error:.3e}")

    oracle_kernels = {
        env_name: oracle_kernel(env_name)
        for env_name in ENVIRONMENTS
    }
    estimated_kernels = {}
    for environment in ENVIRONMENTS:
        for snr_db in SNR_DB:
            for seed in SEEDS:
                estimated_kernels[(environment, snr_db, seed)] = estimate_environment_kernel(
                    environment,
                    snr_db,
                    seed + ESTIMATION_SEED_OFFSET,
                )

    rows = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for environment in ENVIRONMENTS:
                for snr_db in SNR_DB:
                    rows.extend(
                        run_case(
                            signal_name,
                            generator,
                            environment,
                            snr_db,
                            seed,
                            oracle_kernels,
                            estimated_kernels,
                        )
                    )

    expected_cases = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS) * len(SNR_DB)
    expected_rows = expected_cases * 4
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} rows from {expected_cases} cases, got {len(rows)}"
        )

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / "14B_kernel_condition_task_recovery_results.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["environment"], row["condition"], row["snr_db"]),
            [],
        ).append(row)

    summary_rows = []
    for (environment, condition, snr_db), bucket in sorted(grouped.items()):
        item = {
            "environment": environment,
            "condition": condition,
            "snr_db": float(snr_db),
            "rows": len(bucket),
            "independent_seeds": len(SEEDS),
            "mean_heldout_symbol_nmse": float(
                np.mean([r["heldout_symbol_nmse"] for r in bucket])
            ),
            "std_heldout_symbol_nmse": float(
                np.std([r["heldout_symbol_nmse"] for r in bucket], ddof=1)
            ),
            "mean_nmse_ratio_to_informed": float(
                np.mean([r["nmse_ratio_to_informed"] for r in bucket])
            ),
            "mean_nmse_delta_vs_raw": float(
                np.mean([r["nmse_delta_vs_raw"] for r in bucket])
            ),
        }
        summary_rows.append(item)

    summary_path = out / "14B_kernel_condition_task_recovery_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "experiment": "14B_kernel_condition_task_recovery",
        "purpose": (
            "Test whether environment-kernel alignment changes held-out digital "
            "symbol recovery when the downstream decoder is held fixed."
        ),
        "case_grid": {
            "signals": list(SIGNALS),
            "environments": list(ENVIRONMENTS),
            "snr_db": list(SNR_DB),
            "seeds": list(SEEDS),
            "train_symbols": NUM_TRAIN_SYMBOLS,
            "test_symbols": NUM_TEST_SYMBOLS,
            "state_modes": len(DICTIONARY_GAMMAS),
            "environment_estimation_observation_length": ESTIMATION_OBSERVATION_LENGTH,
        },
        "kernel_conditions": {
            "informed": "oracle kernel fit to the declared environment covariance envelope",
            "estimated": (
                "kernel fitted by NNLS from an independent finite/noisy environment-only "
                "probe at the same SNR level"
            ),
            "mismatched": (
                "oracle kernel from the next environment in a fixed cyclic map; not task-tuned"
            ),
            "raw_center": "noisy observed y at the symbol center; no MIN/SOE state",
        },
        "heldout_protocol": (
            "The test symbol sequence is not used to estimate the environment kernel, "
            "fit PCA, or fit the readout."
        ),
        "primary_metric": "heldout_symbol_nmse",
        "comparison_metric": "nmse_ratio_to_informed",
        "representation": (
            "scalar repository SOEMemory/MIN output z=w^T q; the kernel weights therefore "
            "directly affect the task representation. PCA is intentionally omitted."
        ),
        "mismatch_map": MISMATCH_MAP,
        "accelerated_state_equivalence_check": "single startup comparison against repository SOEMemory.state_trajectory",
        "boundaries": [
            "This is an H2 environment-alignment test, not a universal performance benchmark.",
            "The raw observation remains a control because 14A established it as a viable task representation.",

            "No claim is made that the estimated condition must outperform the mismatched or raw condition.",
            "No BER, EVM, neural network, or task-optimized kernel is used.",
            "D_eff is recorded as a kernel-complexity descriptor and is not equated with task dimension.",
        ],
        "outputs": [
            "14B_kernel_condition_task_recovery_results.csv",
            "14B_kernel_condition_task_recovery_summary.csv",
            "14B_kernel_condition_task_recovery_summary.json",
        ],
    }

    (out / "14B_kernel_condition_task_recovery_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(json.dumps(metadata, indent=2))
    print(f"rows={len(rows)} cases={expected_cases}")


if __name__ == "__main__":
    main()
