"""Experiment 14A: task-relevant subspace of environment-informed MIN states.

14A is the first task-level experiment in the MIN Signal Operator program.
It asks whether an environment-informed MIN/SOE state contains recoverable
digital-symbol information, and how much of that state is required by a
simple linear readout.

Protocol:
    digital symbols -> colored environment noise -> observation
    observation -> oracle environment kernel -> MIN/SOE state
    training states -> complex PCA -> linear readout -> held-out symbols

The oracle kernel is fitted from the known environment covariance envelope
using the same fixed 16-rate positive SOE dictionary used in 13C-1/2/3.
The held-out symbol sequence is never used to fit the kernel, PCA basis, or
readout.

This experiment intentionally does NOT use BER, EVM, neural networks, or
task optimization. The primary task metric is held-out normalized symbol
reconstruction MSE (NMSE). The key structural output is performance as a
function of retained complex PCA dimension.

Dimension hierarchy is preserved:
    L != D_eff != D_basis != D_state != D_task

Here D_task is operationally probed by the smallest retained PCA dimension
whose held-out NMSE reaches a declared tolerance relative to the full-state
linear-readout reference. It is a task-specific empirical quantity, not a
new geometric identity.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import nnls

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

DICTIONARY_GAMMAS = np.geomspace(0.5, 100.0, 16)
FIT_HORIZON_S = 0.08
PCA_DIMENSIONS = tuple(range(1, 17))
TASK_TOLERANCE = 1.10  # within 10% of the full-state held-out NMSE

ENV_NOISE_SEEDS = 100_003
READOUT_RIDGE = 1e-10


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


def fit_oracle_kernel(environment: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, FIT_HORIZON_S, 2048)
    target = environment_envelope(t, environment)
    A = np.exp(-np.outer(t, DICTIONARY_GAMMAS))
    weights, _ = nnls(A, target)
    weights /= max(float(weights.sum()), np.finfo(float).tiny)
    return DICTIONARY_GAMMAS.copy(), weights


def kernel_descriptors(gammas: np.ndarray, weights: np.ndarray) -> dict:
    active = weights > 1e-12
    w = weights / max(float(weights.sum()), np.finfo(float).tiny)
    h = -float(np.sum(w[active] * np.log(w[active])))
    scale = (
        math.log10(float(gammas[active].max() / gammas[active].min()))
        if np.sum(active) > 1
        else 0.0
    )
    return {
        "dictionary_mode_count": int(len(gammas)),
        "nonzero_kernel_modes": int(np.sum(active)),
        "gfe_h_mem_nats": h,
        "gfe_entropy_effective_count": float(np.exp(h)),
        "gfe_d_eff": float(len(gammas) * np.exp(h)),
        "gfe_m_scale_decades": scale,
        "gfe_m_res_modes_per_decade": (
            float(np.sum(active) / scale) if scale > 0 else float("nan")
        ),
        "gfe_m_cap_s": float(
            np.sum(w / gammas**2) / max(np.sum(w / gammas), np.finfo(float).tiny)
        ),
    }


def colored_noise(
    n: int, environment: str, seed: int, snr_db: float, reference_power: float
) -> np.ndarray:
    """Generate a real stationary colored-noise probe with the environment covariance."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    c = environment_envelope(t, environment)
    # Circulant embedding for a real Gaussian process with target covariance
    # envelope. Negative numerical spectral bins are clipped to zero.
    circ = np.r_[c, c[-2:0:-1]]
    spectrum = np.maximum(np.real(np.fft.rfft(circ)), 0.0)
    z = rng.normal(size=spectrum.size) + 1j * rng.normal(size=spectrum.size)
    z[0] = rng.normal()
    if circ.size % 2 == 0:
        z[-1] = rng.normal()
    noise = np.fft.irfft(np.sqrt(spectrum) * z, n=circ.size)[:n]
    noise /= max(float(np.std(noise)), np.finfo(float).tiny)

    noise_power = reference_power / (10.0 ** (snr_db / 10.0))
    return noise * math.sqrt(noise_power)


def min_state_trajectory(
    x: np.ndarray, gammas: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Obtain the full MIN/SOE state through the repository operator API.

    The returned columns are the modal memory states q_j. The fitted kernel
    weights remain attached to the SOEMemory model and are not collapsed into
    the scalar MIN output because 14A measures task-relevant structure in the
    full state.
    """
    samples = np.asarray(x, dtype=complex)
    t = np.arange(samples.size, dtype=float) * DT
    return SOEMemory(weights, gammas).state_trajectory(t, samples)


def raw_center_samples(
    observed: np.ndarray, symbol_count: int, sps: int
) -> np.ndarray:
    """Return the noisy observation-space control at symbol centers."""
    return np.asarray(observed).reshape(symbol_count, sps)[:, sps // 2]


def symbol_center_rows(state: np.ndarray, symbols: np.ndarray, sps: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.arange(len(symbols)) * sps + sps // 2
    return state[centers], np.asarray(symbols, dtype=complex)


def complex_pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    centered = X - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh.conj().T
    return mean, singular, components


def complex_pca_project(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int
) -> np.ndarray:
    return (X - mean) @ components[:, :k]


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


def pca_explained_energy(singular: np.ndarray, k: int) -> float:
    energy = singular**2
    return float(np.sum(energy[:k]) / max(np.sum(energy), np.finfo(float).tiny))


def task_dimension(full_nmse: float, curve: dict[int, float]) -> int | None:
    target = full_nmse * TASK_TOLERANCE
    for k in sorted(curve):
        if curve[k] <= target:
            return int(k)
    return None


def run_case(signal_name: str, generator, environment: str, snr_db: float, seed: int):
    gammas, weights = fit_oracle_kernel(environment)
    descriptors = kernel_descriptors(gammas, weights)

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

    # The environment acts as colored additive noise. Scaling is based only
    # on the training signal's observed power and is then held fixed for test.
    signal_power = float(np.mean(np.abs(x_train) ** 2))
    noise_train = colored_noise(
        len(x_train), environment, seed + ENV_NOISE_SEEDS, snr_db, signal_power
    )
    noise_test = colored_noise(
        len(x_test), environment, seed + ENV_NOISE_SEEDS + 50_000, snr_db, signal_power
    )
    y_train = x_train + noise_train
    y_test = x_test + noise_test

    q_train = min_state_trajectory(y_train, gammas, weights)
    q_test = min_state_trajectory(y_test, gammas, weights)

    X_train, y_train_symbols = symbol_center_rows(q_train, train.symbols, SPS)
    X_test, y_test_symbols = symbol_center_rows(q_test, test.symbols, SPS)

    mean, singular, components = complex_pca_fit(X_train)

    curve: dict[int, float] = {}
    rows = []
    for k in PCA_DIMENSIONS:
        Z_train = complex_pca_project(X_train, mean, components, k)
        Z_test = complex_pca_project(X_test, mean, components, k)
        beta = linear_readout_fit(Z_train, y_train_symbols)
        pred = linear_readout_predict(Z_test, beta)
        test_nmse = nmse(y_test_symbols, pred)
        curve[k] = test_nmse
        rows.append({
            "experiment": "14A_task_relevant_min_subspace",
            "signal": signal_name,
            "environment": environment,
            "snr_db": float(snr_db),
            "seed": int(seed),
            "retained_pca_dimension": int(k),
            "total_state_dimension": int(len(gammas)),
            "pca_explained_energy": pca_explained_energy(singular, k),
            "heldout_symbol_nmse": test_nmse,
            "full_state_reference": int(k == len(gammas)),
        })

    full_nmse = curve[len(gammas)]
    d_task = task_dimension(full_nmse, curve)

    # Raw observation linear-readout control at symbol centers.
    raw_train = raw_center_samples(y_train, NUM_TRAIN_SYMBOLS, SPS)
    raw_test = raw_center_samples(y_test, NUM_TEST_SYMBOLS, SPS)
    raw_beta = linear_readout_fit(raw_train[:, None], train.symbols)
    raw_nmse = nmse(test.symbols, linear_readout_predict(raw_test[:, None], raw_beta))

    case = {
        "experiment": "14A_task_relevant_min_subspace",
        "signal": signal_name,
        "environment": environment,
        "snr_db": float(snr_db),
        "seed": int(seed),
        "full_state_nmse": float(full_nmse),
        "raw_center_sample_nmse": float(raw_nmse),
        "task_dimension_10pct": d_task,
        "pca_energy_at_task_dimension": (
            pca_explained_energy(singular, d_task) if d_task is not None else float("nan")
        ),
        "total_state_dimension": int(len(gammas)),
        "task_tolerance_relative_to_full": TASK_TOLERANCE,
    }
    case.update(descriptors)
    return rows, case


def main():
    all_rows = []
    cases = []
    for seed in SEEDS:
        for signal_name, generator in SIGNALS.items():
            for environment in ENVIRONMENTS:
                for snr_db in SNR_DB:
                    rows, case = run_case(
                        signal_name, generator, environment, snr_db, seed
                    )
                    all_rows.extend(rows)
                    cases.append(case)

    expected_cases = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS) * len(SNR_DB)
    expected_rows = expected_cases * len(PCA_DIMENSIONS)
    if len(cases) != expected_cases or len(all_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_cases} cases/{expected_rows} rows, "
            f"got {len(cases)} cases/{len(all_rows)} rows"
        )

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / "14A_task_relevant_min_subspace_results.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    case_path = out / "14A_task_relevant_min_subspace_cases.csv"
    with case_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    metadata = {
        "experiment": "14A_task_relevant_min_subspace",
        "purpose": "Measure task-relevant subspace of environment-informed MIN/SOE states for held-out digital-symbol reconstruction.",
        "grid": {
            "signals": list(SIGNALS),
            "environments": list(ENVIRONMENTS),
            "snr_db": list(SNR_DB),
            "seeds": list(SEEDS),
            "train_symbols": NUM_TRAIN_SYMBOLS,
            "test_symbols": NUM_TEST_SYMBOLS,
            "pca_dimensions": list(PCA_DIMENSIONS),
            "soe_dictionary_modes": len(DICTIONARY_GAMMAS),
            "gamma_range_s_inverse": [float(DICTIONARY_GAMMAS.min()), float(DICTIONARY_GAMMAS.max())],
        },
        "cases": expected_cases,
        "rows": expected_rows,
        "primary_metric": "heldout_symbol_nmse",
        "min_representation": "repository SOEMemory.state_trajectory with fitted oracle kernel weights/gammas; full modal states retained",
        "raw_control_representation": "noisy y_train/y_test center samples only",
        "task_dimension_definition": (
            "smallest retained complex PCA dimension whose held-out NMSE is <= "
            f"{TASK_TOLERANCE:.2f} times the full-state reference NMSE"
        ),
        "controls": [
            "oracle environment-derived kernel from fixed positive 16-rate dictionary",
            "separate held-out symbol sequence",
            "PCA and linear readout fit only on training symbols",
            "raw center-sample linear readout control",
            "same signal/environment/SNR/seed grid across PCA dimensions",
            "repository SOEMemory state_trajectory used for MIN/SOE representation",
        ],
        "explicit_exclusions": [
            "no BER",
            "no EVM",
            "no neural network",
            "no downstream task optimization",
            "no claim that D_task equals D_eff, D_basis, or D_state",
        ],
        "dimension_hierarchy": "L != D_eff != D_basis != D_state != D_task",
        "outputs": [
            "14A_task_relevant_min_subspace_results.csv",
            "14A_task_relevant_min_subspace_cases.csv",
            "14A_task_relevant_min_subspace_summary.json",
        ],
    }
    (out / "14A_task_relevant_min_subspace_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(json.dumps(metadata, indent=2))
    by_group = {}
    for c in cases:
        key = (c["signal"], c["environment"], c["snr_db"])
        by_group.setdefault(key, []).append(c)
    for key, bucket in sorted(by_group.items()):
        vals = [c["task_dimension_10pct"] for c in bucket if c["task_dimension_10pct"] is not None]
        print(
            "group=", key,
            "cases=", len(bucket),
            "task_dimension_mean=",
            float(np.mean(vals)) if vals else None,
            "full_nmse_mean=",
            float(np.mean([c["full_state_nmse"] for c in bucket])),
        )


if __name__ == "__main__":
    main()
