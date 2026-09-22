"""Experiment 12D: regularized SOE state estimation and inverse conditioning.

12D follows 12C by separating representation from inversion. It measures the
conditioning of the induced causal SOE input/output map and compares direct
inversion with Tikhonov/ridge and truncated-SVD estimators under controlled
model/channel/noise conditions.

This is a batch held-out estimator. It is not evidence that MIN should be
inverted in ordinary communications; the conditioning diagnostics are the
primary scientific output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.metrics.equalization import add_awgn, evm


def load_12c():
    path = ROOT / "experiments" / "12C_soe_state_input_estimation.py"
    spec = importlib.util.spec_from_file_location("exp12c", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


E = load_12c()

SIGNALS = E.SIGNALS
SOE_MEMORIES = E.SOE_MEMORIES
CHANNELS = E.CHANNELS
SNR_DB = E.SNR_DB
NUM_SYMBOLS = E.NUM_SYMBOLS
TRAIN_SYMBOLS = E.TRAIN_SYMBOLS
SPS = E.SPS
SYMBOL_RATE = E.SYMBOL_RATE
SEEDS = E.SEEDS

METHODS = ("direct", "ridge", "tsvd")
CASES = (
    "exact_model_exact_channel_noiseless",
    "exact_model_exact_channel_noisy",
    "exact_model_estimated_channel_noiseless",
    "exact_model_estimated_channel_noisy",
    "estimated_model_estimated_channel_noisy",
)

# Dimensionless sweep. Actual ridge values are scaled by sigma_max(H)^2.
RIDGE_FACTORS = (1e-8, 1e-6, 1e-4, 1e-2, 1e0)
TSVD_RELATIVE_CUTOFFS = (1e-8, 1e-6, 1e-4, 1e-2)


def bits(name, s):
    return E.bits(name, s)


def ber(name, tx, z):
    return E.ber(name, tx, z)


def conv_matrix(h, n):
    H = np.zeros((n, n), dtype=complex)
    for r in range(n):
        k = min(r + 1, len(h))
        H[r, :k] = h[:k][::-1]
    return H


def perturb_memory(mem):
    """Deterministic receiver-side SOE mismatch for the model-mismatch case."""
    w, g = SOE_MEMORIES[mem]
    w = np.asarray(w, dtype=float).copy()
    g = np.asarray(g, dtype=float).copy()
    g *= np.linspace(0.93, 1.07, len(g))
    w *= np.linspace(1.06, 0.94, len(w))
    w /= np.sum(w)
    return w, g


def sampled_impulse_custom(w, g, n):
    dt = 1.0 / (SYMBOL_RATE * SPS)
    t = np.arange(256 * SPS) * dt
    k = sum(a * np.exp(-b * t) for a, b in zip(w, g))
    k /= k.sum()
    x = np.zeros(n * SPS, dtype=complex)
    x[:SPS] = 1
    y = np.convolve(x, k, mode="full")[: x.size]
    return y[np.arange(n) * SPS + SPS // 2]


def state_impulse(mem, model_kind):
    if model_kind == "exact":
        return E.state_model(mem, 512)[4]
    w, g = perturb_memory(mem)
    # Reproduce the fixed-pole construction used by 12C for the perturbed SOE.
    h = sampled_impulse_custom(w, g, 512)
    poles = np.exp(-g / SYMBOL_RATE)
    m = len(w)
    A = np.diag(poles.astype(complex))
    D = h[0]
    V = np.vstack([poles**n for n in range(1, m + 1)])
    B = np.linalg.solve(V, h[1 : m + 1])
    C = np.ones((1, m), dtype=complex)
    hh = np.empty(512, dtype=complex)
    hh[0] = D
    state = B.copy()
    for n in range(1, 512):
        hh[n] = (C @ state)[0]
        state = A @ state
    return hh


def subtract_known_prefix(prefix, h_model, n_test):
    """Contribution predicted by h_model from the known training prefix."""
    out = np.zeros(n_test, dtype=complex)
    start = len(prefix)
    for j in range(n_test):
        global_idx = start + j
        kmax = min(global_idx + 1, len(h_model))
        for k in range(kmax):
            src = global_idx - k
            if src < start:
                out[j] += h_model[k] * prefix[src]
    return out


def channel_from_name(x, name, rng):
    return E.channel(x, name, rng)


def channel_estimate(tx_train, obs_train, mem, taps):
    return E.estimate_channel(tx_train, obs_train, mem, taps)


def solve_direct(H, y):
    try:
        return solve_triangular(H, y, lower=True)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.linalg.lstsq(H, y, rcond=None)[0]


def solve_ridge(H, y, factor):
    s = np.linalg.svd(H, compute_uv=False)
    scale = max(float(s[0] ** 2), np.finfo(float).tiny)
    lam = factor * scale
    A = H.conj().T @ H + lam * np.eye(H.shape[1])
    b = H.conj().T @ y
    return np.linalg.solve(A, b), lam


def solve_tsvd(H, y, relative_cutoff):
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    cutoff = relative_cutoff * s[0]
    keep = s > cutoff
    if not np.any(keep):
        return np.zeros(H.shape[1], dtype=complex), 0
    z = Vh[keep].conj().T @ ((U[:, keep].conj().T @ y) / s[keep])
    return z, int(np.count_nonzero(keep))


def complexity(method, state_dim):
    if method == "direct":
        return state_dim, 2 * state_dim + 1
    if method == "ridge":
        return state_dim, 2 * state_dim + 1
    return state_dim, 2 * state_dim + 1


def run_case(signal_name, gen, mem, ch, seed, snr, case):
    sig = gen(NUM_SYMBOLS, samples_per_symbol=SPS, symbol_rate=SYMBOL_RATE, seed=seed)
    clean, true_channel = channel_from_name(
        E.memory_filter(sig.record.samples, mem),
        ch,
        np.random.default_rng(10000 + seed),
    )

    noisy = case.endswith("noisy")
    obs_samples = add_awgn(
        clean,
        snr if noisy else 1000.0,
        np.random.default_rng(20000 + seed * 100 + int(snr)),
    )
    if not noisy:
        obs_samples = clean

    r = obs_samples[sig.symbol_indices]
    tx = sig.symbols
    tx_train = tx[:TRAIN_SYMBOLS]
    y_train = r[:TRAIN_SYMBOLS]
    y_test = r[TRAIN_SYMBOLS:]
    n_test = len(y_test)

    exact_model = case.startswith("exact_model")
    model_kind = "exact" if exact_model else "estimated"
    h_model = state_impulse(mem, model_kind)

    exact_channel = "exact_channel" in case
    if exact_channel:
        h_est = true_channel
    else:
        taps = 1 if ch in ("identity", "flat_rayleigh") else 3
        h_est = channel_estimate(tx_train, y_train, mem, taps)

    h_true = np.convolve(state_impulse(mem, "exact"), true_channel)
    h_model_total = np.convolve(h_model, h_est)

    # The unknown held-out input is preceded by the known training sequence.
    y_unknown = y_test - subtract_known_prefix(tx_train, h_model_total, n_test)
    H = conv_matrix(h_model_total, n_test)

    s = np.linalg.svd(H, compute_uv=False)
    smax = float(s[0])
    smin = float(s[-1])
    cond = float(smax / smin) if smin > 0 else float("inf")
    rank = int(np.count_nonzero(s > max(smax * 1e-12, np.finfo(float).eps)))
    true_H = conv_matrix(h_true, n_test)
    true_s = np.linalg.svd(true_H, compute_uv=False)
    true_cond = float(true_s[0] / true_s[-1]) if true_s[-1] > 0 else float("inf")

    rows = []

    def emit(method, hyperparameter, z, aux=None):
        finite = bool(np.all(np.isfinite(z)))
        if finite:
            evm_pct = 100.0 * evm(tx[TRAIN_SYMBOLS:], z)
            bit_error = ber(signal_name, tx[TRAIN_SYMBOLS:], z)
            mse = float(np.mean(np.abs(z - tx[TRAIN_SYMBOLS:]) ** 2))
            noise_gain = float(
                np.linalg.norm(z - np.linalg.lstsq(H, y_unknown, rcond=None)[0])
                / max(np.linalg.norm(y_unknown), np.finfo(float).tiny)
            )
        else:
            evm_pct = float("inf")
            bit_error = 1.0
            mse = float("inf")
            noise_gain = float("inf")

        params, macs = complexity(method, len(SOE_MEMORIES[mem][0]))
        rows.append(
            dict(
                experiment="12D_regularized_SOE_state_estimation",
                signal=signal_name,
                memory=mem,
                channel=ch,
                case=case,
                method=method,
                hyperparameter=hyperparameter,
                snr_db=snr,
                seed=seed,
                evm_percent=evm_pct,
                ber=bit_error,
                heldout_mse=mse,
                sigma_max=smax,
                sigma_min=smin,
                condition_number=cond,
                true_operator_condition_number=true_cond,
                numerical_rank=rank,
                state_dimension=len(SOE_MEMORIES[mem][0]),
                parameter_count=params,
                macs_per_sample=macs,
                estimated_channel_norm=float(np.linalg.norm(h_est)),
                true_channel_norm=float(np.linalg.norm(true_channel)),
                noise_gain_proxy=noise_gain,
                retained_singular_values=aux if aux is not None else "",
                finite=finite,
            )
        )

    emit("direct", "none", solve_direct(H, y_unknown))

    for factor in RIDGE_FACTORS:
        z, lam = solve_ridge(H, y_unknown, factor)
        emit("ridge", f"{factor:g}", z, f"lambda={lam:.6e}")

    for cutoff in TSVD_RELATIVE_CUTOFFS:
        z, retained = solve_tsvd(H, y_unknown, cutoff)
        emit("tsvd", f"{cutoff:g}", z, retained)

    return rows


def main():
    rows = []
    for seed in SEEDS:
        for signal_name, gen in SIGNALS.items():
            for mem in SOE_MEMORIES:
                for ch in CHANNELS:
                    for snr in SNR_DB:
                        for case in CASES:
                            rows.extend(run_case(signal_name, gen, mem, ch, seed, snr, case))

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "12D_regularized_SOE_state_estimation_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    finite_rows = [r for r in rows if r["finite"]]
    summary = {
        "experiment": "12D_regularized_SOE_state_estimation",
        "rows": len(rows),
        "finite_rows": len(finite_rows),
        "cases": list(CASES),
        "methods": list(METHODS),
        "ridge_factors": list(RIDGE_FACTORS),
        "tsvd_relative_cutoffs": list(TSVD_RELATIVE_CUTOFFS),
        "evaluation": "384 held-out symbols after 128-symbol training prefix",
        "purpose": "conditioning diagnosis and robust batch estimation; not a claim that MIN should be inverted",
        "primary_diagnostics": [
            "singular spectrum",
            "minimum singular value",
            "condition number",
            "noise amplification proxy",
            "EVM",
            "BER",
            "held-out MSE",
            "state dimension",
            "MACs/sample",
        ],
    }
    (out / "12D_regularized_SOE_state_estimation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Compact console diagnostics for CI.
    print(json.dumps(summary, indent=2))
    for mem in SOE_MEMORIES:
        subset = [
            r for r in rows
            if r["memory"] == mem
            and r["case"] == "exact_model_exact_channel_noisy"
            and float(r["snr_db"]) == 30.0
        ]
        if subset:
            best = min(subset, key=lambda r: r["evm_percent"])
            print(
                mem,
                "30dB exact/noisy:",
                "condition=", best["condition_number"],
                "best_method=", best["method"],
                "hyperparameter=", best["hyperparameter"],
                "EVM%=", best["evm_percent"],
                "BER=", best["ber"],
            )


if __name__ == "__main__":
    main()
