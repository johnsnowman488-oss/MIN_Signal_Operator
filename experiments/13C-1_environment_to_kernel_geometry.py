"""Experiment 13C-1: Environment -> Kernel Geometry -> State Geometry.

Task-agnostic continuation of 13A/13B.

An abstract environment is represented by a normalized temporal covariance
envelope C_env(tau). A positive SOE kernel is then identified from that
environment envelope using a fixed logarithmic decay-rate dictionary and NNLS.
The derived kernel is applied to the same controlled digital-signal atlas used
in 13B. This isolates the proposed chain

    environment -> kernel geometry -> temporal basis geometry -> MIN state geometry

without a downstream task, channel, receiver, or learned model.

Important boundary:
- the environment envelope is an oracle covariance geometry in this first
  variant; noisy covariance estimation is deferred to 13C-2.
- the SOE dictionary is fixed at 16 logarithmically spaced rates.
- NNLS is used only to enforce nonnegative kernel weights; it is not a task
  optimizer.
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
from min.signals import generate_16qam, generate_bpsk, generate_qpsk


SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
ENVIRONMENTS = ("white_limit", "short", "multiscale", "powerlaw", "squared_exp")
SEEDS = tuple(range(4))

NUM_SYMBOLS, SPS, SYMBOL_RATE = 512, 16, 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE
HORIZON_S = NUM_SYMBOLS / SYMBOL_RATE

DICTIONARY_MODES = 16
DICTIONARY_GAMMAS = np.geomspace(0.5, 100.0, DICTIONARY_MODES)
FIT_POINTS = 1024


def environment_envelope(t: np.ndarray, name: str) -> np.ndarray:
    """Return a normalized nonnegative covariance envelope C(0)=1."""
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


def fit_positive_soe(t: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit target envelope with a fixed positive exponential dictionary."""
    A = np.exp(-np.outer(t, DICTIONARY_GAMMAS))
    weights, _ = nnls(A, target)
    total = float(np.sum(weights))
    if total <= 0:
        raise RuntimeError("NNLS returned zero total kernel weight")
    weights /= total
    return DICTIONARY_GAMMAS.copy(), weights


def gfe_metrics(gammas: np.ndarray, weights: np.ndarray) -> dict:
    """Record the GGFE/GFE descriptors used in 13A/13B."""
    w = weights / np.sum(weights)
    active = w > 0
    m_cap = float(np.sum(weights / gammas**2) / np.sum(weights / gammas))
    active_gammas = gammas[active]
    m_scale = float(math.log10(np.max(active_gammas) / np.min(active_gammas)))
    m_res = float(np.sum(active) / m_scale) if m_scale > 0 else float("nan")
    h_mem = float(-np.sum(w[active] * np.log(w[active])))
    ecount = float(np.exp(h_mem))
    return {
        "gfe_m_cap_s": m_cap,
        "gfe_m_scale_decades": m_scale,
        "gfe_m_res_modes_per_decade": m_res,
        "gfe_h_mem_nats": h_mem,
        "gfe_entropy_effective_count": ecount,
        "gfe_d_eff": float(len(gammas) * ecount),
    }


def spectrum_metrics(eig: np.ndarray, prefix: str) -> dict:
    eig = np.sort(np.maximum(np.real(eig), 0.0))[::-1]
    total = float(np.sum(eig))
    if total <= 0:
        return {
            f"{prefix}_participation_dimension": 0.0,
            f"{prefix}_entropy_nats": 0.0,
            f"{prefix}_entropy_dimension": 0.0,
            f"{prefix}_rank_90": 0,
            f"{prefix}_rank_99": 0,
            f"{prefix}_condition_number": float("nan"),
            f"{prefix}_effective_rank_1e-12": 0,
        }
    p = eig / total
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    cdf = np.cumsum(p)
    positive = eig[eig > eig[0] * 1e-12]
    return {
        f"{prefix}_participation_dimension": float(1.0 / np.sum(p**2)),
        f"{prefix}_entropy_nats": entropy,
        f"{prefix}_entropy_dimension": float(np.exp(entropy)),
        f"{prefix}_rank_90": int(np.searchsorted(cdf, 0.90) + 1),
        f"{prefix}_rank_99": int(np.searchsorted(cdf, 0.99) + 1),
        f"{prefix}_condition_number": float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf"),
        f"{prefix}_effective_rank_1e-12": int(len(positive)),
    }


def basis_geometry(gammas: np.ndarray, weights: np.ndarray) -> dict:
    g = gammas[:, None] + gammas[None, :]
    gram = -np.expm1(-g * HORIZON_S) / g
    out = spectrum_metrics(np.linalg.eigvalsh(gram), "basis")
    # The unweighted dictionary geometry is a control. The environment enters
    # the realized kernel geometry through its fitted positive weights, so a
    # weighted Gram geometry is the environment-dependent basis descriptor.
    wgram = np.sqrt(weights)[:, None] * gram * np.sqrt(weights)[None, :]
    out.update(spectrum_metrics(np.linalg.eigvalsh(wgram), "weighted_basis"))
    diag = np.sqrt(np.maximum(np.diag(gram), 0.0))
    corr = gram / np.outer(np.maximum(diag, np.finfo(float).tiny),
                           np.maximum(diag, np.finfo(float).tiny))
    off = np.abs(corr - np.diag(np.diag(corr)))
    out["basis_max_coherence"] = float(np.max(off)) if off.size else 0.0
    return out


def exponential_states(x: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    poles = np.exp(-gammas * DT)
    increments = -np.expm1(-gammas * DT) / gammas
    q = np.zeros((x.size, gammas.size), dtype=complex)
    state = np.zeros(gammas.size, dtype=complex)
    for n, sample in enumerate(np.asarray(x, dtype=complex)):
        state = poles * state + increments * sample
        q[n] = state
    return q


def state_geometry(q: np.ndarray, weights: np.ndarray) -> dict:
    centered = q - np.mean(q, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(q.shape[0] - 1, 1)
    out = spectrum_metrics(np.linalg.eigvalsh(cov), "state")

    diag = np.sqrt(np.maximum(np.real(np.diag(cov)), 0.0))
    corr = cov / np.outer(np.maximum(diag, np.finfo(float).tiny),
                          np.maximum(diag, np.finfo(float).tiny))
    off = np.abs(corr - np.diag(np.diag(corr)))
    out["state_max_component_collinearity"] = float(np.max(off)) if off.size else 0.0
    energy = np.mean(np.abs(q) ** 2, axis=0)
    out["state_energy"] = float(np.sum(energy))
    out["state_rms_radius"] = float(np.sqrt(np.mean(np.sum(np.abs(centered) ** 2, axis=1))))
    out["state_trajectory_length"] = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))

    qw = q * np.sqrt(weights)[None, :]
    wc = qw - np.mean(qw, axis=0, keepdims=True)
    wcov = wc.conj().T @ wc / max(q.shape[0] - 1, 1)
    out.update(spectrum_metrics(np.linalg.eigvalsh(wcov), "weighted_state"))
    return out


def environment_metrics(t: np.ndarray, c: np.ndarray) -> dict:
    """Descriptors of the environment covariance geometry."""
    mass = float(np.trapz(c, t))
    first_moment = float(np.trapz(t * c, t))
    half = int(np.searchsorted(c <= 0.5, True))
    t_half = float(t[min(half, len(t) - 1)])
    area2 = float(np.trapz(c * c, t))
    return {
        "env_correlation_integral_s": mass,
        "env_correlation_centroid_s": first_moment / mass if mass > 0 else float("nan"),
        "env_half_decay_s": t_half,
        "env_l2_area_s": area2,
    }


def fit_metrics(t: np.ndarray, target: np.ndarray, fitted: np.ndarray) -> dict:
    residual = fitted - target
    rel_l2 = float(np.linalg.norm(residual) / max(np.linalg.norm(target), np.finfo(float).tiny))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - np.mean(target))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "kernel_fit_relative_l2": rel_l2,
        "kernel_fit_r2": r2,
        "kernel_fit_max_abs_error": float(np.max(np.abs(residual))),
    }


def run_case(signal_name, generator, environment_name, seed):
    fit_t = np.linspace(0.0, HORIZON_S, FIT_POINTS)
    target = environment_envelope(fit_t, environment_name)
    gammas, weights = fit_positive_soe(fit_t, target)
    fitted = np.exp(-np.outer(fit_t, gammas)) @ weights

    signal = generator(NUM_SYMBOLS, samples_per_symbol=SPS,
                       symbol_rate=SYMBOL_RATE, seed=seed)
    x = np.asarray(signal.record.samples, dtype=complex)
    q = exponential_states(x, gammas)

    row = {
        "experiment": "13C-1_environment_to_kernel_geometry",
        "signal": signal_name,
        "environment": environment_name,
        "seed": seed,
        "dictionary_mode_count": DICTIONARY_MODES,
        "sample_rate_hz": SAMPLE_RATE,
        "dt_s": DT,
        "horizon_s": HORIZON_S,
        "gamma_min_dictionary": float(np.min(gammas)),
        "gamma_max_dictionary": float(np.max(gammas)),
        "nonzero_kernel_modes": int(np.sum(weights > 1e-10)),
        "kernel_weight_min": float(np.min(weights)),
        "kernel_weight_max": float(np.max(weights)),
        "kernel_weight_entropy_nats": float(
            -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))
        ),
    }
    row.update(environment_metrics(fit_t, target))
    row.update(fit_metrics(fit_t, target, fitted))
    row.update(gfe_metrics(gammas, weights))
    row.update(basis_geometry(gammas, weights))
    row.update(state_geometry(q, weights))
    row["weighted_basis_to_state_entropy_ratio"] = (
        row["state_entropy_dimension"] / row["weighted_basis_entropy_dimension"]
        if row["weighted_basis_entropy_dimension"] > 0 else float("nan")
    )
    return row


def mean_summary(rows):
    groups = {}
    numeric = [
        "env_correlation_integral_s", "env_correlation_centroid_s", "env_half_decay_s",
        "kernel_fit_relative_l2", "kernel_fit_r2", "gfe_m_scale_decades",
        "gfe_m_res_modes_per_decade", "gfe_h_mem_nats", "gfe_entropy_effective_count",
        "gfe_d_eff", "basis_participation_dimension", "basis_entropy_dimension",
        "basis_max_coherence", "weighted_basis_participation_dimension",
        "weighted_basis_entropy_dimension", "state_participation_dimension",
        "state_entropy_dimension", "state_max_component_collinearity",
        "weighted_state_participation_dimension", "basis_to_state_entropy_ratio",
    ]
    for r in rows:
        groups.setdefault(r["environment"], []).append(r)
    out = []
    for env, bucket in sorted(groups.items()):
        item = {"environment": env, "cases": len(bucket)}
        for name in numeric:
            vals = np.asarray([float(r[name]) for r in bucket])
            item[name + "_mean"] = float(np.mean(vals))
            item[name + "_std"] = float(np.std(vals, ddof=1))
        out.append(item)
    return out


def main():
    rows = [
        run_case(name, gen, env, seed)
        for seed in SEEDS
        for name, gen in SIGNALS.items()
        for env in ENVIRONMENTS
    ]
    expected = len(SEEDS) * len(SIGNALS) * len(ENVIRONMENTS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(rows)}")

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "13C-1_environment_to_kernel_geometry_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = mean_summary(rows)
    summary_csv = out / "13C-1_environment_to_kernel_geometry_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "experiment": "13C-1_environment_to_kernel_geometry",
        "rows": len(rows),
        "grid": {
            "signals": list(SIGNALS),
            "environments": list(ENVIRONMENTS),
            "seeds": list(SEEDS),
            "dictionary_mode_count": DICTIONARY_MODES,
            "dictionary_gamma_range_s_inverse": [
                float(np.min(DICTIONARY_GAMMAS)),
                float(np.max(DICTIONARY_GAMMAS)),
            ],
        },
        "chain": [
            "environment covariance envelope",
            "positive SOE identification by fixed dictionary NNLS",
            "GFE/GGFE kernel descriptors",
            "finite-horizon SOE basis geometry",
            "signal-driven MIN state geometry",
        ],
        "interpretation_boundary": [
            "The environment envelope is an oracle covariance geometry in 13C-1; sample-estimation uncertainty is deferred.",
            "NNLS identifies a positive kernel representation and does not optimize a downstream task.",
            "No channel, receiver, BER/EVM, neural network, or task utility is evaluated.",
            "D_eff remains a kernel-complexity descriptor and is not equated with D_basis or D_state.",
        ],
        "outputs": [
            "13C-1_environment_to_kernel_geometry_results.csv",
            "13C-1_environment_to_kernel_geometry_summary.csv",
        ],
    }
    (out / "13C-1_environment_to_kernel_geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\\n"
    )

    print(json.dumps(summary, indent=2))
    for r in summary_rows:
        print(
            "environment", r["environment"],
            "fit_rel_l2=", r["kernel_fit_relative_l2_mean"],
            "D_eff=", r["gfe_d_eff_mean"],
            "D_basis=", r["basis_participation_dimension_mean"],
            "D_state=", r["state_participation_dimension_mean"],
        )


if __name__ == "__main__":
    main()
