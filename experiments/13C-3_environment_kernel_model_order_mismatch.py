"""Experiment 13C-3: environment -> kernel geometry under SOE dictionary mismatch."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
ENVIRONMENTS = ("white_limit", "short", "multiscale", "powerlaw", "squared_exp")
SEEDS = tuple(range(4))
NUM_SYMBOLS, SPS, SYMBOL_RATE = 512, 16, 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1 / SAMPLE_RATE
HORIZON_S = NUM_SYMBOLS / SYMBOL_RATE
FIT_HORIZON_S = 0.08

# Reference dictionary matches the 13C-1/13C-2 16-mode geometry.
DICTIONARIES = {
    "under_low_8": np.geomspace(0.5, 10.0, 8),
    "under_high_8": np.geomspace(5.0, 100.0, 8),
    "matched_16": np.geomspace(0.5, 100.0, 16),
    "over_dense_32": np.geomspace(0.25, 200.0, 32),
    "shifted_16": np.geomspace(0.75, 150.0, 16),
}
REFERENCE_GAMMAS = DICTIONARIES["matched_16"]


def env(t, name):
    if name == "white_limit":
        return np.exp(-t / 0.005)
    if name == "short":
        return np.exp(-t / 0.05)
    if name == "multiscale":
        return 0.65 * np.exp(-t / 0.03) + 0.35 * np.exp(-t / 0.7)
    if name == "powerlaw":
        return (1 + t / 0.2) ** (-0.7)
    if name == "squared_exp":
        return np.exp(-(t / 0.15) ** 2)
    raise ValueError(name)


def fit(t, y, gammas):
    A = np.exp(-np.outer(t, gammas))
    w, _ = nnls(A, np.maximum(y, 0))
    w /= max(w.sum(), 1e-300)
    return w


def eigmetrics(a, prefix):
    e = np.sort(np.maximum(np.real(np.linalg.eigvalsh(a)), 0))[::-1]
    s = e.sum()
    if s <= 0:
        return {prefix + "_participation_dimension": 0.0,
                prefix + "_entropy_dimension": 0.0,
                prefix + "_rank90": 0, prefix + "_rank99": 0}
    q = e / s
    return {
        prefix + "_participation_dimension": float(1 / (q @ q)),
        prefix + "_entropy_dimension": float(np.exp(-np.sum(q[q > 0] * np.log(q[q > 0])))),
        prefix + "_rank90": int(np.searchsorted(np.cumsum(q), 0.90) + 1),
        prefix + "_rank99": int(np.searchsorted(np.cumsum(q), 0.99) + 1),
    }


def kernel_metrics(w, gammas, horizon=HORIZON_S):
    active = w > 1e-12
    h = -np.sum(w[active] * np.log(w[active]))
    g = gammas[active]
    scale = math.log10(g.max() / g.min()) if len(g) > 1 else 0.0
    z = gammas[:, None] + gammas[None, :]
    G = -np.expm1(-z * horizon) / z
    W = np.sqrt(w)[:, None] * G * np.sqrt(w)[None, :]
    out = {
        "gfe_m_cap_s": float(np.sum(w / gammas**2) / np.sum(w / gammas)),
        "gfe_m_scale_decades": float(scale),
        "gfe_m_res_modes_per_decade": float(active.sum() / scale) if scale else float("nan"),
        "gfe_h_mem_nats": float(h),
        "gfe_entropy_effective_count": float(np.exp(h)),
        # D_eff is the GGFE documented L exp(H_mem), not a geometric rank.
        "gfe_d_eff": float(len(gammas) * np.exp(h)),
        "nonzero_kernel_modes": int(active.sum()),
    }
    out.update(eigmetrics(W, "weighted_basis"))
    diag = np.sqrt(np.maximum(np.diag(W), 0))
    C = W / np.outer(np.maximum(diag, 1e-300), np.maximum(diag, 1e-300))
    off = ~np.eye(len(gammas), dtype=bool)
    out["weighted_basis_max_coherence"] = float(np.max(np.abs(C[off]))) if np.any(off) else 0.0

    # Normalized Gram separates mode distinguishability from unequal finite-horizon
    # mode energy, which is important when fast modes have small L2 norm.
    Gn = G / np.outer(np.sqrt(np.maximum(np.diag(G), 1e-300)),
                      np.sqrt(np.maximum(np.diag(G), 1e-300)))
    out.update(eigmetrics(Gn, "normalized_basis"))
    return out


def states(x, gammas):
    p = np.exp(-gammas * DT)
    inc = -np.expm1(-gammas * DT) / gammas
    q = np.zeros(len(gammas), complex)
    out = np.empty((len(x), len(gammas)), complex)
    for i, v in enumerate(x.astype(complex)):
        q = p * q + inc * v
        out[i] = q
    return out


def stategeom(q, w):
    z = q - q.mean(0)
    C = z.conj().T @ z / max(len(q) - 1, 1)
    out = eigmetrics(C, "state")
    zw = z * np.sqrt(w)
    Cw = zw.conj().T @ zw / max(len(q) - 1, 1)
    out.update(eigmetrics(Cw, "weighted_state"))
    return out


def synth(name, n, seed):
    rng = np.random.default_rng(seed)
    c = env(np.arange(n) * DT, name)
    circ = np.r_[c, c[-2:0:-1]]
    s = np.maximum(np.real(np.fft.rfft(circ)), 0)
    z = rng.normal(size=s.size) + 1j * rng.normal(size=s.size)
    z[0] = rng.normal()
    if circ.size % 2 == 0:
        z[-1] = rng.normal()
    x = np.fft.irfft(np.sqrt(s) * z, n=circ.size)[:n]
    return x / max(np.std(x), 1e-300)


def oracle_environment(name):
    t = np.linspace(0, FIT_HORIZON_S, 2048)
    w = fit(t, env(t, name), REFERENCE_GAMMAS)
    return w, kernel_metrics(w, REFERENCE_GAMMAS)


def main():
    # The reference is always the 16-mode matched dictionary. Each mismatch
    # dictionary is evaluated against that same environment and the same
    # signal probe, so dictionary effects are not confused with task changes.
    ORACLE = {e: oracle_environment(e) for e in ENVIRONMENTS}
    rows = []

    for e in ENVIRONMENTS:
        ref_w, ref_geom = ORACLE[e]
        for seed in SEEDS:
            latent = synth(e, 512, seed)
            sigs = {}
            for signal_name, generator in SIGNALS.items():
                sig = generator(NUM_SYMBOLS, samples_per_symbol=SPS,
                                symbol_rate=SYMBOL_RATE, seed=seed)
                sigs[signal_name] = np.asarray(sig.record.samples, dtype=complex)

            t = np.linspace(0, FIT_HORIZON_S, 1024)
            c = env(t, e)
            for dict_name, gammas in DICTIONARIES.items():
                w = fit(t, c, gammas)
                fitted = np.exp(-np.outer(t, gammas)) @ w
                geom = kernel_metrics(w, gammas)
                # Project the mismatched kernel onto the reference-rate dictionary
                # only for a comparable coefficient-error diagnostic; geometry and
                # state metrics remain native to each dictionary.
                w_ref_on_native = fit(t, c, REFERENCE_GAMMAS)
                kernel_l2 = float(np.linalg.norm(fitted - c) /
                                  max(np.linalg.norm(c), 1e-300))
                row_base = {
                    "experiment": "13C-3_environment_kernel_model_order_mismatch",
                    "environment": e, "seed": seed,
                    "dictionary": dict_name, "dictionary_mode_count": len(gammas),
                    "gamma_min_s_inverse": float(gammas.min()),
                    "gamma_max_s_inverse": float(gammas.max()),
                    "kernel_fit_relative_l2": kernel_l2,
                    "kernel_weight_relative_l2_to_reference": float(
                        np.linalg.norm(w_ref_on_native - ref_w) /
                        max(np.linalg.norm(ref_w), 1e-300)),
                }
                row_base.update(geom)
                row_base["reference_d_eff"] = ref_geom["gfe_d_eff"]
                row_base["reference_weighted_basis_participation_dimension"] = ref_geom[
                    "weighted_basis_participation_dimension"]
                row_base["d_eff_relative_error"] = abs(
                    geom["gfe_d_eff"] - ref_geom["gfe_d_eff"]) / max(
                        abs(ref_geom["gfe_d_eff"]), 1e-300)
                row_base["weighted_basis_relative_error"] = abs(
                    geom["weighted_basis_participation_dimension"] -
                    ref_geom["weighted_basis_participation_dimension"]) / max(
                        ref_geom["weighted_basis_participation_dimension"], 1e-300)

                ref_q = states(sigs["BPSK"], ref_w)
                for signal_name, signal in sigs.items():
                    q = states(signal, gammas)
                    sg = stategeom(q, w)
                    rq = stategeom(ref_q if signal_name == "BPSK"
                                   else states(signal, REFERENCE_GAMMAS), ref_w)
                    r = dict(row_base)
                    r["signal"] = signal_name
                    r.update(sg)
                    r["reference_weighted_state_participation_dimension"] = rq[
                        "weighted_state_participation_dimension"]
                    r["reference_weighted_state_entropy_dimension"] = rq[
                        "weighted_state_entropy_dimension"]
                    r["weighted_state_relative_error"] = abs(
                        r["weighted_state_participation_dimension"] -
                        rq["weighted_state_participation_dimension"]) / max(
                        rq["weighted_state_participation_dimension"], 1e-300)
                    rows.append(r)

    expected = len(ENVIRONMENTS) * len(SEEDS) * len(DICTIONARIES) * len(SIGNALS)
    assert len(rows) == expected, (len(rows), expected)

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "13C-3_environment_kernel_model_order_mismatch_results.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    group_keys = ("environment", "dictionary")
    metrics = (
        "kernel_fit_relative_l2", "gfe_d_eff",
        "weighted_basis_participation_dimension",
        "normalized_basis_participation_dimension",
        "weighted_state_participation_dimension",
        "weighted_state_entropy_dimension", "d_eff_relative_error",
        "weighted_basis_relative_error", "weighted_state_relative_error",
    )
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    summary = []
    for key, bucket in sorted(groups.items()):
        d = dict(zip(group_keys, key))
        d["rows"] = len(bucket)
        d["independent_environment_seeds"] = len(SEEDS)
        for m in metrics:
            vals = [r[m] for r in bucket]
            d[m + "_mean"] = float(np.mean(vals))
            d[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(d)

    summary_path = out / "13C-3_environment_kernel_model_order_mismatch_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "experiment": "13C-3_environment_kernel_model_order_mismatch",
        "rows": len(rows),
        "unique_environment_probes": len(ENVIRONMENTS) * len(SEEDS),
        "grid": {
            "environments": list(ENVIRONMENTS), "seeds": list(SEEDS),
            "signals": list(SIGNALS), "dictionaries": {
                k: [float(v.min()), float(v.max()), len(v)]
                for k, v in DICTIONARIES.items()
            },
        },
        "primary_comparison":
            "matched 16-mode reference versus under-specified, over-dense, and shifted rate dictionaries",
        "metrics": [
            "kernel fit fidelity", "M_cap", "M_scale", "M_res", "H_mem",
            "D_eff", "weighted basis geometry", "normalized Gram geometry",
            "weighted MIN state geometry",
        ],
        "dimension_hierarchy":
            "L != D_eff != D_basis != D_state; D_eff is entropy-effective count, not realized state rank",
        "controls": [
            "same environment realization seed across dictionaries",
            "same signal probe seed across dictionaries",
            "task-agnostic protocol",
            "normalized Gram control separates distinguishability from unequal mode energy",
        ],
        "boundary": [
            "fixed SOE dictionary families; adaptive rate learning is deferred",
            "oracle environment covariance is used here to isolate dictionary mismatch from finite-observation uncertainty",
            "no BER, EVM, receiver, neural-network, or downstream task metric",
        ],
    }
    (out / "13C-3_environment_kernel_model_order_mismatch_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
