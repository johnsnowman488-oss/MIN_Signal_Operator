"""Experiment 13B: Geometry -> State Structure.

Bridge GFE/GGFE kernel descriptors and empirical MIN temporal-state geometry.
Task-agnostic: no channels, noise, receivers, or downstream tasks.

Dimension labels:
  L = nominal SOE modes
  D_eff = GGFE kernel effective complexity
  D_basis = effective temporal-basis dimension
  D_state = effective signal-driven MIN state dimension
  D_task = reserved for later task-level experiments
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}
MODE_COUNTS = (1, 2, 4, 8, 16)
RATE_GEOMETRIES = ("clustered", "logspread", "wide")
WEIGHT_PATTERNS = ("uniform", "slow_dominant", "fast_dominant")
SEEDS = tuple(range(5))
NUM_SYMBOLS, SPS, SYMBOL_RATE = 512, 16, 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE
HORIZON_S = NUM_SYMBOLS / SYMBOL_RATE

def rate_pattern(L, geometry):
    if geometry == "clustered": return 10.0 * np.exp(np.linspace(-0.05, 0.05, L))
    if geometry == "logspread": return np.geomspace(2.0, 30.0, L)
    if geometry == "wide": return np.geomspace(0.5, 100.0, L)
    raise ValueError(geometry)

def weight_pattern(L, pattern):
    if L == 1: return np.ones(1)
    if pattern == "uniform": return np.full(L, 1.0 / L)
    w = np.full(L, 0.3 / (L - 1))
    if pattern == "slow_dominant": w[0] = 0.7
    elif pattern == "fast_dominant": w[-1] = 0.7
    else: raise ValueError(pattern)
    return w

def exponential_states(x, gammas, dt):
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

def gfe_metrics(gammas, weights):
    wt = weights / np.sum(weights)
    m_cap = float(np.sum(weights / gammas**2) / np.sum(weights / gammas))
    m_scale = float(math.log10(np.max(gammas) / np.min(gammas)))
    m_res = float(len(gammas) / m_scale) if m_scale > 0 else float("nan")
    h_mem = float(-np.sum(wt * np.log(wt)))
    ecount = float(np.exp(h_mem))
    return {"gfe_m_cap_s": m_cap, "gfe_m_scale_decades": m_scale,
            "gfe_m_res_modes_per_decade": m_res, "gfe_h_mem_nats": h_mem,
            "gfe_entropy_effective_count": ecount,
            "gfe_d_eff": float(len(gammas) * ecount)}

def spectrum_metrics(eig, prefix):
    eig = np.maximum(np.real(eig), 0.0)
    eig = np.sort(eig)[::-1]
    total = float(np.sum(eig))
    if total <= 0:
        return {f"{prefix}_trace": 0.0, f"{prefix}_participation_dimension": 0.0,
                f"{prefix}_entropy_nats": 0.0, f"{prefix}_entropy_dimension": 0.0,
                f"{prefix}_rank_90": 0, f"{prefix}_rank_99": 0,
                f"{prefix}_condition_number": float("nan"),
                f"{prefix}_effective_rank_1e-12": 0}
    p = eig / total
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    cdf = np.cumsum(p)
    rank90 = int(np.searchsorted(cdf, 0.90) + 1)
    rank99 = int(np.searchsorted(cdf, 0.99) + 1)
    positive = eig[eig > eig[0] * 1e-12]
    condition = float(eig[0] / eig[-1]) if eig[-1] > 0 else float("inf")
    return {f"{prefix}_trace": total,
            f"{prefix}_participation_dimension": float(1.0 / np.sum(p**2)),
            f"{prefix}_entropy_nats": entropy,
            f"{prefix}_entropy_dimension": float(np.exp(entropy)),
            f"{prefix}_rank_90": rank90, f"{prefix}_rank_99": rank99,
            f"{prefix}_condition_number": condition,
            f"{prefix}_effective_rank_1e-12": int(len(positive))}

def basis_geometry(gammas, horizon_s):
    g = gammas[:, None] + gammas[None, :]
    gram = -np.expm1(-g * horizon_s) / g
    eig = np.linalg.eigvalsh(gram)
    out = spectrum_metrics(eig, "basis")
    diag = np.sqrt(np.maximum(np.diag(gram), 0.0))
    corr = gram / np.outer(np.maximum(diag, np.finfo(float).tiny),
                           np.maximum(diag, np.finfo(float).tiny))
    off = np.abs(corr - np.diag(np.diag(corr)))
    out["basis_max_coherence"] = float(np.max(off)) if off.size else 0.0
    out["basis_min_eigenvalue"] = float(np.min(np.maximum(eig, 0.0)))
    out["basis_max_eigenvalue"] = float(np.max(np.maximum(eig, 0.0)))
    return out

def state_geometry(q, weights):
    centered = q - np.mean(q, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(centered.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    out = spectrum_metrics(eig, "state")
    diag = np.sqrt(np.maximum(np.real(np.diag(cov)), 0.0))
    corr = cov / np.outer(np.maximum(diag, np.finfo(float).tiny),
                          np.maximum(diag, np.finfo(float).tiny))
    off = np.abs(corr - np.diag(np.diag(corr)))
    out["state_max_component_collinearity"] = float(np.max(off)) if off.size else 0.0
    energy = np.mean(np.abs(q) ** 2, axis=0)
    out["state_energy"] = float(np.sum(energy))
    out["state_rms_radius"] = float(np.sqrt(np.mean(np.sum(np.abs(centered) ** 2, axis=1))))
    out["state_trajectory_length"] = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    out["state_max_component_energy"] = float(np.max(energy))
    out["state_min_component_energy"] = float(np.min(energy))
    qw = q * np.sqrt(weights)[None, :]
    wc = qw - np.mean(qw, axis=0, keepdims=True)
    wcov = wc.conj().T @ wc / max(wc.shape[0] - 1, 1)
    out.update(spectrum_metrics(np.linalg.eigvalsh(wcov), "weighted_state"))
    return out

def rank_values(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    return r

def corr_pair(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return {"n": int(len(x)), "pearson_r": float(np.corrcoef(x, y)[0,1]),
            "spearman_rho": float(np.corrcoef(rank_values(x), rank_values(y))[0,1])}

def relationship(rows, left, right):
    return corr_pair(np.asarray([float(r[left]) for r in rows]),
                     np.asarray([float(r[right]) for r in rows]))

def run_case(signal_name, generator, L, geometry, weight_name, seed):
    signal = generator(NUM_SYMBOLS, samples_per_symbol=SPS, symbol_rate=SYMBOL_RATE, seed=seed)
    x = np.asarray(signal.record.samples, dtype=complex)
    gammas = rate_pattern(L, geometry)
    weights = weight_pattern(L, weight_name)
    q = exponential_states(x, gammas, DT)
    row = {"experiment": "13B_geometry_to_state_structure", "signal": signal_name,
           "seed": seed, "mode_count": L, "rate_geometry": geometry,
           "weight_pattern": weight_name, "gamma_min": float(np.min(gammas)),
           "gamma_max": float(np.max(gammas)),
           "gamma_geometric_span": float(np.max(gammas)/np.min(gammas)),
           "weight_min": float(np.min(weights)), "weight_max": float(np.max(weights)),
           "sample_rate_hz": SAMPLE_RATE, "dt_s": DT, "horizon_s": HORIZON_S}
    row.update(gfe_metrics(gammas, weights))
    row.update(basis_geometry(gammas, HORIZON_S))
    row.update(state_geometry(q, weights))
    row["basis_to_state_entropy_ratio"] = (row["state_entropy_dimension"] /
        row["basis_entropy_dimension"] if row["basis_entropy_dimension"] > 0 else float("nan"))
    row["state_to_basis_participation_ratio"] = (row["state_participation_dimension"] /
        row["basis_participation_dimension"] if row["basis_participation_dimension"] > 0 else float("nan"))
    return row

def main():
    rows = [run_case(name, gen, L, geo, wp, seed)
            for seed in SEEDS for name, gen in SIGNALS.items()
            for L in MODE_COUNTS for geo in RATE_GEOMETRIES
            for wp in WEIGHT_PATTERNS]
    if len(rows) != 675: raise RuntimeError(f"Expected 675 rows, got {len(rows)}")
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "13B_geometry_to_state_structure_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    relationships = {
        "M_scale_vs_basis_participation": relationship(rows, "gfe_m_scale_decades", "basis_participation_dimension"),
        "M_scale_vs_basis_entropy_dimension": relationship(rows, "gfe_m_scale_decades", "basis_entropy_dimension"),
        "M_scale_vs_basis_coherence": relationship(rows, "gfe_m_scale_decades", "basis_max_coherence"),
        "M_scale_vs_state_participation": relationship(rows, "gfe_m_scale_decades", "state_participation_dimension"),
        "M_scale_vs_state_entropy_dimension": relationship(rows, "gfe_m_scale_decades", "state_entropy_dimension"),
        "D_eff_vs_state_participation": relationship(rows, "gfe_d_eff", "state_participation_dimension"),
        "D_eff_vs_state_entropy_dimension": relationship(rows, "gfe_d_eff", "state_entropy_dimension"),
        "basis_participation_vs_state_participation": relationship(rows, "basis_participation_dimension", "state_participation_dimension"),
        "basis_coherence_vs_state_participation": relationship(rows, "basis_max_coherence", "state_participation_dimension"),
        "basis_condition_vs_state_participation": relationship(rows, "basis_condition_number", "state_participation_dimension"),
    }
    grouped = {}
    for r in rows:
        grouped.setdefault((r["mode_count"], r["rate_geometry"], r["weight_pattern"]), []).append(r)
    metrics = ["gfe_d_eff","gfe_m_scale_decades","basis_participation_dimension",
               "basis_entropy_dimension","basis_max_coherence","basis_condition_number",
               "basis_effective_rank_1e-12","state_participation_dimension",
               "state_entropy_dimension","state_max_component_collinearity",
               "state_condition_number","state_rank_99","weighted_state_participation_dimension"]
    summary_rows = []
    for key, bucket in sorted(grouped.items(), key=lambda z: z[0]):
        sr = {"mode_count": key[0], "rate_geometry": key[1], "weight_pattern": key[2], "n": len(bucket)}
        for m in metrics:
            v = np.asarray([float(r[m]) for r in bucket])
            sr[m+"_mean"] = float(np.nanmean(v)); sr[m+"_std"] = float(np.nanstd(v, ddof=1))
        summary_rows.append(sr)
    summary = {
        "experiment": "13B_geometry_to_state_structure", "rows": len(rows),
        "grid": {"signals": list(SIGNALS), "mode_counts": list(MODE_COUNTS),
                 "rate_geometries": list(RATE_GEOMETRIES), "weight_patterns": list(WEIGHT_PATTERNS),
                 "seeds": list(SEEDS)}, "horizon_s": HORIZON_S,
        "definitions": {
            "basis_gram": "G_ij=(1-exp(-(gamma_i+gamma_j)T))/(gamma_i+gamma_j)",
            "basis_participation_dimension": "1/sum(p_i^2), p_i=Gram eigenvalue_i/sum(eigenvalues)",
            "basis_entropy_dimension": "exp(-sum(p_i log p_i))",
            "basis_max_coherence": "maximum absolute off-diagonal normalized Gram entry",
            "state_participation_dimension": "participation ratio of centered complex state covariance",
            "state_entropy_dimension": "exp of normalized covariance spectral entropy",
            "D_eff": "L*exp(H_mem), with H_mem from normalized positive SOE weights"},
        "dimension_hierarchy": ["L = nominal SOE modes", "D_eff = GFE/GGFE kernel effective complexity",
                                "D_basis = effective temporal-basis dimension", "D_state = signal-driven MIN state dimension",
                                "D_task = reserved for task-level evaluation"],
        "relationships": relationships,
        "interpretation_boundary": [
            "Task-agnostic forward representation experiment.",
            "D_eff is not assumed equal to D_basis or D_state.",
            "Basis geometry is the intermediate mechanism between spectral descriptors and signal-driven state geometry.",
            "No d_s is assigned to this finite positive SOE atlas."] }
    (out/"13B_geometry_to_state_structure_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True)+"\n")
    with (out/"13B_geometry_to_state_structure_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)
    print(json.dumps(summary, indent=2, allow_nan=True))

if __name__ == "__main__":
    main()
