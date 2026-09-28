"""Experiment 15B-2: explicit process-specific temporal alignment intervention.

For each generating process, compare:
  matched   : SOE rates concentrated around the process decay rate(s)
  displaced: same 16 positive uniform modes, but rate concentrations shifted
  broad     : wide log-spaced support control

L=16, uniform weights, D_eff=256, observation, split, PCA and readout are held
fixed within each paired case. The experiment records state conditioning so
ill-conditioned representations are not mistaken for alignment effects.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("min_exp15_utils", ROOT / "experiments" / "15_utils.py")
U = module_from_spec(spec); sys.modules[spec.name] = U; spec.loader.exec_module(U)

PROCESSES = ("short", "multiscale", "long", "hidden_mix")
TASKS = ("prediction", "latent_estimation")
ROLES = U.ALIGNMENT_ROLES
SEEDS = U.SEEDS
SNR_DB = U.SNR_DB
PCA_DIMENSIONS = U.PCA_DIMENSIONS
MODES = U.MODES
DT = U.DT
TASK_TOLERANCE = U.TASK_TOLERANCE
N = 8192
TRAIN_FRAC = 0.5
NOISE_SEED_OFFSET = 151_000

PROCESS_PARAMS = {
    "short": {"taus": (0.025,), "weights": (1.0,)},
    "multiscale": {"taus": (0.025, 0.50), "weights": (0.65, 0.35)},
    "long": {"taus": (0.50,), "weights": (1.0,)},
    "hidden_mix": {"taus": (0.025, 0.50), "weights": (0.50, 0.50)},
}

def ar_component(n, tau, rng):
    a = float(np.exp(-DT / tau))
    eps = rng.normal(size=n)
    q = np.empty(n); q[0] = eps[0]
    scale = np.sqrt(max(1.0 - a * a, np.finfo(float).tiny))
    for i in range(1, n):
        q[i] = a * q[i - 1] + scale * eps[i]
    return q

def generate_process(name, seed):
    p = PROCESS_PARAMS[name]
    rng = np.random.default_rng(seed)
    components = np.column_stack([ar_component(N, tau, rng) for tau in p["taus"]])
    w = np.asarray(p["weights"], dtype=float); w /= w.sum()
    x = components @ w
    x /= max(float(np.std(x)), np.finfo(float).tiny)
    return x.astype(float), components.astype(float)

def add_observation_noise(x, snr_db, seed):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(float(np.mean(x*x)) / (10.0 ** (snr_db / 10.0)))
    return x + sigma * rng.normal(size=len(x))

def state_condition_number(X):
    centered = X - np.mean(X, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    positive = singular[singular > np.finfo(float).eps * max(singular[0], 1.0)]
    return float(singular[0] / positive[-1]) if positive.size else float("inf")

def fit_eval(Xtr, ytr, Xte, yte):
    mean, singular, components = U.complex_pca_fit(Xtr.astype(complex))
    curve = {}
    for k in PCA_DIMENSIONS:
        ztr = U.pca_project(Xtr.astype(complex), mean, components, k)
        zte = U.pca_project(Xte.astype(complex), mean, components, k)
        beta = U.linear_readout_fit(ztr, ytr.astype(complex))
        pred = U.linear_readout_predict(zte, beta)
        curve[k] = U.nmse(yte, pred)
    full = curve[MODES]
    d_task = next((k for k in PCA_DIMENSIONS if curve[k] <= TASK_TOLERANCE * full), None)
    return curve, singular, d_task

def run_case(process, snr_db, seed):
    x, latent = generate_process(process, seed)
    y = add_observation_noise(x, snr_db, seed + NOISE_SEED_OFFSET)
    split = int(N * TRAIN_FRAC)
    rows, curves = [], []

    for task in TASKS:
        for role in ROLES:
            spec = U.ALIGNMENT_KERNELS[process][role]
            q = U.exact_piecewise_linear_states(y, spec["gammas"], spec["coefficients"])
            X = q[:-1] if task == "prediction" else q
            target = x[1:] if task == "prediction" else latent[:, 0]
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = target[:split], target[split:]
            curve, singular, d_task = fit_eval(Xtr, ytr, Xte, yte)
            cond = state_condition_number(Xtr)
            desc = U.ALIGNMENT_KERNEL_DESCRIPTORS[process][role]
            for k, value in curve.items():
                curves.append({
                    "experiment": "15B2_process_specific_alignment",
                    "process": process, "task": task, "role": role,
                    "snr_db": float(snr_db), "seed": int(seed),
                    "retained_pca_dimension": int(k),
                    "heldout_nmse": float(value),
                    "full_state_reference": int(k == MODES),
                    "pca_explained_energy": U.pca_energy(singular, k),
                    "state_condition_number": cond,
                    **desc,
                })
            rows.append({
                "experiment": "15B2_process_specific_alignment",
                "process": process, "task": task, "role": role,
                "snr_db": float(snr_db), "seed": int(seed),
                "full_state_nmse": float(curve[MODES]),
                "d_task_10pct": d_task if d_task is not None else -1,
                "state_condition_number": cond,
                **desc,
            })
    return rows, curves

def summarize(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault((r["process"], r["task"], r["role"], r["snr_db"]), []).append(r)
    out = []
    for key, b in sorted(grouped.items()):
        p, t, role, snr = key
        d = [r["d_task_10pct"] for r in b if r["d_task_10pct"] > 0]
        out.append({
            "process": p, "task": t, "role": role, "snr_db": float(snr), "n": len(b),
            "mean_full_state_nmse": float(np.mean([r["full_state_nmse"] for r in b])),
            "std_full_state_nmse": float(np.std([r["full_state_nmse"] for r in b], ddof=1)),
            "mean_d_task": float(np.mean(d)) if d else float("nan"),
            "mean_state_condition_number": float(np.mean([r["state_condition_number"] for r in b])),
            "median_state_condition_number": float(np.median([r["state_condition_number"] for r in b])),
            "mean_gfe_d_eff": float(np.mean([r["gfe_d_eff"] for r in b])),
        })
    return out

def main():
    eq = U.validate_repository_equivalence()
    rows, curves = [], []
    for seed in SEEDS:
        for process in PROCESSES:
            for snr in SNR_DB:
                r, c = run_case(process, snr, seed)
                rows.extend(r); curves.extend(c)

    expected_cases = len(SEEDS)*len(PROCESSES)*len(SNR_DB)
    expected_rows = expected_cases*len(TASKS)*len(ROLES)
    assert len(rows) == expected_rows
    assert len(curves) == expected_rows*len(PCA_DIMENSIONS)

    out = ROOT / "experiments" / "results"; out.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": out/"15B2_process_specific_alignment_cases.csv",
        "pca": out/"15B2_process_specific_alignment_pca_results.csv",
        "summary": out/"15B2_process_specific_alignment_summary.csv",
        "metadata": out/"15B2_process_specific_alignment_summary.json",
    }
    for key in ("cases", "pca"):
        data = rows if key == "cases" else curves
        with paths[key].open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary = summarize(rows)
    with paths["summary"].open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta = {
        "experiment": "15B2_process_specific_alignment",
        "purpose": "Isolate temporal alignment by matching or displacing SOE decay-rate support relative to the generating process.",
        "processes": PROCESSES, "tasks": TASKS, "roles": ROLES,
        "snr_db": list(SNR_DB), "seeds": list(SEEDS), "N": N,
        "nominal_modes": MODES, "pca_dimensions": list(PCA_DIMENSIONS),
        "controls": {
            "same_mode_count": True, "same_uniform_weights": True, "same_d_eff": True,
            "same_observation_per_paired_case": True, "same_train_test_split": True,
            "same_pca_and_readout": True, "readout_ridge": U.READOUT_RIDGE,
        },
        "process_parameters": PROCESS_PARAMS,
        "repository_state_equivalence_relative_error": eq,
        "rows": len(rows), "pca_rows": len(curves),
    }
    paths["metadata"].write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
