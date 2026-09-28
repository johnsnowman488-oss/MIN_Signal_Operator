"""Experiment 15B: temporal-process structure x memory geometry.

15B asks whether the geometry effect from 15A becomes functionally useful when
the SOE decay-rate geometry is aligned or misaligned with the temporal scales
of the generating process.

The experiment uses controlled latent processes rather than communication
waveforms:
  short       : one fast AR(1) component
  multiscale  : fast + slow latent components
  long        : one slow AR(1) component
  hidden_mix  : observed mixture of separated latent AR components

For every process, the same noisy observation is passed through clustered,
logspread, and wide 16-mode SOE representations. We evaluate causal prediction
and latent-state estimation with the same PCA/readout protocol.

No 14-series experiment module is imported.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path(__file__).resolve().parents[1]
UTIL_SPEC = spec_from_file_location("min_exp15_utils", ROOT / "experiments" / "15_utils.py")
UTIL = module_from_spec(UTIL_SPEC)
sys.modules[UTIL_SPEC.name] = UTIL
UTIL_SPEC.loader.exec_module(UTIL)

GEOMETRIES = UTIL.GEOMETRIES
SEEDS = UTIL.SEEDS
SNR_DB = UTIL.SNR_DB
PCA_DIMENSIONS = UTIL.PCA_DIMENSIONS
MODES = UTIL.MODES
DT = UTIL.DT
TASK_TOLERANCE = UTIL.TASK_TOLERANCE
READOUT_RIDGE = UTIL.READOUT_RIDGE

PROCESSES = ("short", "multiscale", "long", "hidden_mix")
TASKS = ("prediction", "latent_estimation")
N = 8192
TRAIN_FRAC = 0.5
NOISE_SEED_OFFSET = 150_000

# Timescales are deliberately chosen relative to the SOE rate families.
# The exact values are part of the experiment metadata.
PROCESS_PARAMS = {
    "short": {"taus": (0.025,), "weights": (1.0,)},
    "multiscale": {"taus": (0.025, 0.50), "weights": (0.65, 0.35)},
    "long": {"taus": (0.50,), "weights": (1.0,)},
    "hidden_mix": {"taus": (0.025, 0.50), "weights": (0.50, 0.50)},
}


def ar_component(n, tau, rng):
    a = float(np.exp(-DT / tau))
    eps = rng.normal(size=n)
    q = np.empty(n)
    q[0] = eps[0]
    scale = np.sqrt(max(1.0 - a * a, np.finfo(float).tiny))
    for i in range(1, n):
        q[i] = a * q[i - 1] + scale * eps[i]
    return q


def generate_process(name, seed):
    p = PROCESS_PARAMS[name]
    rng = np.random.default_rng(seed)
    components = np.column_stack([
        ar_component(N, tau, rng) for tau in p["taus"]
    ])
    w = np.asarray(p["weights"], dtype=float)
    w /= w.sum()
    x = components @ w
    x /= max(float(np.std(x)), np.finfo(float).tiny)
    return x.astype(float), components.astype(float)


def add_observation_noise(x, snr_db, seed):
    rng = np.random.default_rng(seed)
    power = float(np.mean(x * x))
    sigma = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return x + sigma * rng.normal(size=len(x))


def task_arrays(x, q, components, split, task):
    if task == "prediction":
        return q[:-1], x[1:]
    if task == "latent_estimation":
        # Recover the first latent component. For single-component processes
        # this is the complete latent state; for mixed processes it tests
        # access to one constituent temporal scale.
        target = components[:, 0]
        return q, target
    raise ValueError(task)


def fit_eval(Xtr, ytr, Xte, yte):
    mean, singular, components = UTIL.complex_pca_fit(Xtr.astype(complex))
    curve = {}
    for k in PCA_DIMENSIONS:
        ztr = UTIL.pca_project(Xtr.astype(complex), mean, components, k)
        zte = UTIL.pca_project(Xte.astype(complex), mean, components, k)
        beta = UTIL.linear_readout_fit(ztr, ytr.astype(complex))
        pred = UTIL.linear_readout_predict(zte, beta)
        curve[k] = UTIL.nmse(yte, pred)
    full = curve[MODES]
    d_task = next((k for k in PCA_DIMENSIONS
                   if curve[k] <= TASK_TOLERANCE * full), None)
    return curve, singular, UTIL.spectrum_geometry(Xtr.astype(complex)), d_task


def run_case(process, snr_db, seed):
    x, latent = generate_process(process, seed)
    y = add_observation_noise(x, snr_db, seed + NOISE_SEED_OFFSET)
    split = int(N * TRAIN_FRAC)

    rows, curves = [], []
    for task in TASKS:
        for geometry in GEOMETRIES:
            spec = UTIL.KERNELS[geometry]
            q = UTIL.exact_piecewise_linear_states(
                y, spec["gammas"], spec["coefficients"]
            )
            if task == "prediction":
                X = q[:-1]
                target = x[1:]
            else:
                X = q
                target = latent[:, 0]

            # Same temporal split for every geometry.
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = target[:split], target[split:]
            curve, singular, state_geom, d_task = fit_eval(
                Xtr, ytr, Xte, yte
            )
            for k, value in curve.items():
                curves.append({
                    "experiment": "15B_process_geometry_alignment",
                    "process": process, "task": task,
                    "geometry": geometry, "snr_db": float(snr_db),
                    "seed": int(seed), "retained_pca_dimension": int(k),
                    "heldout_nmse": float(value),
                    "full_state_reference": int(k == MODES),
                    "pca_explained_energy": UTIL.pca_energy(singular, k),
                    **UTIL.KERNEL_DESCRIPTORS[geometry], **state_geom,
                })
            rows.append({
                "experiment": "15B_process_geometry_alignment",
                "process": process, "task": task,
                "geometry": geometry, "snr_db": float(snr_db),
                "seed": int(seed),
                "full_state_nmse": float(curve[MODES]),
                "d_task_10pct": d_task if d_task is not None else -1,
                **UTIL.KERNEL_DESCRIPTORS[geometry], **state_geom,
            })
    return rows, curves


def summarize(rows):
    grouped = {}
    for r in rows:
        key = (r["process"], r["task"], r["geometry"], r["snr_db"])
        grouped.setdefault(key, []).append(r)
    out = []
    for key, b in sorted(grouped.items()):
        process, task, geometry, snr = key
        d = [r["d_task_10pct"] for r in b if r["d_task_10pct"] > 0]
        out.append({
            "process": process, "task": task, "geometry": geometry,
            "snr_db": float(snr), "n": len(b),
            "mean_full_state_nmse": float(np.mean([r["full_state_nmse"] for r in b])),
            "std_full_state_nmse": float(np.std([r["full_state_nmse"] for r in b], ddof=1)),
            "mean_d_task": float(np.mean(d)) if d else float("nan"),
            "std_d_task": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "mean_state_participation_dimension": float(
                np.mean([r["state_participation_dimension"] for r in b])
            ),
        })
    return out


def main():
    eq = UTIL.validate_repository_equivalence()
    rows, curves = [], []
    for seed in SEEDS:
        for process in PROCESSES:
            for snr in SNR_DB:
                r, c = run_case(process, snr, seed)
                rows.extend(r); curves.extend(c)

    expected_cases = len(SEEDS) * len(PROCESSES) * len(SNR_DB)
    expected_rows = expected_cases * len(TASKS) * len(GEOMETRIES)
    expected_curves = expected_rows * len(PCA_DIMENSIONS)
    assert len(rows) == expected_rows
    assert len(curves) == expected_curves

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": out / "15B_process_geometry_alignment_cases.csv",
        "pca": out / "15B_process_geometry_alignment_pca_results.csv",
        "summary": out / "15B_process_geometry_alignment_summary.csv",
        "metadata": out / "15B_process_geometry_alignment_summary.json",
    }
    for key, path in (("cases", paths["cases"]), ("pca", paths["pca"])):
        data = rows if key == "cases" else curves
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    summary = summarize(rows)
    with paths["summary"].open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    meta = {
        "experiment": "15B_process_geometry_alignment",
        "purpose": "Test whether memory-kernel geometry becomes functionally useful when its timescales align with temporal scales in the generating process.",
        "processes": PROCESSES, "tasks": TASKS, "geometries": GEOMETRIES,
        "snr_db": list(SNR_DB), "seeds": list(SEEDS), "N": N,
        "nominal_modes": MODES, "pca_dimensions": list(PCA_DIMENSIONS),
        "process_parameters": PROCESS_PARAMS,
        "controls": {
            "same_observation_per_paired_geometry_case": True,
            "same_train_test_split": True,
            "same_pca_and_readout": True,
            "matched_geometry_mode_count_and_d_eff": True,
        },
        "repository_state_equivalence_relative_error": eq,
        "rows": len(rows), "pca_rows": len(curves),
    }
    paths["metadata"].write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
