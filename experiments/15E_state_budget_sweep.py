"""Experiment 15E: actual temporal-state budget sweep.

Question:
    When the representation itself is constrained to N temporal states,
    how efficiently does process-adapted MIN allocate those states compared
    with a conventional logarithmic IIR pole bank, its dense coordinate
    control, and an N-sample FIR history?

Budgets:
    N in {1, 2, 4, 8, 16}.

Representations:
    min_N              process-adapted positive exponential dictionary
    iir_logspread_N    conventional logarithmic exponential bank spanning
                       the same endpoint rate range as the MIN construction
    dense_ss_N         orthogonal similarity transform of iir_logspread_N
                       (coordinate control)
    fir_N              N-sample finite-history delay line

Unlike 15D, the primary result here is direct full-state readout at the
actual state budget. No post-hoc 16D representation followed by PCA is used
to manufacture the budget. A compact state-vs-performance PCA curve is also
recorded up to N only as a secondary diagnostic.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location(
    "exp15c",
    ROOT / "experiments" / "15C_extended_timescale_iir_control.py",
)
U15C = module_from_spec(spec)
sys.modules[spec.name] = U15C
spec.loader.exec_module(U15C)
U = U15C.U

PROCESSES = U15C.PROCESSES
TASKS = U15C.TASKS
SEEDS = U15C.SEEDS
SNR_DB = U15C.SNR_DB
DT = U15C.DT
N = U15C.N
TRAIN_FRAC = U15C.TRAIN_FRAC
NOISE_SEED_OFFSET = U15C.NOISE_SEED_OFFSET
SCALE_FACTORS = U15C.SCALE_FACTORS
BUDGETS = (1, 2, 4, 8, 16)
MODES = U15C.MODES

PROCESS_PARAMS = U15C.PROCESS_PARAMS


def budget_rates(process: str, factor: float, budget: int) -> np.ndarray:
    """Deterministically allocate exponential rates to an N-state budget.

    For one-timescale processes, states are log-spread around the process
    rate exactly as in 15D. For multi-timescale processes, each component gets
    at least one state when the budget permits, with remaining states assigned
    by the component weights. Within each component the same ±0.15
    log-neighborhood is used.
    """
    if budget < 1:
        raise ValueError("budget must be positive")

    taus = np.asarray(PROCESS_PARAMS[process]["taus"], dtype=float) / factor
    weights = np.asarray(PROCESS_PARAMS[process]["weights"], dtype=float)
    weights = weights / weights.sum()

    if len(taus) == 1:
        return taus[0] ** -1 * np.exp(np.linspace(-0.15, 0.15, budget))

    if budget < len(taus):
        # One state cannot explicitly cover both scales. Use the dominant
        # weighted component as the deterministic single-state representative.
        idx = int(np.argmax(weights))
        return np.asarray([taus[idx] ** -1], dtype=float)

    counts = np.ones(len(taus), dtype=int)
    remaining = budget - len(taus)
    for _ in range(remaining):
        # Allocate each additional state where the current count-to-weight
        # ratio is smallest; stable tie-break follows process component order.
        scores = counts / np.maximum(weights, np.finfo(float).tiny)
        counts[int(np.argmin(scores))] += 1

    pieces = []
    for tau, count in zip(taus, counts):
        pieces.append(tau ** -1 * np.exp(np.linspace(-0.15, 0.15, int(count))))
    return np.concatenate(pieces)


def logspread_states(x: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    return U.exact_piecewise_linear_states(
        x, gammas, U.state_coefficients(gammas)
    )


def fir_states(x: np.ndarray, taps: int) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    out = np.zeros((x.size, taps), dtype=complex)
    for k in range(taps):
        if k == 0:
            out[:, k] = x
        else:
            out[k:, k] = x[:-k]
    return out


def dense_state_similarity(
    x: np.ndarray, gammas: np.ndarray, seed: int = 1504
) -> np.ndarray:
    """Coordinate-control realization of the same diagonal pole bank."""
    z = logspread_states(x, gammas)
    rng = np.random.default_rng(seed + len(gammas))
    q, _ = np.linalg.qr(rng.normal(size=(len(gammas), len(gammas))))
    return z @ q


def state_condition_number(X: np.ndarray) -> float:
    centered = X - np.mean(X, axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    pos = s[s > np.finfo(float).eps * max(s[0], 1.0)]
    return float(s[0] / pos[-1]) if pos.size else float("inf")


def direct_readout(
    Xtr: np.ndarray,
    target_tr: np.ndarray,
    Xte: np.ndarray,
    target_te: np.ndarray,
) -> float:
    beta = U.linear_readout_fit(Xtr.astype(complex), target_tr.astype(complex))
    pred = U.linear_readout_predict(Xte.astype(complex), beta)
    return float(U.nmse(target_te, pred))


def pca_curve(
    Xtr: np.ndarray,
    target_tr: np.ndarray,
    Xte: np.ndarray,
    target_te: np.ndarray,
) -> tuple[dict[int, float], np.ndarray, int]:
    """Secondary compact diagnostic: all retained dimensions up to state N."""
    mean, singular, components = U.complex_pca_fit(Xtr.astype(complex))
    max_k = Xtr.shape[1]
    curve: dict[int, float] = {}
    for k in range(1, max_k + 1):
        ztr = U.pca_project(Xtr.astype(complex), mean, components, k)
        zte = U.pca_project(Xte.astype(complex), mean, components, k)
        beta = U.linear_readout_fit(ztr, target_tr.astype(complex))
        curve[k] = float(
            U.nmse(target_te, U.linear_readout_predict(zte, beta))
        )
    full = curve[max_k]
    threshold = 1.10 * full
    d = next((k for k in range(1, max_k + 1) if curve[k] <= threshold), -1)
    return curve, singular, d


def representation_specs(
    process: str, factor: float, y: np.ndarray, budget: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    min_gammas = budget_rates(process, factor, budget)
    spread = np.geomspace(float(min_gammas.min()), float(min_gammas.max()), budget)
    reps = {
        f"min_{budget}": logspread_states(
            y, min_gammas
        ),
        f"iir_logspread_{budget}": logspread_states(y, spread),
        f"dense_ss_{budget}": dense_state_similarity(y, spread),
        f"fir_{budget}": fir_states(y, budget),
    }
    return reps, min_gammas, spread


def costs(budget: int) -> dict[str, dict[str, int]]:
    return {
        f"min_{budget}": {
            "state_dimension": budget,
            "parameters": 2 * budget,
            "ops_per_sample": 3 * budget,
            "memory_values": budget,
        },
        f"iir_logspread_{budget}": {
            "state_dimension": budget,
            "parameters": 2 * budget,
            "ops_per_sample": 3 * budget,
            "memory_values": budget,
        },
        f"dense_ss_{budget}": {
            "state_dimension": budget,
            "parameters": budget * budget + 2 * budget,
            "ops_per_sample": budget * budget + 2 * budget,
            "memory_values": budget,
        },
        f"fir_{budget}": {
            "state_dimension": budget,
            "parameters": budget,
            "ops_per_sample": budget,
            "memory_values": budget,
        },
    }


def run_case(
    process: str, factor: float, snr_db: float, seed: int, budget: int
) -> tuple[list[dict], list[dict], dict]:
    x, latent = U15C.generate_process(process, seed)
    y = U15C.add_observation_noise(x, snr_db, seed + NOISE_SEED_OFFSET)
    split = int(N * TRAIN_FRAC)

    reps, min_gammas, spread = representation_specs(process, factor, y, budget)
    geometry_by_rep = {
        name: U.spectrum_geometry(X[:split]) for name, X in reps.items()
    }
    cond_by_rep = {
        name: state_condition_number(X[:split]) for name, X in reps.items()
    }

    case_rows: list[dict] = []
    pca_rows: list[dict] = []
    cost_map = costs(budget)

    for name, X in reps.items():
        geometry = geometry_by_rep[name]
        cond = cond_by_rep[name]
        for task in TASKS:
            Xi = X[:-1] if task == "prediction" else X
            target = x[1:] if task == "prediction" else latent[:, 0]
            Xtr, Xte = Xi[:split], Xi[split:]
            ttr, tte = target[:split], target[split:]

            full_nmse = direct_readout(Xtr, ttr, Xte, tte)
            curve, singular, d = pca_curve(Xtr, ttr, Xte, tte)
            for k, value in curve.items():
                pca_rows.append(
                    {
                        "experiment": "15E_state_budget_sweep",
                        "process": process,
                        "representation": name,
                        "task": task,
                        "state_budget": budget,
                        "retained_dimension": k,
                        "scale_factor": factor,
                        "snr_db": float(snr_db),
                        "seed": int(seed),
                        "heldout_nmse": float(value),
                        "pca_explained_energy": float(U.pca_energy(singular, k)),
                        "state_condition_number": cond,
                        **geometry,
                        **cost_map[name],
                        "min_rate_min": float(min_gammas.min()),
                        "min_rate_max": float(min_gammas.max()),
                        "spread_rate_min": float(spread.min()),
                        "spread_rate_max": float(spread.max()),
                    }
                )

            case_rows.append(
                {
                    "experiment": "15E_state_budget_sweep",
                    "process": process,
                    "representation": name,
                    "task": task,
                    "state_budget": budget,
                    "scale_factor": float(factor),
                    "snr_db": float(snr_db),
                    "seed": int(seed),
                    "full_state_nmse": full_nmse,
                    "pca_d_task_10pct": int(d),
                    "state_condition_number": cond,
                    **geometry,
                    **cost_map[name],
                    "min_rate_min": float(min_gammas.min()),
                    "min_rate_max": float(min_gammas.max()),
                    "spread_rate_min": float(spread.min()),
                    "spread_rate_max": float(spread.max()),
                }
            )

    dense_error = float(
        np.max(
            [
                abs(
                    case_rows[i]["full_state_nmse"]
                    - case_rows[i + 1]["full_state_nmse"]
                )
                for i in range(len(case_rows) - 1)
                for _ in [0]
                if case_rows[i]["representation"].startswith("iir_logspread_")
                and case_rows[i + 1]["representation"].startswith("dense_ss_")
                and case_rows[i]["process"] == case_rows[i + 1]["process"]
                and case_rows[i]["task"] == case_rows[i + 1]["task"]
            ]
        )
    ) if case_rows else float("nan")
    return case_rows, pca_rows, {"dense_state_max_local_nmse_abs_diff": dense_error}


def summarize(rows: list[dict]) -> list[dict]:
    keys = (
        "process",
        "representation",
        "task",
        "state_budget",
        "scale_factor",
        "snr_db",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out = []
    for key, batch in sorted(groups.items()):
        process, rep, task, budget, factor, snr = key
        out.append(
            {
                "process": process,
                "representation": rep,
                "task": task,
                "state_budget": int(budget),
                "scale_factor": float(factor),
                "snr_db": float(snr),
                "n": len(batch),
                "mean_full_state_nmse": float(
                    np.mean([r["full_state_nmse"] for r in batch])
                ),
                "std_full_state_nmse": float(
                    np.std([r["full_state_nmse"] for r in batch], ddof=1)
                ),
                "mean_pca_d_task_10pct": float(
                    np.mean(
                        [
                            r["pca_d_task_10pct"]
                            for r in batch
                            if r["pca_d_task_10pct"] > 0
                        ]
                    )
                )
                if any(r["pca_d_task_10pct"] > 0 for r in batch)
                else float("nan"),
                "median_state_condition_number": float(
                    np.median([r["state_condition_number"] for r in batch])
                ),
            }
        )
    return out


def paired_budget_deltas(rows: list[dict]) -> list[dict]:
    by_key = {}
    for r in rows:
        base = (
            r["process"],
            r["task"],
            r["state_budget"],
            r["scale_factor"],
            r["snr_db"],
            r["seed"],
        )
        by_key.setdefault(base, {})[r["representation"]] = r["full_state_nmse"]

    out = []
    for base, reps in sorted(by_key.items()):
        process, task, budget, factor, snr, seed = base
        min_name = f"min_{budget}"
        iir_name = f"iir_logspread_{budget}"
        dense_name = f"dense_ss_{budget}"
        fir_name = f"fir_{budget}"
        if not all(nm in reps for nm in (min_name, iir_name, dense_name, fir_name)):
            continue
        out.append(
            {
                "experiment": "15E_state_budget_sweep",
                "process": process,
                "task": task,
                "state_budget": int(budget),
                "scale_factor": float(factor),
                "snr_db": float(snr),
                "seed": int(seed),
                "min_minus_iir_nmse": float(reps[min_name] - reps[iir_name]),
                "min_minus_dense_nmse": float(reps[min_name] - reps[dense_name]),
                "min_minus_fir_nmse": float(reps[min_name] - reps[fir_name]),
                "iir_minus_dense_nmse": float(reps[iir_name] - reps[dense_name]),
                "min_nmse": float(reps[min_name]),
                "iir_nmse": float(reps[iir_name]),
                "dense_nmse": float(reps[dense_name]),
                "fir_nmse": float(reps[fir_name]),
            }
        )
    return out


def marginal_state_value(rows: list[dict]) -> list[dict]:
    lookup = {}
    for r in rows:
        lookup[
            (
                r["process"],
                r["representation"],
                r["task"],
                r["scale_factor"],
                r["snr_db"],
                r["seed"],
                r["state_budget"],
            )
        ] = r["full_state_nmse"]

    out = []
    for process in PROCESSES:
        for rep_prefix in ("min", "iir_logspread", "dense_ss", "fir"):
            for task in TASKS:
                for factor in SCALE_FACTORS:
                    for snr in SNR_DB:
                        for seed in SEEDS:
                            for b0, b1 in zip(BUDGETS[:-1], BUDGETS[1:]):
                                a = lookup.get(
                                    (process, f"{rep_prefix}_{b0}", task, factor, snr, seed, b0)
                                )
                                b = lookup.get(
                                    (process, f"{rep_prefix}_{b1}", task, factor, snr, seed, b1)
                                )
                                if a is None or b is None:
                                    continue
                                out.append(
                                    {
                                        "experiment": "15E_state_budget_sweep",
                                        "process": process,
                                        "representation": rep_prefix,
                                        "task": task,
                                        "from_budget": b0,
                                        "to_budget": b1,
                                        "scale_factor": float(factor),
                                        "snr_db": float(snr),
                                        "seed": int(seed),
                                        "nmse_reduction": float(a - b),
                                    }
                                )
    return out


def main() -> None:
    eq = U.validate_repository_equivalence()
    all_cases: list[dict] = []
    all_pca: list[dict] = []
    max_dense_abs_diff = 0.0

    for seed in SEEDS:
        for process in PROCESSES:
            for factor in SCALE_FACTORS:
                for snr in SNR_DB:
                    for budget in BUDGETS:
                        rows, pca_rows, controls = run_case(
                            process, factor, snr, seed, budget
                        )
                        all_cases.extend(rows)
                        all_pca.extend(pca_rows)
                        max_dense_abs_diff = max(
                            max_dense_abs_diff,
                            controls["dense_state_max_local_nmse_abs_diff"],
                        )

    expected_cases = (
        len(SEEDS)
        * len(PROCESSES)
        * len(SCALE_FACTORS)
        * len(SNR_DB)
        * len(BUDGETS)
        * len(TASKS)
        * 4
    )
    expected_pca_min = expected_cases  # at least one retained-dimension row per case
    assert len(all_cases) == expected_cases
    assert len(all_pca) >= expected_pca_min

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / "15E_state_budget_sweep_cases.csv"
    pca_path = out / "15E_state_budget_sweep_pca_results.csv"
    summary_path = out / "15E_state_budget_sweep_summary.csv"
    deltas_path = out / "15E_state_budget_sweep_paired_deltas.csv"
    marginal_path = out / "15E_state_budget_sweep_marginal_state_value.csv"
    metadata_path = out / "15E_state_budget_sweep_summary.json"

    for path, data in (
        (cases_path, all_cases),
        (pca_path, all_pca),
        (summary_path, summarize(all_cases)),
        (deltas_path, paired_budget_deltas(all_cases)),
        (marginal_path, marginal_state_value(all_cases)),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    metadata = {
        "experiment": "15E_state_budget_sweep",
        "question": (
            "How does task performance scale with actual temporal-state budget?"
        ),
        "processes": PROCESSES,
        "tasks": TASKS,
        "budgets": list(BUDGETS),
        "scale_factors": list(SCALE_FACTORS),
        "snr_db": list(SNR_DB),
        "seeds": list(SEEDS),
        "N": N,
        "train_fraction": TRAIN_FRAC,
        "representations": [
            "min_N",
            "iir_logspread_N",
            "dense_ss_N",
            "fir_N",
        ],
        "controls": {
            "same_observation": True,
            "same_split": True,
            "same_readout": True,
            "same_process_and_noise_grid": True,
            "actual_state_budget_equals_representation_dimension": True,
            "min_deterministic_process_adaptation": True,
            "dense_state_is_similarity_control": True,
            "posthoc_pca_not_primary_metric": True,
        },
        "repository_state_equivalence_relative_error": eq,
        "max_dense_state_full_state_nmse_abs_diff": max_dense_abs_diff,
        "case_rows": len(all_cases),
        "pca_rows": len(all_pca),
        "paired_delta_rows": len(paired_budget_deltas(all_cases)),
        "marginal_state_value_rows": len(marginal_state_value(all_cases)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
