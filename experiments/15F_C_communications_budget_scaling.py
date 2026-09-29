"""Experiment 15F-C: communications performance versus temporal-state budget.

Lineage:
    15C -> matched-timescale control
    15D -> equal-state representation controls
    15E -> actual budgets N in {1,2,4,8,16}
    15F-A -> first communications transfer
    15F-B -> fixed-N channel stress
    15F-C -> explicit communications budget scaling

The temporal representation constructors are imported directly from 15E.
Only the communications environment and evaluation protocol are defined here.
"""

from __future__ import annotations

import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = spec_from_file_location(
    "exp15e", ROOT / "experiments" / "15E_state_budget_sweep.py"
)
U15E = module_from_spec(spec)
sys.modules[spec.name] = U15E
spec.loader.exec_module(U15E)
U = U15E.U

from min.metrics.equalization import add_awgn, evm
from min.signals import generate_bpsk, generate_qpsk

SEEDS = U15E.SEEDS
BUDGETS = U15E.BUDGETS
MODULATIONS = ("BPSK", "QPSK")
PROCESSES = U15E.PROCESSES
SCALE_FACTORS = (4.0, 12.0, 32.0)
SNR_DB = (0.0, 10.0, 20.0)

NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
DT = U15E.DT
NOISE_SEED_OFFSET = U15E.NOISE_SEED_OFFSET
READOUT_RIDGE = U15E.U.READOUT_RIDGE


def generate(modulation: str, seed: int):
    fn = {"BPSK": generate_bpsk, "QPSK": generate_qpsk}[modulation]
    return fn(
        NUM_TRAIN_SYMBOLS + NUM_TEST_SYMBOLS,
        samples_per_symbol=SPS,
        symbol_rate=SYMBOL_RATE,
        seed=seed,
    )


def channel_memory(x: np.ndarray, process: str, factor: float) -> np.ndarray:
    """15E process-family memory applied as a causal linear channel."""
    params = U15E.PROCESS_PARAMS[process]
    taus = np.asarray(params["taus"], dtype=float) / float(factor)
    weights = np.asarray(params["weights"], dtype=float)
    weights = weights / weights.sum()
    x = np.asarray(x, dtype=complex)

    components = np.empty((len(taus), x.size), dtype=complex)
    for j, tau in enumerate(taus):
        a = float(np.exp(-DT / tau))
        state = np.empty(x.size, dtype=complex)
        state[0] = x[0]
        for n in range(1, x.size):
            state[n] = a * state[n - 1] + (1.0 - a) * x[n]
        components[j] = state
    return weights @ components


def linear_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=complex)
    y = np.asarray(y, dtype=complex)
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    gram.flat[:: gram.shape[0] + 1] += READOUT_RIDGE
    return np.linalg.solve(gram, design.conj().T @ y)


def linear_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.ones(X.shape[0], dtype=complex), np.asarray(X, dtype=complex)]
    ) @ beta


def bits(modulation: str, symbols: np.ndarray) -> np.ndarray:
    s = np.asarray(symbols)
    if modulation == "BPSK":
        return (np.real(s) >= 0).astype(np.uint8)[:, None]

    phase = np.mod(np.angle(s) - np.pi / 4.0, 2.0 * np.pi)
    idx = np.mod(np.rint(phase / (np.pi / 2.0)).astype(int), 4)
    return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)


def decide(modulation: str, estimate: np.ndarray) -> np.ndarray:
    e = np.asarray(estimate)
    if modulation == "BPSK":
        return np.where(np.real(e) >= 0.0, 1.0, -1.0).astype(complex)

    constellation = np.exp(
        1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0)
    )
    return constellation[
        np.argmin(np.abs(e[:, None] - constellation[None, :]), axis=1)
    ]


def ber(modulation: str, reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_bits = bits(modulation, reference)
    est_bits = bits(modulation, decide(modulation, estimate))
    return float(np.mean(ref_bits != est_bits))


def ser(modulation: str, reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(decide(modulation, estimate) != reference))


def evaluate(
    X: np.ndarray, tx: np.ndarray, modulation: str
) -> dict[str, float]:
    beta = linear_fit(X[:NUM_TRAIN_SYMBOLS], tx[:NUM_TRAIN_SYMBOLS])
    estimate = linear_predict(X[NUM_TRAIN_SYMBOLS:], beta)
    reference = tx[NUM_TRAIN_SYMBOLS:]
    return {
        "ber": ber(modulation, reference, estimate),
        "ser": ser(modulation, reference, estimate),
        "evm_percent": 100.0 * evm(reference, estimate),
        "nmse": float(
            np.mean(np.abs(estimate - reference) ** 2)
            / np.mean(np.abs(reference) ** 2)
        ),
    }


def run_case(
    modulation: str,
    process: str,
    factor: float,
    snr_db: float,
    seed: int,
) -> list[dict]:
    signal = generate(modulation, seed)
    filtered = channel_memory(signal.record.samples, process, factor)

    noisy = add_awgn(
        filtered,
        snr_db,
        np.random.default_rng(
            NOISE_SEED_OFFSET + seed * 1000 + int(snr_db) + int(factor)
        ),
    )

    idx = signal.symbol_indices
    tx = signal.symbols

    reps_by_budget: dict[int, dict[str, np.ndarray]] = {}
    for budget in BUDGETS:
        reps, _, _ = U15E.representation_specs(
            process, factor, noisy, int(budget)
        )
        reps_by_budget[int(budget)] = reps

    rows: list[dict] = []

    current_metrics = evaluate(noisy[idx][:, None], tx, modulation)
    rows.append(
        {
            "experiment": "15F_C_communications_budget_scaling",
            "modulation": modulation,
            "process": process,
            "scale_factor": float(factor),
            "snr_db": float(snr_db),
            "seed": int(seed),
            "representation": "current_1",
            "state_budget": 1,
            **current_metrics,
        }
    )

    for budget in BUDGETS:
        for name, state in reps_by_budget[int(budget)].items():
            metrics = evaluate(state[idx], tx, modulation)
            rows.append(
                {
                    "experiment": "15F_C_communications_budget_scaling",
                    "modulation": modulation,
                    "process": process,
                    "scale_factor": float(factor),
                    "snr_db": float(snr_db),
                    "seed": int(seed),
                    "representation": name,
                    "state_budget": int(budget),
                    **metrics,
                }
            )

    return rows


def paired(rows: list[dict]) -> list[dict]:
    lookup: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row["representation"] == "current_1":
            continue
        key = (
            row["modulation"],
            row["process"],
            row["scale_factor"],
            row["snr_db"],
            row["seed"],
            row["state_budget"],
        )
        lookup.setdefault(key, {})[row["representation"]] = row

    out: list[dict] = []
    for key, reps in sorted(lookup.items()):
        min_name = f"min_{key[5]}"
        min_row = reps.get(min_name)
        if min_row is None:
            continue
        for control in (
            f"iir_logspread_{key[5]}",
            f"dense_ss_{key[5]}",
            f"fir_{key[5]}",
        ):
            ctrl = reps.get(control)
            if ctrl is None:
                continue
            out.append(
                {
                    "experiment": "15F_C_communications_budget_scaling",
                    "modulation": key[0],
                    "process": key[1],
                    "scale_factor": float(key[2]),
                    "snr_db": float(key[3]),
                    "seed": int(key[4]),
                    "state_budget": int(key[5]),
                    "control": control,
                    "delta_ber_min_minus_control": float(
                        min_row["ber"] - ctrl["ber"]
                    ),
                    "delta_ser_min_minus_control": float(
                        min_row["ser"] - ctrl["ser"]
                    ),
                    "delta_evm_min_minus_control": float(
                        min_row["evm_percent"] - ctrl["evm_percent"]
                    ),
                    "delta_nmse_min_minus_control": float(
                        min_row["nmse"] - ctrl["nmse"]
                    ),
                }
            )

    return out


def budget_deltas(rows: list[dict]) -> list[dict]:
    lookup: dict[tuple, dict[int, dict]] = {}
    for row in rows:
        rep = row["representation"]
        if rep == "current_1":
            continue

        family, budget_text = rep.rsplit("_", 1)
        budget = int(budget_text)
        base = (
            row["modulation"],
            row["process"],
            row["scale_factor"],
            row["snr_db"],
            row["seed"],
            family,
        )
        lookup.setdefault(base, {})[budget] = row

    transitions = tuple(zip(BUDGETS[:-1], BUDGETS[1:]))
    out: list[dict] = []

    for key, by_budget in sorted(lookup.items()):
        for n_from, n_to in transitions:
            lo = by_budget[int(n_from)]
            hi = by_budget[int(n_to)]
            out.append(
                {
                    "experiment": "15F_C_communications_budget_scaling",
                    "modulation": key[0],
                    "process": key[1],
                    "scale_factor": float(key[2]),
                    "snr_db": float(key[3]),
                    "seed": int(key[4]),
                    "representation": key[5],
                    "n_from": int(n_from),
                    "n_to": int(n_to),
                    "improvement_ber": float(
                        lo["ber"] - hi["ber"]
                    ),
                    "improvement_ser": float(
                        lo["ser"] - hi["ser"]
                    ),
                    "improvement_evm_percent": float(
                        lo["evm_percent"] - hi["evm_percent"]
                    ),
                    "improvement_nmse": float(
                        lo["nmse"] - hi["nmse"]
                    ),
                }
            )

    return out


def summarize(rows: list[dict]) -> list[dict]:
    keys = (
        "modulation",
        "process",
        "scale_factor",
        "snr_db",
        "representation",
        "state_budget",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out: list[dict] = []
    for key, batch in sorted(groups.items()):
        out.append(
            {
                "experiment": "15F_C_communications_budget_scaling",
                **dict(zip(keys, key)),
                "n": len(batch),
                "mean_ber": float(np.mean([r["ber"] for r in batch])),
                "mean_ser": float(np.mean([r["ser"] for r in batch])),
                "mean_evm_percent": float(
                    np.mean([r["evm_percent"] for r in batch])
                ),
                "mean_nmse": float(np.mean([r["nmse"] for r in batch])),
            }
        )
    return out


def main() -> None:
    equivalence = U.validate_repository_equivalence()

    rows: list[dict] = []
    for seed in SEEDS:
        for modulation in MODULATIONS:
            for process in PROCESSES:
                for factor in SCALE_FACTORS:
                    for snr in SNR_DB:
                        rows.extend(
                            run_case(
                                modulation,
                                process,
                                factor,
                                snr,
                                int(seed),
                            )
                        )

    per_condition = 1 + 4 * len(BUDGETS)
    expected = (
        len(SEEDS)
        * len(MODULATIONS)
        * len(PROCESSES)
        * len(SCALE_FACTORS)
        * len(SNR_DB)
        * per_condition
    )
    assert len(rows) == expected

    summary_rows = summarize(rows)
    paired_rows = paired(rows)
    delta_rows = budget_deltas(rows)

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "cases": out / "15F_C_communications_budget_scaling_cases.csv",
        "summary": out / "15F_C_communications_budget_scaling_summary.csv",
        "paired": out / "15F_C_communications_budget_scaling_paired_deltas.csv",
        "budget_deltas": out / "15F_C_communications_budget_scaling_budget_deltas.csv",
        "metadata": out / "15F_C_communications_budget_scaling_summary.json",
    }

    for path, data in (
        (paths["cases"], rows),
        (paths["summary"], summary_rows),
        (paths["paired"], paired_rows),
        (paths["budget_deltas"], delta_rows),
    ):
        with path.open("w", newline="") as f:
            fieldnames = sorted({k for row in data for k in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    metadata = {
        "experiment": "15F_C_communications_budget_scaling",
        "question": (
            "How does communications performance change as temporal-state "
            "budget increases from 1 to 2 to 4 to 8 to 16?"
        ),
        "lineage": [
            "15C_matched_timescale_control",
            "15D_equal_budget",
            "15E_actual_state_budget",
            "15F_A_synthetic_communications",
            "15F_B_channel_stress",
        ],
        "budgets": list(BUDGETS),
        "representations": [
            "min_N",
            "iir_logspread_N",
            "dense_ss_N",
            "fir_N",
        ],
        "baseline": "current_1",
        "modulations": list(MODULATIONS),
        "processes": list(PROCESSES),
        "scale_factors": list(SCALE_FACTORS),
        "snr_db": list(SNR_DB),
        "seeds": list(SEEDS),
        "protocol": {
            "same_noisy_iq_stream_across_budgets": True,
            "same_process_parameters_as_15E": True,
            "representation_constructors_imported_from_15E": True,
            "same_linear_readout": True,
            "train_symbols": NUM_TRAIN_SYMBOLS,
            "test_symbols": NUM_TEST_SYMBOLS,
            "causal_state": True,
            "heldout_evaluation": True,
            "nonlinear_receiver": False,
            "adaptive_kernel_tuning": False,
            "test_set_tuning": False,
        },
        "metrics": ["ber", "ser", "evm_percent", "nmse"],
        "budget_delta_definition": (
            "improvement_metric = metric_at_N_from - metric_at_N_to; "
            "positive values indicate lower error at the larger budget"
        ),
        "repository_state_equivalence_relative_error": equivalence,
        "case_rows": len(rows),
        "summary_rows": len(summary_rows),
        "paired_delta_rows": len(paired_rows),
        "budget_delta_rows": len(delta_rows),
    }

    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
