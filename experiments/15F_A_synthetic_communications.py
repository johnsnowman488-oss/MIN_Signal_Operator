"""Experiment 15F-A: synthetic communications validation built directly on 15D/15E.

15F-A is the task-transfer test for the 15-series:
15C established the matched-timescale one-pole control;
15D established the equal-state representation controls;
15E established the actual N={1,2,4,8,16} state-budget sweep.
This experiment carries those controls into symbol recovery and measures BER,
SER and EVM.

The representation constructors are imported from 15E rather than reimplemented.
The communication channel uses the same 15E process timescales and scale-factor
parameterization, but is a controlled complex-baseband ISI channel.
"""

from __future__ import annotations
import csv, json, sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = spec_from_file_location("exp15e", ROOT / "experiments" / "15E_state_budget_sweep.py")
U15E = module_from_spec(spec)
sys.modules[spec.name] = U15E
spec.loader.exec_module(U15E)
U = U15E.U

from min.metrics.equalization import add_awgn, evm
from min.signals import generate_bpsk, generate_qpsk

SEEDS = U15E.SEEDS
MODULATIONS = ("BPSK", "QPSK")
PROCESSES = U15E.PROCESSES
BUDGETS = U15E.BUDGETS
SCALE_FACTORS = (4.0, 12.0, 32.0)
SNR_DB = (0.0, 10.0, 20.0)
NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
DT = U15E.DT
NOISE_SEED_OFFSET = U15E.NOISE_SEED_OFFSET
READOUT_RIDGE = U15E.U.READOUT_RIDGE

def generate(modulation, seed):
    fn = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return fn(NUM_TRAIN_SYMBOLS + NUM_TEST_SYMBOLS, samples_per_symbol=SPS,
              symbol_rate=SYMBOL_RATE, seed=seed)

def channel_memory(x, process, factor):
    p = U15E.PROCESS_PARAMS[process]
    taus = np.asarray(p["taus"], dtype=float) / float(factor)
    weights = np.asarray(p["weights"], dtype=float)
    weights /= weights.sum()
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

def bits(modulation, symbols):
    symbols = np.asarray(symbols)
    if modulation == "BPSK":
        return (np.real(symbols) >= 0).astype(np.uint8)[:, None]
    phase = np.mod(np.angle(symbols) - np.pi / 4.0, 2.0 * np.pi)
    idx = np.mod(np.rint(phase / (np.pi / 2.0)).astype(int), 4)
    return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)

def decide(modulation, estimate):
    estimate = np.asarray(estimate)
    if modulation == "BPSK":
        return np.where(np.real(estimate) >= 0.0, 1.0, -1.0).astype(complex)
    constellation = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    return constellation[np.argmin(abs(estimate[:, None] - constellation[None, :]), axis=1)]

def ber(modulation, reference, estimate):
    return float(np.mean(bits(modulation, reference) != bits(modulation, decide(modulation, estimate))))

def ser(modulation, reference, estimate):
    return float(np.mean(decide(modulation, estimate) != reference))

def linear_fit(X, y):
    X = np.asarray(X, dtype=complex)
    y = np.asarray(y, dtype=complex)
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    gram = design.conj().T @ design
    gram.flat[:: gram.shape[0] + 1] += READOUT_RIDGE
    return np.linalg.solve(gram, design.conj().T @ y)

def linear_predict(X, beta):
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta

def run_case(modulation, process, factor, snr_db, seed):
    signal = generate(modulation, seed)
    channel = channel_memory(signal.record.samples, process, factor)
    noisy = add_awgn(channel, snr_db, np.random.default_rng(
        NOISE_SEED_OFFSET + seed * 1000 + int(snr_db) + int(factor)))
    idx, tx = signal.symbol_indices, signal.symbols
    reference = tx[NUM_TRAIN_SYMBOLS:]
    raw = noisy[idx[NUM_TRAIN_SYMBOLS:]]
    rows = [{
        "experiment": "15F_A_synthetic_communications",
        "modulation": modulation, "process": process, "scale_factor": float(factor),
        "snr_db": float(snr_db), "seed": int(seed), "representation": "current_1",
        "state_budget": 1, "train_symbols": NUM_TRAIN_SYMBOLS, "test_symbols": NUM_TEST_SYMBOLS,
        "ber": ber(modulation, reference, raw), "ser": ser(modulation, reference, raw),
        "evm_percent": 100.0 * evm(reference, raw),
        "nmse": float(np.mean(np.abs(raw-reference)**2) / np.mean(np.abs(reference)**2)),
    }]
    for budget in BUDGETS:
        reps, min_gammas, spread = U15E.representation_specs(process, factor, noisy, budget)
        for representation, state in reps.items():
            sampled = state[idx]
            beta = linear_fit(sampled[:NUM_TRAIN_SYMBOLS], tx[:NUM_TRAIN_SYMBOLS])
            estimate = linear_predict(sampled[NUM_TRAIN_SYMBOLS:], beta)
            rows.append({
                "experiment": "15F_A_synthetic_communications",
                "modulation": modulation, "process": process, "scale_factor": float(factor),
                "snr_db": float(snr_db), "seed": int(seed), "representation": representation,
                "state_budget": int(budget), "train_symbols": NUM_TRAIN_SYMBOLS,
                "test_symbols": NUM_TEST_SYMBOLS,
                "ber": ber(modulation, reference, estimate), "ser": ser(modulation, reference, estimate),
                "evm_percent": 100.0 * evm(reference, estimate),
                "nmse": float(np.mean(np.abs(estimate-reference)**2) / np.mean(np.abs(reference)**2)),
                "min_rate_min": float(min_gammas.min()), "min_rate_max": float(min_gammas.max()),
                "spread_rate_min": float(spread.min()), "spread_rate_max": float(spread.max()),
            })
    return rows

def paired(rows):
    lookup = {}
    for r in rows:
        key = (r["modulation"], r["process"], r["scale_factor"], r["snr_db"], r["seed"], r["state_budget"])
        lookup.setdefault(key, {})[r["representation"]] = r
    out = []
    for key, reps in sorted(lookup.items()):
        b = key[-1]
        min_name = f"min_{b}"
        if min_name not in reps:
            continue
        for control in (f"iir_logspread_{b}", f"dense_ss_{b}", f"fir_{b}"):
            if control in reps:
                out.append({
                    "experiment": "15F_A_synthetic_communications",
                    "modulation": key[0], "process": key[1], "scale_factor": float(key[2]),
                    "snr_db": float(key[3]), "seed": int(key[4]), "state_budget": int(b),
                    "control": control,
                    "delta_ber_min_minus_control": float(reps[min_name]["ber"] - reps[control]["ber"]),
                    "delta_ser_min_minus_control": float(reps[min_name]["ser"] - reps[control]["ser"]),
                    "delta_evm_min_minus_control": float(reps[min_name]["evm_percent"] - reps[control]["evm_percent"]),
                    "delta_nmse_min_minus_control": float(reps[min_name]["nmse"] - reps[control]["nmse"]),
                })
    return out

def summarize(rows):
    keys = ("modulation", "process", "scale_factor", "snr_db", "representation", "state_budget")
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [{
        "experiment": "15F_A_synthetic_communications", **dict(zip(keys, key)), "n": len(batch),
        "mean_ber": float(np.mean([r["ber"] for r in batch])),
        "mean_ser": float(np.mean([r["ser"] for r in batch])),
        "mean_evm_percent": float(np.mean([r["evm_percent"] for r in batch])),
        "mean_nmse": float(np.mean([r["nmse"] for r in batch])),
    } for key, batch in sorted(groups.items())]

def main():
    eq = U.validate_repository_equivalence()
    rows = []
    for seed in SEEDS:
        for modulation in MODULATIONS:
            for process in PROCESSES:
                for factor in SCALE_FACTORS:
                    for snr in SNR_DB:
                        rows.extend(run_case(modulation, process, factor, snr, seed))
    expected = len(SEEDS)*len(MODULATIONS)*len(PROCESSES)*len(SCALE_FACTORS)*len(SNR_DB)*(1 + 4*len(BUDGETS))
    assert len(rows) == expected
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": out / "15F_A_synthetic_communications_cases.csv",
        "summary": out / "15F_A_synthetic_communications_summary.csv",
        "paired": out / "15F_A_synthetic_communications_paired_deltas.csv",
        "metadata": out / "15F_A_synthetic_communications_summary.json",
    }
    for path, data in ((paths["cases"], rows), (paths["summary"], summarize(rows)), (paths["paired"], paired(rows))):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for row in data for k in row}))
            w.writeheader(); w.writerows(data)
    metadata = {
        "experiment": "15F_A_synthetic_communications",
        "question": "Does the 15D/15E temporal-state result transfer to receiver metrics?",
        "lineage": ["15C_matched_timescale_control", "15D_equal_budget", "15E_actual_state_budget"],
        "modulations": list(MODULATIONS), "processes": list(PROCESSES), "budgets": list(BUDGETS),
        "scale_factors": list(SCALE_FACTORS), "snr_db": list(SNR_DB), "seeds": list(SEEDS),
        "representations": ["current_1", "min_N", "iir_logspread_N", "dense_ss_N", "fir_N"],
        "controls": {
            "representation_constructors_imported_from_15E": True,
            "same_actual_state_budget_as_15E": True, "same_process_parameters_as_15E": True,
            "same_scale_parameterization_as_15E": True, "same_seed_and_snr_convention": True,
            "linear_readout": True, "train_test_split": "first 512 symbols fit readout; final 512 held out",
            "causal_state": True, "dense_state_is_coordinate_control": True,
        },
        "channel": "causal normalized exponential mixture using 15E process taus/weights",
        "repository_state_equivalence_relative_error": eq,
        "case_rows": len(rows), "summary_rows": len(summarize(rows)), "paired_delta_rows": len(paired(rows)),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
