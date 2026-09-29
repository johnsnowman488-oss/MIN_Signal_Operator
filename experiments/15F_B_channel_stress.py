"""Experiment 15F-B: communications memory/channel stress at fixed 16-state budget.

Lineage:
    15C -> matched-timescale control
    15D -> equal-budget representation controls
    15E -> actual state budget N in {1,2,4,8,16}
    15F-A -> first transfer to BPSK/QPSK symbol recovery
    15F-B -> stress the same representations at fixed N=16

Only the communication environment changes here. The temporal state machinery
is imported directly from 15E and is not reimplemented.
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

spec = spec_from_file_location("exp15e", ROOT / "experiments" / "15E_state_budget_sweep.py")
U15E = module_from_spec(spec)
sys.modules[spec.name] = U15E
spec.loader.exec_module(U15E)
U = U15E.U

from min.metrics.equalization import add_awgn, evm
from min.signals import generate_bpsk, generate_qpsk, generate_16qam

SEEDS = U15E.SEEDS
MODULATIONS = ("BPSK", "QPSK", "16QAM")
PROCESSES = U15E.PROCESSES
BUDGET = 16
SCALE_FACTORS = (4.0, 12.0, 32.0)
SNR_DB = (0.0, 10.0, 20.0)
CHANNEL_PROFILES = (
    "none", "multipath_mild", "multipath_strong",
    "fading_mild", "fading_strong",
)
NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
DT = U15E.DT
NOISE_SEED_OFFSET = U15E.NOISE_SEED_OFFSET
READOUT_RIDGE = U15E.U.READOUT_RIDGE

MULTIPATH = {
    "none": (),
    "multipath_mild": ((16, 0.25),),
    "multipath_strong": ((16, 0.35), (32, 0.20)),
}
FADING_DEPTH = {"fading_mild": 0.20, "fading_strong": 0.50}
FADING_TAU = 0.20

def generate(modulation, seed):
    fn = {"BPSK": generate_bpsk, "QPSK": generate_qpsk, "16QAM": generate_16qam}[modulation]
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

def apply_multipath(x, profile):
    taps = MULTIPATH[profile]
    if not taps:
        return np.asarray(x, dtype=complex).copy()
    x = np.asarray(x, dtype=complex)
    y = x.copy()
    for delay, gain in taps:
        if delay < x.size:
            y[delay:] += float(gain) * x[:-delay]
    return y / max(float(np.sqrt(np.mean(np.abs(y) ** 2))), np.finfo(float).tiny)

def fading_process(n, seed):
    rng = np.random.default_rng(seed)
    alpha = float(np.exp(-DT / FADING_TAU))
    innovations = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0)
    out = np.empty(n, dtype=complex)
    out[0] = innovations[0]
    scale = np.sqrt(max(1.0 - alpha * alpha, np.finfo(float).tiny))
    for k in range(1, n):
        out[k] = alpha * out[k - 1] + scale * innovations[k]
    return out / max(float(np.sqrt(np.mean(np.abs(out) ** 2))), np.finfo(float).tiny)

def apply_fading(x, profile, seed):
    depth = FADING_DEPTH.get(profile, 0.0)
    if depth == 0.0:
        return np.asarray(x, dtype=complex).copy()
    h = fading_process(len(x), seed)
    gain = np.sqrt(1.0 - depth) + np.sqrt(depth) * h
    y = np.asarray(x, dtype=complex) * gain
    return y / max(float(np.sqrt(np.mean(np.abs(y) ** 2))), np.finfo(float).tiny)

def channel(x, process, factor, profile, seed):
    y = channel_memory(x, process, factor)
    if profile in MULTIPATH:
        return apply_multipath(y, profile)
    return apply_fading(y, profile, 700_000 + seed * 101 + int(factor))

def bits(modulation, symbols):
    s = np.asarray(symbols)
    if modulation == "BPSK":
        return (np.real(s) >= 0).astype(np.uint8)[:, None]
    if modulation == "QPSK":
        phase = np.mod(np.angle(s) - np.pi / 4.0, 2.0 * np.pi)
        idx = np.mod(np.rint(phase / (np.pi / 2.0)).astype(int), 4)
        return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    gray = {-3.0: (0, 0), -1.0: (0, 1), 1.0: (1, 1), 3.0: (1, 0)}
    real_levels = levels[np.argmin(np.abs(np.real(s)[:, None] * np.sqrt(10.0) - levels[None, :]), axis=1)]
    imag_levels = levels[np.argmin(np.abs(np.imag(s)[:, None] * np.sqrt(10.0) - levels[None, :]), axis=1)]
    return np.asarray([gray[i] + gray[q] for i, q in zip(real_levels, imag_levels)], dtype=np.uint8)

def decide(modulation, estimate):
    e = np.asarray(estimate)
    if modulation == "BPSK":
        return np.where(np.real(e) >= 0.0, 1.0, -1.0).astype(complex)
    if modulation == "QPSK":
        constellation = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    else:
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        constellation = np.asarray([(i + 1j * q) / np.sqrt(10.0) for i in levels for q in levels], dtype=complex)
    return constellation[np.argmin(np.abs(e[:, None] - constellation[None, :]), axis=1)]

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

def evaluate(X, tx, modulation):
    beta = linear_fit(X[:NUM_TRAIN_SYMBOLS], tx[:NUM_TRAIN_SYMBOLS])
    estimate = linear_predict(X[NUM_TRAIN_SYMBOLS:], beta)
    reference = tx[NUM_TRAIN_SYMBOLS:]
    return {
        "ber": ber(modulation, reference, estimate),
        "ser": ser(modulation, reference, estimate),
        "evm_percent": 100.0 * evm(reference, estimate),
        "nmse": float(np.mean(np.abs(estimate - reference) ** 2) / np.mean(np.abs(reference) ** 2)),
    }

def run_case(modulation, process, factor, snr_db, profile, seed):
    signal = generate(modulation, seed)
    channel_out = channel(signal.record.samples, process, factor, profile, seed)
    noisy = add_awgn(channel_out, snr_db, np.random.default_rng(
        NOISE_SEED_OFFSET + seed * 1000 + int(snr_db) + int(factor)
    ))
    idx, tx = signal.symbol_indices, signal.symbols
    reps, min_gammas, spread = U15E.representation_specs(process, factor, noisy, BUDGET)
    rows = []

    raw_metrics = evaluate(noisy[idx][:, None], tx, modulation)
    rows.append({
        "experiment": "15F_B_channel_stress", "modulation": modulation, "process": process,
        "scale_factor": float(factor), "snr_db": float(snr_db), "channel_profile": profile,
        "seed": int(seed), "representation": "current_1", "state_budget": 1, **raw_metrics,
    })

    for name, state in reps.items():
        metrics = evaluate(state[idx], tx, modulation)
        rows.append({
            "experiment": "15F_B_channel_stress", "modulation": modulation, "process": process,
            "scale_factor": float(factor), "snr_db": float(snr_db), "channel_profile": profile,
            "seed": int(seed), "representation": name, "state_budget": BUDGET, **metrics,
            "min_rate_min": float(min_gammas.min()), "min_rate_max": float(min_gammas.max()),
            "spread_rate_min": float(spread.min()), "spread_rate_max": float(spread.max()),
        })
    return rows

def paired(rows):
    lookup = {}
    for row in rows:
        key = (row["modulation"], row["process"], row["scale_factor"], row["snr_db"], row["channel_profile"], row["seed"])
        lookup.setdefault(key, {})[row["representation"]] = row
    out = []
    for key, reps in sorted(lookup.items()):
        mn = reps.get("min_16")
        if mn is None:
            continue
        for control in ("iir_logspread_16", "dense_ss_16", "fir_16"):
            if control not in reps:
                continue
            out.append({
                "experiment": "15F_B_channel_stress", "modulation": key[0], "process": key[1],
                "scale_factor": float(key[2]), "snr_db": float(key[3]), "channel_profile": key[4],
                "seed": int(key[5]), "state_budget": BUDGET, "control": control,
                "delta_ber_min_minus_control": float(mn["ber"] - reps[control]["ber"]),
                "delta_ser_min_minus_control": float(mn["ser"] - reps[control]["ser"]),
                "delta_evm_min_minus_control": float(mn["evm_percent"] - reps[control]["evm_percent"]),
                "delta_nmse_min_minus_control": float(mn["nmse"] - reps[control]["nmse"]),
            })
    return out

def summarize(rows):
    keys = ("modulation", "process", "scale_factor", "snr_db", "channel_profile", "representation", "state_budget")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [{
        "experiment": "15F_B_channel_stress", **dict(zip(keys, key)), "n": len(batch),
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
                        for profile in CHANNEL_PROFILES:
                            rows.extend(run_case(modulation, process, factor, snr, profile, seed))
    expected = len(SEEDS) * len(MODULATIONS) * len(PROCESSES) * len(SCALE_FACTORS) * len(SNR_DB) * len(CHANNEL_PROFILES) * 5
    assert len(rows) == expected
    summary_rows = summarize(rows)
    paired_rows = paired(rows)
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": out / "15F_B_channel_stress_cases.csv",
        "summary": out / "15F_B_channel_stress_summary.csv",
        "paired": out / "15F_B_channel_stress_paired_deltas.csv",
        "metadata": out / "15F_B_channel_stress_summary.json",
    }
    for path, data in ((paths["cases"], rows), (paths["summary"], summary_rows), (paths["paired"], paired_rows)):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in data for k in row}))
            writer.writeheader()
            writer.writerows(data)
    metadata = {
        "experiment": "15F_B_channel_stress",
        "question": "Does the 15D/15E representation relationship survive controlled communications channel stress at fixed N=16?",
        "lineage": ["15C_matched_timescale_control", "15D_equal_budget", "15E_actual_state_budget", "15F_A_synthetic_communications"],
        "modulations": list(MODULATIONS), "processes": list(PROCESSES), "budget": BUDGET,
        "scale_factors": list(SCALE_FACTORS), "snr_db": list(SNR_DB),
        "channel_profiles": list(CHANNEL_PROFILES), "seeds": list(SEEDS),
        "representations": ["current_1", "min_16", "iir_logspread_16", "dense_ss_16", "fir_16"],
        "controls": {
            "representation_constructors_imported_from_15E": True,
            "fixed_actual_state_budget": 16,
            "same_process_parameters_as_15E": True,
            "same_scale_parameterization_as_15E": True,
            "same_linear_readout": True,
            "same_train_test_split": True,
            "causal_state": True,
            "dense_state_is_coordinate_control": True,
            "current_observation_readout_fitted_on_training": True,
        },
        "stress_axes": {
            "snr": list(SNR_DB),
            "memory": "15E process families and scale factors",
            "modulation_order": list(MODULATIONS),
            "multipath": "none, one delayed tap, two delayed taps",
            "fading": "slow complex Gauss-Markov fading at two depths",
        },
        "repository_state_equivalence_relative_error": eq,
        "case_rows": len(rows), "summary_rows": len(summary_rows), "paired_delta_rows": len(paired_rows),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
