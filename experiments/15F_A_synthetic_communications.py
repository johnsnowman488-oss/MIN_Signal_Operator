"""Experiment 15F-A: synthetic communications-task validation.

Question:
    Does temporal information retained by a fixed MIN state translate into
    measurable BER/EVM/SER performance on a controlled communications task?

Design:
    BPSK/QPSK baseband -> causal channel memory -> complex AWGN -> representation
    -> one linear readout -> held-out symbol estimates.

Representations:
    current_1       current noisy observation
    min_N           process-memory-matched positive exponential MIN/SOE states
    iir_logspread_N fixed conventional log-spread exponential bank
    fir_N           N-sample causal finite-history state

Budgets:
    N in {1,2,4,8,16}.
"""

from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min.metrics.equalization import add_awgn, evm
from min.signals import generate_bpsk, generate_qpsk
from min import SOEMemory

SEEDS = tuple(range(5))
MODULATIONS = ("BPSK", "QPSK")
MEMORIES = ("identity", "tau_0.005", "tau_0.020", "tau_0.050")
TAUS = {"identity": 0.0, "tau_0.005": 0.005, "tau_0.020": 0.020, "tau_0.050": 0.050}
SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
BUDGETS = (1, 2, 4, 8, 16)
NUM_SYMBOLS = 512
SPS = 8
SYMBOL_RATE = 100.0
SAMPLE_RATE = SYMBOL_RATE * SPS
DT = 1.0 / SAMPLE_RATE
TRAIN_SYMBOLS = 256
MIN_LOG_HALF_WIDTH = 0.35
IIR_RATE_RANGE = (0.5, 200.0)
READOUT_RIDGE = 1e-10
NOISE_SEED_OFFSET = 50_000

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
    gram = design.conj().T @ design + READOUT_RIDGE * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.conj().T @ y)

def linear_predict(X, beta):
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta

def channel_memory(x, tau):
    x = np.asarray(x, dtype=complex)
    if tau == 0.0:
        return x.copy()
    alpha = float(np.exp(-DT / tau))
    y = np.empty_like(x)
    y[0] = x[0]
    for n in range(1, x.size):
        y[n] = alpha * y[n - 1] + (1.0 - alpha) * x[n]
    return y

def rates_for_min(tau, budget):
    center = 1.0 / DT if tau == 0.0 else 1.0 / tau
    return center * np.exp(np.linspace(-MIN_LOG_HALF_WIDTH, MIN_LOG_HALF_WIDTH, budget))

def rates_for_iir(budget):
    return np.geomspace(IIR_RATE_RANGE[0], IIR_RATE_RANGE[1], budget)

def min_states(y, tau, budget):
    gammas = rates_for_min(tau, budget)
    return SOEMemory(np.full(budget, 1.0 / budget), gammas).state_trajectory(
        np.arange(y.size, dtype=float) * DT, y
    )

def iir_states(y, budget):
    gammas = rates_for_iir(budget)
    return SOEMemory(np.full(budget, 1.0 / budget), gammas).state_trajectory(
        np.arange(y.size, dtype=float) * DT, y
    )

def fir_states(y, budget):
    out = np.zeros((y.size, budget), dtype=complex)
    out[:, 0] = y
    for k in range(1, budget):
        out[k:, k] = y[:-k]
    return out

def generate(modulation, seed):
    fn = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return fn(NUM_SYMBOLS, samples_per_symbol=SPS, symbol_rate=SYMBOL_RATE, seed=seed)

def run_case(modulation, memory, snr_db, seed):
    signal = generate(modulation, seed)
    channel_out = channel_memory(signal.record.samples, TAUS[memory])
    noisy = add_awgn(channel_out, snr_db, np.random.default_rng(NOISE_SEED_OFFSET + seed * 100 + int(snr_db)))
    idx, tx = signal.symbol_indices, signal.symbols
    representations = {"current_1": noisy[idx, None]}
    for budget in BUDGETS:
        representations[f"min_{budget}"] = min_states(noisy, TAUS[memory], budget)[idx]
        representations[f"iir_logspread_{budget}"] = iir_states(noisy, budget)[idx]
        representations[f"fir_{budget}"] = fir_states(noisy, budget)[idx]
    rows = []
    for representation, X in representations.items():
        beta = linear_fit(X[:TRAIN_SYMBOLS], tx[:TRAIN_SYMBOLS])
        estimate = linear_predict(X[TRAIN_SYMBOLS:], beta)
        state_budget = int(representation.rsplit("_", 1)[-1]) if representation != "current_1" else 1
        reference = tx[TRAIN_SYMBOLS:]
        rows.append({
            "experiment": "15F_A_synthetic_communications",
            "modulation": modulation, "memory": memory, "tau_s": float(TAUS[memory]),
            "snr_db": float(snr_db), "seed": int(seed), "representation": representation,
            "state_budget": state_budget, "train_symbols": TRAIN_SYMBOLS,
            "test_symbols": NUM_SYMBOLS - TRAIN_SYMBOLS,
            "ber": ber(modulation, reference, estimate),
            "ser": ser(modulation, reference, estimate),
            "evm_percent": 100.0 * evm(reference, estimate),
            "nmse": float(np.mean(np.abs(estimate-reference)**2) / np.mean(np.abs(reference)**2)),
        })
    return rows

def main():
    rows = []
    for seed in SEEDS:
        for modulation in MODULATIONS:
            for memory in MEMORIES:
                for snr_db in SNR_DB:
                    rows.extend(run_case(modulation, memory, snr_db, seed))
    expected = len(SEEDS) * len(MODULATIONS) * len(MEMORIES) * len(SNR_DB) * (1 + 3 * len(BUDGETS))
    assert len(rows) == expected
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    cases = out / "15F_A_synthetic_communications_cases.csv"
    summary = out / "15F_A_synthetic_communications_summary.csv"
    metadata = out / "15F_A_synthetic_communications_summary.json"
    with cases.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    keys = ("modulation", "memory", "snr_db", "representation", "state_budget")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary_rows = []
    for key, batch in sorted(groups.items()):
        modulation, memory, snr_db, representation, state_budget = key
        summary_rows.append({
            "experiment": "15F_A_synthetic_communications",
            "modulation": modulation, "memory": memory, "snr_db": float(snr_db),
            "representation": representation, "state_budget": int(state_budget), "n": len(batch),
            "mean_ber": float(np.mean([r["ber"] for r in batch])),
            "mean_ser": float(np.mean([r["ser"] for r in batch])),
            "mean_evm_percent": float(np.mean([r["evm_percent"] for r in batch])),
            "mean_nmse": float(np.mean([r["nmse"] for r in batch])),
        })
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    metadata.write_text(json.dumps({
        "experiment": "15F_A_synthetic_communications", "stage": "15F-A",
        "question": "Does temporal information preserved by MIN translate into BER/EVM/SER performance?",
        "modulations": list(MODULATIONS), "memories": list(MEMORIES), "taus_s": TAUS,
        "snr_db": list(SNR_DB), "budgets": list(BUDGETS), "seeds": list(SEEDS),
        "representations": ["current_1", "min_N", "iir_logspread_N", "fir_N"],
        "train_test_split": "first 256 symbols fit the linear readout; last 256 symbols are held out",
        "causality": "all representations are causal; no test symbols enter state construction or readout fitting",
        "evm_definition": "100 * sqrt(mean(|estimate-reference|^2) / mean(|reference|^2))",
        "expected_case_rows": expected, "case_rows": len(rows),
        "note": "Synthetic validation only; real IQ is reserved for the 16-series."
    }, indent=2) + "\n")
    print(json.dumps({"case_rows": len(rows), "summary_rows": len(summary_rows)}, indent=2))

if __name__ == "__main__":
    main()
