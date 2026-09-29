"""Experiment 15F-D: communications closure under memory geometry.

This is the final 15F stress experiment.
# CI rerun marker: final clean 15F-D test suite. It separates:
1) symbol-spaced sparse/dense delay memory,
2) smooth continuous delay-spread memory,
3) multiscale continuous memory.

It reuses the validated 15E representation constructors and 15F train/test
discipline. No existing 15F artifact is overwritten.

Primary comparison:
    MIN_N vs log-spread IIR_N vs dense similarity control vs FIR_N

Closure controls:
    * equal state budget N in {1,2,4,8,16}
    * horizon-matched FIR at the MIN kernel 95% temporal horizon
    * compute metadata for every representation
    * full memory-family x process x SNR x modulation x seed grid

The continuous families are defined at sample resolution from analytic
continuous-time profiles; they are not selected from the observed signal.
"""
from __future__ import annotations
import csv, json, sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = spec_from_file_location("exp15e_d", ROOT / "experiments" / "15E_state_budget_sweep.py")
U15E = module_from_spec(spec)
sys.modules[spec.name] = U15E
spec.loader.exec_module(U15E)
U = U15E.U
from min.metrics.equalization import add_awgn, evm
from min.signals import generate_bpsk, generate_qpsk

SEEDS = U15E.SEEDS
BUDGETS = U15E.BUDGETS
HORIZON_BUDGET = 16
HORIZON_REFERENCE_SNR = 10.0
PROCESSES = U15E.PROCESSES
SCALE_FACTORS = (4.0, 12.0, 32.0)
SNR_DB = (0.0, 10.0, 20.0)
MODULATIONS = ("BPSK", "QPSK")
NUM_TRAIN_SYMBOLS = 512
NUM_TEST_SYMBOLS = 512
SPS = 16
SYMBOL_RATE = 100.0
DT = U15E.DT
READOUT_RIDGE = U.READOUT_RIDGE
NOISE_SEED_OFFSET = U15E.NOISE_SEED_OFFSET
CHANNEL_FAMILIES = ("sparse_taps", "dense_taps", "diffuse_exp", "diffuse_gamma", "continuous_multiscale")

def generate(modulation: str, seed: int):
    fn = {"BPSK": generate_bpsk, "QPSK": generate_qpsk}[modulation]
    return fn(NUM_TRAIN_SYMBOLS + NUM_TEST_SYMBOLS, samples_per_symbol=SPS,
              symbol_rate=SYMBOL_RATE, seed=seed)

def channel_kernel(family: str, scale: float) -> np.ndarray:
    # The support is deliberately expressed in physical sample time.
    t = np.arange(0.0, 0.80, DT)
    if family == "sparse_taps":
        h = np.zeros_like(t)
        for d, a in ((0, 1.0), (1, 0.55), (2, 0.30)):
            h[int(d * SPS)] = a
    elif family == "dense_taps":
        h = np.zeros_like(t)
        delays = np.arange(0, 16) * SPS
        h[delays] = np.exp(-np.arange(16) / 4.0)
    elif family == "diffuse_exp":
        tau = 0.11 / scale * 4.0
        h = np.exp(-t / max(tau, DT))
    elif family == "diffuse_gamma":
        tau = 0.18 / scale * 4.0
        u = t / max(tau, DT)
        h = u * np.exp(-u)
    elif family == "continuous_multiscale":
        tau1 = 0.035 / scale * 4.0
        tau2 = 0.30 / scale * 4.0
        h = 0.62 * np.exp(-t / max(tau1, DT)) + 0.38 * np.exp(-t / max(tau2, DT))
    else:
        raise ValueError(family)
    h = np.maximum(h, 0.0)
    h /= max(float(h.sum()), np.finfo(float).tiny)
    return h

def apply_channel(x: np.ndarray, family: str, scale: float) -> np.ndarray:
    h = channel_kernel(family, scale)
    y = fftconvolve(np.asarray(x, dtype=complex), h, mode="full")[:len(x)]
    return y

def linear_fit(X, y):
    X = np.asarray(X, dtype=complex); y = np.asarray(y, dtype=complex)
    A = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    G = A.conj().T @ A
    G.flat[::G.shape[0] + 1] += READOUT_RIDGE
    return np.linalg.solve(G, A.conj().T @ y)

def linear_predict(X, beta):
    return np.column_stack([np.ones(X.shape[0], dtype=complex), X]) @ beta

def bits(modulation, symbols):
    s = np.asarray(symbols)
    if modulation == "BPSK":
        return (np.real(s) >= 0).astype(np.uint8)[:, None]
    phase = np.mod(np.angle(s) - np.pi / 4.0, 2*np.pi)
    idx = np.mod(np.rint(phase/(np.pi/2)).astype(int), 4)
    return np.column_stack(((idx >> 1) & 1, idx & 1)).astype(np.uint8)

def decide(modulation, estimate):
    e = np.asarray(estimate)
    if modulation == "BPSK":
        return np.where(np.real(e) >= 0.0, 1.0, -1.0).astype(complex)
    const = np.exp(1j*(np.pi/4 + np.arange(4)*np.pi/2))
    return const[np.argmin(np.abs(e[:,None] - const[None,:]), axis=1)]

def metrics(X, tx, modulation):
    beta = linear_fit(X[:NUM_TRAIN_SYMBOLS], tx[:NUM_TRAIN_SYMBOLS])
    est = linear_predict(X[NUM_TRAIN_SYMBOLS:], beta)
    ref = tx[NUM_TEST_SYMBOLS*0 + NUM_TRAIN_SYMBOLS:]
    dec = decide(modulation, est)
    return {
        "ber": float(np.mean(bits(modulation, ref) != bits(modulation, dec))),
        "ser": float(np.mean(dec != ref)),
        "evm_percent": float(100.0 * evm(ref, est)),
        "nmse": float(np.mean(np.abs(est-ref)**2) / max(np.mean(np.abs(ref)**2), np.finfo(float).tiny)),
    }

def fir_states(x, taps):
    x = np.asarray(x, dtype=complex)
    out = np.zeros((x.size, taps), dtype=complex)
    for k in range(taps):
        out[k:, k] = x[:x.size-k] if k else x
    return out

def min_horizon_samples(gammas, weights=None, fraction=0.95):
    w = np.ones(len(gammas), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w /= w.sum()
    # Continuous exponential kernel integral from 0..T is 1-exp(-gamma*T).
    t = np.linspace(0.0, max(2.0, 12.0/float(gammas.min())), 20000)
    mass = np.sum(w[None,:] * (1.0 - np.exp(-t[:,None]*gammas[None,:])), axis=1)
    idx = int(np.searchsorted(mass, fraction))
    return max(1, int(np.ceil(t[min(idx, len(t)-1)] / DT)) + 1)

def representation_cost(name, n):
    if name.startswith("fir"):
        return {"state_dimension": int(n), "ops_per_sample": int(n), "parameters": int(n)}
    if name.startswith("dense"):
        return {"state_dimension": int(n), "ops_per_sample": int(n*n + 2*n), "parameters": int(n*n + 2*n)}
    return {"state_dimension": int(n), "ops_per_sample": int(3*n), "parameters": int(2*n)}

def run_case(modulation, process, family, factor, snr_db, seed, budget):
    sig = generate(modulation, seed)
    ch = apply_channel(sig.record.samples, family, factor)
    noisy = add_awgn(ch, snr_db, np.random.default_rng(NOISE_SEED_OFFSET + seed*1000 + int(snr_db) + int(factor)))
    idx, tx = sig.symbol_indices, sig.symbols
    reps, gammas, spread = U15E.representation_specs(process, factor, noisy, budget)
    rows = []
    for name, state in reps.items():
        m = metrics(state[idx], tx, modulation)
        rows.append({
            "experiment": "15F_D_memory_geometry_closure",
            "modulation": modulation, "process": process, "channel_family": family,
            "scale_factor": float(factor), "snr_db": float(snr_db), "seed": int(seed),
            "representation": name, "state_budget": int(budget),
            **m, **representation_cost(name, state.shape[1]),
            "temporal_horizon_samples": int(state.shape[1]),
        })
    # Horizon-matched FIR is a diagnostic, not a second full grid. It is
    # evaluated only at N=16 and at the reference SNR to keep the control
    # computationally commensurate with the main experiment. The FIR uses
    # symbol-spaced taps, so its state count is the 95% MIN kernel horizon
    # measured in symbols rather than raw ADC samples.
    if budget == HORIZON_BUDGET and float(snr_db) == HORIZON_REFERENCE_SNR:
        k95_samples = min_horizon_samples(gammas)
        k95 = max(1, int(np.ceil(k95_samples / SPS)))
        symbol_stream = noisy[idx]
        hf = fir_states(symbol_stream, k95)
        m = metrics(hf, tx, modulation)
        rows.append({
            "experiment": "15F_D_memory_geometry_closure",
            "modulation": modulation, "process": process, "channel_family": family,
            "scale_factor": float(factor), "snr_db": float(snr_db), "seed": int(seed),
            "representation": f"fir_horizon95_{k95}", "state_budget": int(budget),
            **m, **representation_cost("fir", k95),
            "temporal_horizon_samples": int(k95), "horizon_match_fraction": 0.95,
            "horizon_control_scope": "N=16, SNR=10 dB, symbol-spaced FIR",
        })
    return rows

def summarize(rows):
    keys = ("channel_family","process","representation","state_budget","scale_factor","snr_db")
    out=[]
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for key,b in sorted(groups.items()):
        out.append({"experiment":"15F_D_memory_geometry_closure", **dict(zip(keys,key)),
                    "n":len(b), "mean_ber":float(np.mean([r["ber"] for r in b])),
                    "mean_ser":float(np.mean([r["ser"] for r in b])),
                    "mean_evm_percent":float(np.mean([r["evm_percent"] for r in b])),
                    "mean_nmse":float(np.mean([r["nmse"] for r in b])),
                    "mean_ops_per_sample":float(np.mean([r["ops_per_sample"] for r in b]))})
    return out

def paired(rows):
    groups={}
    for r in rows:
        if r["representation"].startswith("fir_horizon95"): continue
        key=(r["channel_family"],r["process"],r["scale_factor"],r["snr_db"],r["seed"],r["state_budget"])
        groups.setdefault(key,{})[r["representation"]]=r
    out=[]
    for key,reps in sorted(groups.items()):
        b=key[-1]; mn=reps.get(f"min_{b}")
        if not mn: continue
        for c in (f"iir_logspread_{b}",f"dense_ss_{b}",f"fir_{b}"):
            if c in reps:
                out.append({"experiment":"15F_D_memory_geometry_closure",
                    "channel_family":key[0],"process":key[1],"scale_factor":float(key[2]),
                    "snr_db":float(key[3]),"seed":int(key[4]),"state_budget":int(b),
                    "control":c,"delta_ber_min_minus_control":float(mn["ber"]-reps[c]["ber"]),
                    "delta_ser_min_minus_control":float(mn["ser"]-reps[c]["ser"]),
                    "delta_evm_min_minus_control":float(mn["evm_percent"]-reps[c]["evm_percent"]),
                    "delta_nmse_min_minus_control":float(mn["nmse"]-reps[c]["nmse"])})
    return out

def main():
    eq=U.validate_repository_equivalence()
    rows=[]
    for seed in SEEDS:
      for modulation in MODULATIONS:
       for process in PROCESSES:
        for family in CHANNEL_FAMILIES:
         for factor in SCALE_FACTORS:
          for snr in SNR_DB:
           for budget in BUDGETS:
            rows.extend(run_case(modulation,process,family,factor,snr,seed,budget))
    out=ROOT/"experiments"/"results"; out.mkdir(exist_ok=True)
    cases=out/"15F_D_memory_geometry_closure_cases.csv"
    summary=out/"15F_D_memory_geometry_closure_summary.csv"
    deltas=out/"15F_D_memory_geometry_closure_paired_deltas.csv"
    meta=out/"15F_D_memory_geometry_closure_summary.json"
    for p,data in ((cases,rows),(summary,summarize(rows)),(deltas,paired(rows))):
        with p.open("w",newline="") as f:
            fieldnames = sorted({key for row in data for key in row})
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    md={"experiment":"15F_D_memory_geometry_closure",
        "question":"Does the 15F result change when channel memory is distributed continuously rather than concentrated in symbol-spaced taps?",
        "channel_families":CHANNEL_FAMILIES,"budgets":list(BUDGETS),"processes":list(PROCESSES),
        "scale_factors":list(SCALE_FACTORS),"snr_db":list(SNR_DB),"modulations":list(MODULATIONS),
        "seeds":list(SEEDS),"train_symbols":NUM_TRAIN_SYMBOLS,"test_symbols":NUM_TEST_SYMBOLS,
        "representations":["min_N","iir_logspread_N","dense_ss_N","fir_N","fir_horizon95_K"],
        "horizon_control":"At N=16 and 10 dB reference SNR, FIR_K is chosen from analytic MIN kernel 95% integrated mass and converted to symbol-spaced taps; K is not fitted to channel output.",
        "repository_state_equivalence_relative_error":eq,
        "case_rows":len(rows),"paired_delta_rows":len(paired(rows)),
        "interpretation_rule":"No family is selected post hoc; all channel families and all processes are reported.",
        "note":"Continuous families are analytic sample-resolution approximations to continuous-time delay profiles; sparse/dense families are explicit symbol-spaced controls."}
    meta.write_text(json.dumps(md,indent=2)+"\n")
    print(json.dumps(md,indent=2))

if __name__=="__main__":
    main()
