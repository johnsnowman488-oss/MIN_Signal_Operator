"""Experiment 15D: equal-budget temporal representation comparison.

Question:
    At the same 16-state representation budget, does MIN's process-adapted
    multi-exponential geometry retain a task-performance advantage over
    conventional temporal representations?

Representations:
    min_16              process-clustered 16-mode MIN/SOE
    iir_logspread_16    conventional 16-pole logarithmic filter bank spanning
                         the same endpoint rate range
    dense_ss_16         dense state-space similarity transform of that same
                         16-pole bank (coordinate-control)
    fir_16              16-sample finite-history delay line

All representations see the identical noisy observation, use the identical
train/test split, linear readout, PCA sweep, processes, SNRs, scales, and seeds.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
from scipy.signal import lfilter

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("exp15c",ROOT/"experiments"/"15C_extended_timescale_iir_control.py")
U15C=module_from_spec(spec); sys.modules[spec.name]=U15C; spec.loader.exec_module(U15C)
U=U15C.U

PROCESSES=U15C.PROCESSES
TASKS=U15C.TASKS
SEEDS=U15C.SEEDS
SNR_DB=U15C.SNR_DB
PCA_DIMENSIONS=U15C.PCA_DIMENSIONS
MODES=U15C.MODES
DT=U15C.DT
N=U15C.N
TRAIN_FRAC=U15C.TRAIN_FRAC
SCALE_FACTORS=U15C.SCALE_FACTORS

def min_rates(process,factor):
    return U15C.sweep_rates(process,factor)

def logspread_states(x,gammas):
    return U.exact_piecewise_linear_states(x,gammas,U.state_coefficients(gammas))

def fir_states(x,taps=MODES):
    x=np.asarray(x,dtype=complex)
    out=np.zeros((x.size,taps),dtype=complex)
    for k in range(taps):
        if k==0: out[:,k]=x
        else: out[k:,k]=x[:-k]
    return out

def dense_state_similarity(x,gammas,seed=1504):
    """Stable dense realization similar to the diagonal pole bank.

    A random orthogonal similarity transform preserves the underlying poles
    exactly, so any difference from iir_logspread_16 is numerical only. This
    is a coordinate-control rather than a claim of a distinct memory kernel.
    """
    z=logspread_states(x,gammas)
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(len(gammas),len(gammas))))
    return z @ q

def state_condition_number(X):
    centered=X-np.mean(X,axis=0,keepdims=True)
    s=np.linalg.svd(centered,compute_uv=False)
    pos=s[s>np.finfo(float).eps*max(s[0],1.0)]
    return float(s[0]/pos[-1]) if pos.size else float("inf")

def fit_curve(Xtr,target_tr,Xte,target_te):
    mean,singular,components=U.complex_pca_fit(Xtr.astype(complex))
    curve={}
    for k in PCA_DIMENSIONS:
        ztr=U.pca_project(Xtr.astype(complex),mean,components,k)
        zte=U.pca_project(Xte.astype(complex),mean,components,k)
        beta=U.linear_readout_fit(ztr,target_tr.astype(complex))
        curve[k]=U.nmse(target_te,U.linear_readout_predict(zte,beta))
    full=curve[MODES]
    d=next((k for k in PCA_DIMENSIONS if curve[k]<=1.10*full),None)
    return curve,singular,d

def representation_specs(process,factor,y):
    mg=min_rates(process,factor)
    spread=np.geomspace(float(mg.min()),float(mg.max()),MODES)
    return {
        "min_16": U.exact_piecewise_linear_states(y,mg,U.state_coefficients(mg)),
        "iir_logspread_16": logspread_states(y,spread),
        "dense_ss_16": dense_state_similarity(y,spread),
        "fir_16": fir_states(y,MODES),
    }, mg, spread

COSTS={
    "min_16":{"state_dimension":16,"parameters":32,"ops_per_sample":48,"memory_values":16},
    "iir_logspread_16":{"state_dimension":16,"parameters":32,"ops_per_sample":48,"memory_values":16},
    "dense_ss_16":{"state_dimension":16,"parameters":288,"ops_per_sample":288,"memory_values":16},
    "fir_16":{"state_dimension":16,"parameters":16,"ops_per_sample":16,"memory_values":16},
}

def run_case(process,factor,snr_db,seed):
    x,latent=U15C.generate_process(process,seed)
    y=U15C.add_observation_noise(x,snr_db,seed+U15C.NOISE_SEED_OFFSET)
    split=int(N*TRAIN_FRAC)
    reps,mg,spread=representation_specs(process,factor,y)
    rows=[]; pca_rows=[]
    for name,X in reps.items():
        geometry=U.spectrum_geometry(X[:split])
        cond=state_condition_number(X[:split])
        for task in TASKS:
            Xi=X[:-1] if task=="prediction" else X
            target=x[1:] if task=="prediction" else latent[:,0]
            curve,singular,d=fit_curve(Xi[:split],target[:split],Xi[split:],target[split:])
            for k,v in curve.items():
                pca_rows.append({
                    "experiment":"15D_equal_budget_comparison",
                    "process":process,"representation":name,"task":task,
                    "scale_factor":factor,"snr_db":float(snr_db),"seed":int(seed),
                    "retained_dimension":int(k),"heldout_nmse":float(v),
                    "pca_explained_energy":U.pca_energy(singular,k),
                    "state_condition_number":cond,
                    **geometry,**COSTS[name],
                    "min_rate_min":float(mg.min()),"min_rate_max":float(mg.max()),
                    "spread_rate_min":float(spread.min()),"spread_rate_max":float(spread.max())
                })
            rows.append({
                "experiment":"15D_equal_budget_comparison",
                "process":process,"representation":name,"task":task,
                "scale_factor":factor,"snr_db":float(snr_db),"seed":int(seed),
                "full_state_nmse":float(curve[MODES]),
                "d_task_10pct":d if d is not None else -1,
                "state_condition_number":cond,**geometry,**COSTS[name],
                "min_rate_min":float(mg.min()),"min_rate_max":float(mg.max()),
                "spread_rate_min":float(spread.min()),"spread_rate_max":float(spread.max())
            })
    return rows,pca_rows

def summarize(rows):
    keys=("process","representation","task","scale_factor","snr_db")
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,b in sorted(groups.items()):
        process,rep,task,factor,snr=key
        ds=[r["d_task_10pct"] for r in b if r["d_task_10pct"]>0]
        out.append({
            "process":process,"representation":rep,"task":task,
            "scale_factor":float(factor),"snr_db":float(snr),"n":len(b),
            "mean_full_state_nmse":float(np.mean([r["full_state_nmse"] for r in b])),
            "std_full_state_nmse":float(np.std([r["full_state_nmse"] for r in b],ddof=1)),
            "mean_d_task":float(np.mean(ds)) if ds else float("nan"),
            "median_state_condition_number":float(np.median([r["state_condition_number"] for r in b])),
            "mean_state_participation_dimension":float(np.mean([r["state_participation_dimension"] for r in b])),
            "mean_state_entropy_dimension":float(np.mean([r["state_entropy_dimension"] for r in b])),
            "state_dimension":b[0]["state_dimension"],
            "parameters":b[0]["parameters"],
            "ops_per_sample":b[0]["ops_per_sample"],
            "memory_values":b[0]["memory_values"]
        })
    return out

def main():
    eq=U.validate_repository_equivalence()
    rows=[]; pca_rows=[]
    for seed in SEEDS:
        for process in PROCESSES:
            for factor in SCALE_FACTORS:
                for snr in SNR_DB:
                    r,p=run_case(process,factor,snr,seed); rows.extend(r); pca_rows.extend(p)
    expected=len(SEEDS)*len(PROCESSES)*len(SCALE_FACTORS)*len(SNR_DB)*len(TASKS)*4
    assert len(rows)==expected
    assert len(pca_rows)==expected*len(PCA_DIMENSIONS)
    out=ROOT/"experiments"/"results"; out.mkdir(parents=True,exist_ok=True)
    files={
        "cases":out/"15D_equal_budget_comparison_cases.csv",
        "pca":out/"15D_equal_budget_comparison_pca_results.csv",
        "summary":out/"15D_equal_budget_comparison_summary.csv",
        "metadata":out/"15D_equal_budget_comparison_summary.json"}
    for key,data in (("cases",rows),("pca",pca_rows)):
        with files[key].open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary=summarize(rows)
    with files["summary"].open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta={
        "experiment":"15D_equal_budget_comparison",
        "processes":PROCESSES,"tasks":TASKS,"scale_factors":list(SCALE_FACTORS),
        "snr_db":list(SNR_DB),"seeds":list(SEEDS),"N":N,"nominal_modes":MODES,
        "representations":list(COSTS),
        "controls":{"same_observation":True,"same_split":True,"same_readout":True,
                    "same_state_budget":True,"min_process_adapted":True,
                    "dense_state_is_similarity_control":True},
        "repository_state_equivalence_relative_error":eq,
        "rows":len(rows),"pca_rows":len(pca_rows)}
    files["metadata"].write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))

if __name__=="__main__": main()
