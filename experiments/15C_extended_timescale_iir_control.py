"""Experiment 15C: extended timescale sweep with a one-pole exponential control.

The MIN/SOE representation uses 16 positive uniform exponential modes. The
control is a conventional single-pole exponential/IIR state whose decay rate
is chosen to have the same kernel t50 as the corresponding MIN kernel. This
tests whether any performance advantage persists beyond ordinary one-pole
filtering when effective temporal scale is matched.

The sweep extends beyond 15B-3 into the 15B-2 long-process displaced regime.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
from scipy.signal import lfilter

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("min_exp15_utils",ROOT/"experiments"/"15_utils.py")
U=module_from_spec(spec); sys.modules[spec.name]=U; spec.loader.exec_module(U)

PROCESSES=("short","multiscale","long","hidden_mix")
TASKS=("prediction","latent_estimation")
SEEDS=U.SEEDS
SNR_DB=U.SNR_DB
PCA_DIMENSIONS=U.PCA_DIMENSIONS
MODES=U.MODES
DT=U.DT
TASK_TOLERANCE=U.TASK_TOLERANCE
N=8192
TRAIN_FRAC=0.5
NOISE_SEED_OFFSET=153_000
SCALE_FACTORS=(4.0,8.0,12.0,20.0,32.0,64.0)

PROCESS_PARAMS={
    "short":{"taus":(0.025,),"weights":(1.0,)},
    "multiscale":{"taus":(0.025,0.50),"weights":(0.65,0.35)},
    "long":{"taus":(0.50,),"weights":(1.0,)},
    "hidden_mix":{"taus":(0.025,0.50),"weights":(0.50,0.50)},
}

def ar_component(n,tau,rng):
    a=float(np.exp(-DT/tau)); eps=rng.normal(size=n)
    q=np.empty(n); q[0]=eps[0]
    scale=np.sqrt(max(1-a*a,np.finfo(float).tiny))
    for i in range(1,n): q[i]=a*q[i-1]+scale*eps[i]
    return q

def generate_process(name,seed):
    p=PROCESS_PARAMS[name]; rng=np.random.default_rng(seed)
    components=np.column_stack([ar_component(N,t,rng) for t in p["taus"]])
    w=np.asarray(p["weights"],float); w/=w.sum()
    x=components@w; x/=max(float(np.std(x)),np.finfo(float).tiny)
    return x.astype(float),components.astype(float)

def add_observation_noise(x,snr_db,seed):
    rng=np.random.default_rng(seed)
    sigma=np.sqrt(float(np.mean(x*x))/(10.0**(snr_db/10.0)))
    return x+sigma*rng.normal(size=len(x))

def sweep_rates(process,factor):
    taus=np.asarray(PROCESS_PARAMS[process]["taus"],float)/factor
    clusters=[]
    if len(taus)==1:
        gammas=taus[0]**-1*np.exp(np.linspace(-0.15,0.15,MODES))
    else:
        per=MODES//len(taus)
        for tau in taus:
            clusters.append(tau**-1*np.exp(np.linspace(-0.15,0.15,per)))
        gammas=np.concatenate(clusters)
    return gammas

def one_pole_state(x,gamma):
    # Exact causal piecewise-linear one-pole realization, matching the MIN
    # state convention and using the same input samples.
    decay=np.exp(-gamma*DT)
    bc=1.0/gamma-(1.0-decay)/(DT*gamma**2)
    bp=(1.0-decay)/gamma-bc
    forcing=np.empty(len(x),complex); forcing[0]=0
    forcing[1:]=bc*x[1:]+bp*x[:-1]
    return lfilter([1.0],[1.0,-decay],forcing)[:,None]

def one_pole_gamma_for_t50(t50):
    return np.log(2.0)/max(float(t50),np.finfo(float).tiny)

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
    full=curve[min(PCA_DIMENSIONS) if len(PCA_DIMENSIONS)==1 else MODES]
    d=next((k for k in PCA_DIMENSIONS if curve[k]<=TASK_TOLERANCE*full),None)
    return curve,singular,d

def evaluate_representation(X,target,split):
    out={}
    for task in TASKS:
        curve,singular,d=fit_curve(X[:split],target[:split],X[split:],target[split:])
        out[task]=(curve,singular,d)
    return out

def run_case(process,factor,snr_db,seed):
    x,latent=generate_process(process,seed)
    y=add_observation_noise(x,snr_db,seed+NOISE_SEED_OFFSET)
    split=int(N*TRAIN_FRAC)
    gammas=sweep_rates(process,factor)
    weights=U.uniform_weights()
    q=U.exact_piecewise_linear_states(y,gammas,U.state_coefficients(gammas))
    desc=U.kernel_descriptors(gammas,weights)
    t50=desc["kernel_t50_s"]
    pole_gamma=one_pole_gamma_for_t50(t50)
    pole=one_pole_state(y,pole_gamma)
    reps={"min_16":q,"one_pole_t50":pole,"observation":y[:,None].astype(complex)}
    rows=[]; pca_rows=[]
    for name,X in reps.items():
        for task in TASKS:
            Xi=X[:-1] if task=="prediction" else X
            target=x[1:] if task=="prediction" else latent[:,0]
            curve,singular,d=fit_curve(Xi[:split],target[:split],Xi[split:],target[split:])
            cond=state_condition_number(X[:split])
            for k,v in curve.items():
                pca_rows.append({"experiment":"15C_extended_timescale_iir_control",
                    "process":process,"representation":name,"task":task,"scale_factor":factor,
                    "snr_db":float(snr_db),"seed":int(seed),"retained_dimension":int(k),
                    "heldout_nmse":float(v),"pca_explained_energy":U.pca_energy(singular,k),
                    "state_condition_number":cond,"min_kernel_t50_s":t50,
                    "one_pole_gamma":pole_gamma,**desc})
            rows.append({"experiment":"15C_extended_timescale_iir_control",
                "process":process,"representation":name,"task":task,"scale_factor":factor,
                "snr_db":float(snr_db),"seed":int(seed),"full_state_nmse":float(curve[min(PCA_DIMENSIONS) if len(PCA_DIMENSIONS)==1 else MODES]),
                "d_task_10pct":d if d is not None else -1,"state_condition_number":cond,
                "min_kernel_t50_s":t50,"one_pole_gamma":pole_gamma,**desc})
    return rows,pca_rows

def summarize(rows):
    keys=("process","representation","task","scale_factor","snr_db")
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,b in sorted(groups.items()):
        process,rep,task,factor,snr=key
        ds=[r["d_task_10pct"] for r in b if r["d_task_10pct"]>0]
        out.append({"process":process,"representation":rep,"task":task,
            "scale_factor":float(factor),"snr_db":float(snr),"n":len(b),
            "mean_full_state_nmse":float(np.mean([r["full_state_nmse"] for r in b])),
            "std_full_state_nmse":float(np.std([r["full_state_nmse"] for r in b],ddof=1)),
            "mean_d_task":float(np.mean(ds)) if ds else float("nan"),
            "median_state_condition_number":float(np.median([r["state_condition_number"] for r in b])),
            "min_kernel_t50_s":float(np.mean([r["min_kernel_t50_s"] for r in b])),
            "one_pole_gamma":float(np.mean([r["one_pole_gamma"] for r in b]))})
    return out

def main():
    eq=U.validate_repository_equivalence()
    rows=[]; pca_rows=[]
    for seed in SEEDS:
        for process in PROCESSES:
            for factor in SCALE_FACTORS:
                for snr in SNR_DB:
                    r,p=run_case(process,factor,snr,seed); rows.extend(r); pca_rows.extend(p)
    expected=len(SEEDS)*len(PROCESSES)*len(SCALE_FACTORS)*len(SNR_DB)*len(TASKS)*3
    assert len(rows)==expected
    assert len(pca_rows)==expected*len(PCA_DIMENSIONS)
    out=ROOT/"experiments"/"results"; out.mkdir(parents=True,exist_ok=True)
    files={"cases":out/"15C_extended_timescale_iir_control_cases.csv",
           "pca":out/"15C_extended_timescale_iir_control_pca_results.csv",
           "summary":out/"15C_extended_timescale_iir_control_summary.csv",
           "metadata":out/"15C_extended_timescale_iir_control_summary.json"}
    for key,data in (("cases",rows),("pca",pca_rows)):
        with files[key].open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary=summarize(rows)
    with files["summary"].open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta={"experiment":"15C_extended_timescale_iir_control","processes":PROCESSES,
          "tasks":TASKS,"scale_factors":list(SCALE_FACTORS),"snr_db":list(SNR_DB),
          "seeds":list(SEEDS),"N":N,"nominal_modes":MODES,
          "representations":["min_16","one_pole_t50","observation"],
          "controls":{"same_observation":True,"same_split":True,"same_readout":True,
                      "min_uniform_positive_weights":True,"one_pole_matches_min_t50":True},
          "repository_state_equivalence_relative_error":eq,
          "rows":len(rows),"pca_rows":len(pca_rows)}
    files["metadata"].write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))

if __name__=="__main__": main()
