"""Experiment 13C-2: finite/noisy environment observations -> kernel uncertainty."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from min.signals import generate_16qam, generate_bpsk, generate_qpsk
SIGNALS={"BPSK":generate_bpsk,"QPSK":generate_qpsk,"16QAM":generate_16qam}
ENVIRONMENTS=("white_limit","short","multiscale","powerlaw","squared_exp")
SEEDS=tuple(range(4)); OBS_LENGTHS=(512,2048,8192); SNR_DB=(0.0,10.0,20.0,30.0)
NUM_SYMBOLS,SPS,SYMBOL_RATE=512,16,100.0; SAMPLE_RATE=SYMBOL_RATE*SPS; DT=1/SAMPLE_RATE
HORIZON_S=NUM_SYMBOLS/SYMBOL_RATE; FIT_HORIZON_S=0.08; GAMMAS=np.geomspace(0.5,100.0,16)
def env(t,name):
    if name=="white_limit": return np.exp(-t/.005)
    if name=="short": return np.exp(-t/.05)
    if name=="multiscale": return .65*np.exp(-t/.03)+.35*np.exp(-t/.7)
    if name=="powerlaw": return (1+t/.2)**(-.7)
    if name=="squared_exp": return np.exp(-(t/.15)**2)
    raise ValueError(name)
def fit(t,y):
    w,_=nnls(np.exp(-np.outer(t,GAMMAS)),np.maximum(y,0)); w/=max(w.sum(),1e-300); return w
def synth(name,n,seed):
    rng=np.random.default_rng(seed); c=env(np.arange(n)*DT,name); circ=np.r_[c,c[-2:0:-1]]
    s=np.maximum(np.real(np.fft.rfft(circ)),0); z=rng.normal(size=s.size)+1j*rng.normal(size=s.size); z[0]=rng.normal()
    if circ.size%2==0: z[-1]=rng.normal()
    x=np.fft.irfft(np.sqrt(s)*z,n=circ.size)[:n]; return x/max(np.std(x),1e-300)
def noisy(x,snr,seed):
    rng=np.random.default_rng(seed+1000003)
    if math.isinf(snr): return x.copy()
    return x+rng.normal(scale=math.sqrt(np.mean(x*x)/10**(snr/10)),size=x.size)
def cov_est(x,maxlag):
    x=x-np.mean(x); n=x.size; c=np.array([np.dot(x[:n-k],x[k:])/max(n-k,1) for k in range(maxlag+1)])
    return np.arange(maxlag+1)*DT,c/max(c[0],1e-300)
def eigmetrics(a,p):
    e=np.sort(np.maximum(np.real(np.linalg.eigvalsh(a)),0))[::-1]; s=e.sum()
    if s<=0:return {p+"_participation_dimension":0.,p+"_entropy_dimension":0.}
    q=e/s; return {p+"_participation_dimension":float(1/(q@q)),p+"_entropy_dimension":float(np.exp(-np.sum(q[q>0]*np.log(q[q>0]))))}
def gfe(w):
    a=w>1e-12; h=-np.sum(w[a]*np.log(w[a])); g=GAMMAS[a]; scale=math.log10(g.max()/g.min()) if len(g)>1 else 0.
    return {"gfe_m_cap_s":float(np.sum(w/GAMMAS**2)/np.sum(w/GAMMAS)),"gfe_m_scale_decades":scale,
            "gfe_m_res_modes_per_decade":float(a.sum()/scale) if scale else float("nan"),"gfe_h_mem_nats":float(h),
            "gfe_entropy_effective_count":float(np.exp(h)),"gfe_d_eff":float(16*np.exp(h)),"nonzero_kernel_modes":int(a.sum())}
def basis(w):
    z=GAMMAS[:,None]+GAMMAS[None,:]; G=-np.expm1(-z*HORIZON_S)/z; W=np.sqrt(w)[:,None]*G*np.sqrt(w)[None,:]
    o=eigmetrics(W,"weighted_basis"); d=np.sqrt(np.maximum(np.diag(W),0)); C=W/np.outer(np.maximum(d,1e-300),np.maximum(d,1e-300))
    o["weighted_basis_max_coherence"]=float(np.max(np.abs(C-np.eye(16)))); return o
def states(x):
    p=np.exp(-GAMMAS*DT); inc=-np.expm1(-GAMMAS*DT)/GAMMAS; q=np.zeros(16,complex); out=np.empty((len(x),16),complex)
    for i,v in enumerate(x.astype(complex)): q=p*q+inc*v; out[i]=q
    return out
def stategeom(q,w):
    z=q-q.mean(0); C=z.conj().T@z/max(len(q)-1,1); out=eigmetrics(C,"state"); z=z*np.sqrt(w); C=z.conj().T@z/max(len(q)-1,1); out.update(eigmetrics(C,"weighted_state")); return out
def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
def oracle(name):
    t=np.linspace(0,HORIZON_S,1024); w=fit(t,env(t,name)); return w,{**gfe(w),**basis(w)}
OR={e:oracle(e) for e in ENVIRONMENTS}
def main():
    rows=[]
    for e in ENVIRONMENTS:
      ow,om=OR[e]
      for seed in SEEDS:
       for n in OBS_LENGTHS:
        latent=synth(e,n,seed)
        for snr in SNR_DB:
         x=noisy(latent,snr,seed+n); maxlag=int(FIT_HORIZON_S/DT); t,c=cov_est(x,maxlag); sel=t<=FIT_HORIZON_S; t,c=t[sel],c[sel]
         w=fit(t,c); fitted=np.exp(-np.outer(t,GAMMAS))@w
         base={"experiment":"13C-2_finite_observation_kernel_uncertainty","environment":e,"seed":seed,"observation_length":n,"observation_duration_s":n*DT,"snr_db":snr,"fit_horizon_s":FIT_HORIZON_S,
               "covariance_relative_l2_error":rel(c,env(t,e)),"covariance_integral_relative_error":abs(trapezoid(c,t)-trapezoid(env(t,e),t))/max(abs(trapezoid(env(t,e),t)),1e-12),
               "kernel_fit_relative_l2":rel(fitted,c),"kernel_fit_r2":float(1-np.sum((fitted-c)**2)/max(np.sum((c-c.mean())**2),1e-300)),
               "kernel_weight_relative_l2_error":rel(w,ow),"kernel_weight_max_abs_error":float(np.max(abs(w-ow)))}
         base.update(gfe(w)); base.update(basis(w))
         for k,v in om.items(): base["oracle_"+k]=v
         base["d_eff_relative_error"]=abs(base["gfe_d_eff"]-om["gfe_d_eff"])/max(abs(om["gfe_d_eff"]),1e-300)
         base["weighted_basis_relative_error"]=abs(base["weighted_basis_participation_dimension"]-om["weighted_basis_participation_dimension"])/max(om["weighted_basis_participation_dimension"],1e-300)
         for s,gen in SIGNALS.items():
          sig=gen(NUM_SYMBOLS,samples_per_symbol=SPS,symbol_rate=SYMBOL_RATE,seed=seed); q=states(np.asarray(sig.record.samples,dtype=complex)); sg=stategeom(q,w); oq=stategeom(q,ow)
          r=dict(base); r["signal"]=s; r.update(sg); r["oracle_weighted_state_participation_dimension"]=oq["weighted_state_participation_dimension"]; r["oracle_weighted_state_entropy_dimension"]=oq["weighted_state_entropy_dimension"]
          r["weighted_state_relative_error"]=abs(r["weighted_state_participation_dimension"]-oq["weighted_state_participation_dimension"])/max(oq["weighted_state_participation_dimension"],1e-300); rows.append(r)
    expected=len(ENVIRONMENTS)*len(SEEDS)*len(OBS_LENGTHS)*len(SNR_DB)*len(SIGNALS); assert len(rows)==expected,(len(rows),expected)
    out=ROOT/"experiments"/"results"; out.mkdir(parents=True,exist_ok=True); rp=out/"13C-2_finite_observation_kernel_uncertainty_results.csv"
    with rp.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    keys=("environment","observation_length","snr_db"); metrics=("covariance_relative_l2_error","kernel_weight_relative_l2_error","kernel_fit_relative_l2","gfe_d_eff","weighted_basis_participation_dimension","weighted_state_participation_dimension","d_eff_relative_error","weighted_basis_relative_error","weighted_state_relative_error"); groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    sm=[]
    for key,b in sorted(groups.items()):
      d=dict(zip(keys,key)); d["rows"]=len(b); d["independent_environment_seeds"]=len(SEEDS)
      for m in metrics: d[m+"_mean"]=float(np.mean([r[m] for r in b])); d[m+"_std"]=float(np.std([r[m] for r in b],ddof=1))
      sm.append(d)
    sp=out/"13C-2_finite_observation_kernel_uncertainty_summary.csv"
    with sp.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sm[0]); w.writeheader(); w.writerows(sm)
    meta={"experiment":"13C-2_finite_observation_kernel_uncertainty","rows":len(rows),"unique_environment_realizations":len(ENVIRONMENTS)*len(SEEDS)*len(OBS_LENGTHS)*len(SNR_DB),
          "grid":{"signals":list(SIGNALS),"environments":list(ENVIRONMENTS),"seeds":list(SEEDS),"observation_lengths":list(OBS_LENGTHS),"snr_db":list(SNR_DB),"dictionary_mode_count":16,"gamma_range_s_inverse":[.5,100.]},
          "chain":["finite noisy environment realization","sample covariance estimation","positive SOE identification by fixed dictionary NNLS","GFE/GGFE descriptors","weighted basis geometry","MIN state geometry"],
          "boundary":["fixed 16-rate dictionary inherited from 13C-1","signal families share each environment estimate and are not independent environment evidence","model mismatch deferred to 13C-3","no downstream task metric"]}
    (out/"13C-2_finite_observation_kernel_uncertainty_summary.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
