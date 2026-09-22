"""Experiment 12C: explicit SOE state/input estimation.

This experiment replaces transfer-function equalization with a structured
state realization. The sampled SOE impulse response is represented by a
fixed-pole state-space model. Channel taps are estimated from the training
prefix, then held-out symbols are recovered by regularized batch state/input
estimation. This is an offline smoother, not a causal online receiver.
"""

from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from min.metrics.equalization import add_awgn, design_fir_equalizer, apply_fir_equalizer, design_iir_equalizer, apply_iir_equalizer, evm
from min.signals import generate_16qam, generate_bpsk, generate_qpsk

SIGNALS={"BPSK":generate_bpsk,"QPSK":generate_qpsk,"16QAM":generate_16qam}
SOE_MEMORIES={"exponential_tau_0.10":(np.array([1.0]),np.array([10.0])),
              "soe_two_scale":(np.array([0.7,0.3]),np.array([2.0,30.0]))}
CHANNELS=("identity","flat_rayleigh","multipath_3tap")
RECEIVERS=("raw","fir7","iir2","soe_state")
SNR_DB=(0.,10.,20.,30.); NUM_SYMBOLS=512; TRAIN_SYMBOLS=128
SPS=16; SYMBOL_RATE=100.; SAMPLE_DT=1/(SYMBOL_RATE*SPS); SEEDS=tuple(range(5))

def bits(name,s):
    s=np.asarray(s)
    if name=="BPSK": return (np.real(s)<0).astype(np.uint8)[:,None]
    if name=="QPSK":
        q=np.mod(np.angle(s)-np.pi/4,2*np.pi); i=np.mod(np.rint(q/(np.pi/2)).astype(int),4)
        return np.column_stack(((i>>1)&1,i&1)).astype(np.uint8)
    lv=np.array([-3.,-1.,1.,3.])/np.sqrt(10)
    ii=np.argmin(abs(np.real(s)[:,None]-lv),axis=1); qq=np.argmin(abs(np.imag(s)[:,None]-lv),axis=1)
    return np.column_stack(((ii>>1)&1,ii&1,(qq>>1)&1,qq&1)).astype(np.uint8)

def decide(name,z):
    if name=="BPSK": return np.where(np.real(z)>=0,1,-1).astype(complex)
    if name=="QPSK": c=np.exp(1j*(np.pi/4+np.arange(4)*np.pi/2))
    else:
        lv=np.array([-3.,-1.,1.,3.])/np.sqrt(10); c=(lv[:,None]+1j*lv[None,:]).reshape(-1)
    return c[np.argmin(abs(np.asarray(z)[:,None]-c[None,:]),axis=1)]

def ber(name,tx,z): return float(np.mean(bits(name,tx)!=bits(name,decide(name,z))))

def memory_filter(x,mem):
    w,g=SOE_MEMORIES[mem]; l=np.arange(32*SPS)*SAMPLE_DT
    k=sum(a*np.exp(-b*l) for a,b in zip(w,g)); k/=k.sum()
    return np.convolve(x,k,mode="full")[:x.size]

def sampled_impulse(mem,n):
    x=np.zeros(n*SPS,dtype=complex); x[:SPS]=1
    y=memory_filter(x,mem)
    return y[np.arange(n)*SPS+SPS//2]

def state_model(mem,horizon=256):
    w,g=SOE_MEMORIES[mem]; m=len(w)
    h=sampled_impulse(mem,horizon)
    poles=np.exp(-g/SYMBOL_RATE)
    A=np.diag(poles.astype(complex))
    # D and B are fitted to the first m+1 Markov parameters of the actual
    # rectangular-pulse sampled realization.
    D=h[0]
    V=np.vstack([poles**n for n in range(1,m+1)])
    B=np.linalg.solve(V,h[1:m+1])
    C=np.ones((1,m),dtype=complex)
    hh=np.empty(horizon,dtype=complex); hh[0]=D
    state=B.copy()
    for n in range(1,horizon):
        hh[n]=(C@state)[0]
        state=A@state
    return A,B,C,D,hh

def toeplitz_apply(h,u):
    y=np.zeros(len(u),dtype=complex)
    for n in range(len(u)):
        k=min(n+1,len(h)); y[n]=np.dot(h[:k],u[n::-1][:k])
    return y

def ridge_solve(H,y,ridge):
    return np.linalg.solve(H.conj().T@H+ridge*np.eye(H.shape[1]),H.conj().T@y)

def conv_matrix(h,n):
    H=np.zeros((n,n),dtype=complex)
    for r in range(n):
        k=min(r+1,len(h)); H[r,:k]=h[:k][::-1]
    return H

def channel(x,name,rng):
    if name=="identity": return x,np.array([1+0j])
    if name=="flat_rayleigh":
        h=(rng.normal()+1j*rng.normal())/np.sqrt(2); h/=max(abs(h),1e-12); return h*x,np.array([h])
    ph=rng.uniform(0,2*np.pi,2); h=np.array([1,.45*np.exp(1j*ph[0]),.25*np.exp(1j*ph[1])],complex); h/=np.sqrt(np.sum(abs(h)**2))
    return np.convolve(x,h)[:len(x)],h

def estimate_channel(tx,obs,mem,taps):
    _,_,_,_,h=state_model(mem,64)
    z=toeplitz_apply(h,tx)
    X=np.column_stack([z[taps-1-k:len(z)-k] for k in range(taps)])
    d=obs[taps-1:]
    return np.linalg.lstsq(X,d,rcond=None)[0]

def state_receiver(tx_train,obs_train,obs_test,mem,h_est,ridge=1e-3):
    _,_,_,_,h=state_model(mem,512)
    combined=np.convolve(h,h_est)
    prefix=np.asarray(tx_train)
    known=np.zeros(len(obs_test),dtype=complex)
    for i,idx in enumerate(np.arange(len(prefix),len(prefix)+len(obs_test))):
        k=min(idx+1,len(combined),len(prefix))
        known[i]=np.dot(combined[:k],prefix[idx::-1][:k])
    innovation=obs_test-known
    H=conv_matrix(combined,len(obs_test))
    return ridge_solve(H,innovation,ridge)

def complexity(mem):
    m=len(SOE_MEMORIES[mem][0])
    return 2*m+1,m,2*m+1

def run_case(name,gen,mem,ch,seed,snr):
    sig=gen(NUM_SYMBOLS,samples_per_symbol=SPS,symbol_rate=SYMBOL_RATE,seed=seed)
    y,_=channel(memory_filter(sig.record.samples,mem),ch,np.random.default_rng(10000+seed))
    r=add_awgn(y,snr,np.random.default_rng(20000+seed*100+int(snr)))[sig.symbol_indices]
    tx=sig.symbols; tr=slice(0,TRAIN_SYMBOLS); te=slice(TRAIN_SYMBOLS,NUM_SYMBOLS)
    rows=[]
    for rec in RECEIVERS:
        if rec=="raw": z=r[te]
        elif rec=="fir7":
            b=design_fir_equalizer(r[tr],tx[tr],7,1e-5); z=apply_fir_equalizer(r,b)[te]
        elif rec=="iir2":
            b,a,pr=design_iir_equalizer(r[tr],tx[tr],2,2,1e-5,.98); z=apply_iir_equalizer(r,b,a)[te]
        else:
            # Training channel estimate uses known SOE structure but only the
            # training symbols. Held-out recovery then estimates the unknown
            # input sequence through the fixed state realization.
            h_est=estimate_channel(tx[tr],r[tr],mem,1 if ch in ("identity","flat_rayleigh") else 3)
            z=state_receiver(tx[tr],r[tr],r[te],mem,h_est)
            pr=np.nan
        p,s,mac=complexity(mem) if rec=="soe_state" else ((0,0,0) if rec=="raw" else ((7,6,7) if rec=="fir7" else (4,2,4)))
        rows.append(dict(experiment="12C_SOE_state_input_estimation",signal=name,memory=mem,channel=ch,receiver=rec,snr_db=snr,seed=seed,
                         evm_percent=100*evm(tx[te],z),ber=ber(name,tx[te],z),heldout_mse=float(np.mean(abs(z-tx[te])**2)),
                         parameter_count=p,state_dimension=s,macs_per_sample=mac,effective_memory_samples=s,
                         channel_estimate_norm=float(np.linalg.norm(h_est)) if rec=="soe_state" else np.nan))
    return rows

def main():
    rows=[]
    for seed in SEEDS:
      for n,g in SIGNALS.items():
       for mem in SOE_MEMORIES:
        for ch in CHANNELS:
         for snr in SNR_DB: rows+=run_case(n,g,mem,ch,seed,snr)
    out=ROOT/"experiments/results"; out.mkdir(parents=True,exist_ok=True)
    p=out/"12C_soe_state_input_estimation_results.csv"
    with p.open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary={"experiment":"12C_SOE_state_input_estimation","rows":len(rows),"evaluation":"384 held-out symbols after 128-symbol training prefix",
             "receiver":"fixed-pole SOE state realization with regularized batch state/input inversion; channel taps estimated from training",
             "boundary":"offline smoother, not a causal online receiver"}
    (out/"12C_soe_state_input_estimation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
