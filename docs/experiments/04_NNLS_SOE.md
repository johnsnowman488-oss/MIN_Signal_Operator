# Experiment 04 — Positive NNLS finite-SOE approximation

Date: 2026-09-21

## Objective

Test a structure-aware finite sum of exponentials (SOE) against the completely monotone power-law memory kernel

\[
k(t)=\left(1+\frac{t}{\tau}\right)^{-\alpha}
\]

using fixed positive decay rates and non-negative least-squares (NNLS) weights.

The decay rates are fixed on a logarithmic dictionary with \(\gamma_j\in[0.05,20]\), and the fit solves

\[
\min_{w_j\ge0}\|Aw-k\|_2^2,
\qquad A_{ij}=e^{-\gamma_jt_i}.
\]

No joint optimization of weights and decay rates is used.

## Reference problem

- \(\alpha=0.7\)
- \(\tau=0.5\)
- \(\Delta t=0.02\)
- horizon \(T=8\)
- 401 samples
- test signal \(x(t)=\sin(2\pi0.4t)+0.35\sin(2\pi1.7t)\)

Errors:

\[
E_K=\frac{\|k-k_L\|_2}{\|k\|_2},
\qquad
E_M=\frac{\|M_kx-M_{k_L}x\|_2}{\|M_kx\|_2}.
\]

## Results

| Candidate rates L | Active weights | Kernel relative error E_K | Operator relative error E_M | NNLS solve time* |
|---:|---:|---:|---:|---:|
| 2  | 2  | 3.8156e-01 | 5.1070e-01 | 0.12 ms |
| 4  | 3  | 2.5082e-02 | 3.5564e-02 | 0.04 ms |
| 8  | 7  | 1.5700e-04 | 1.7482e-04 | 0.06 ms |
| 16 | 11 | 6.4615e-06 | 1.0235e-05 | 0.09 ms |

*Solver time is machine-dependent and is only a computational reference for this run.*

All fitted weights were non-negative and every decay rate was strictly positive.

## Rate-dictionary sensitivity

At nominal L=4:

| Positive rate range | Kernel relative error | Operator relative error |
|---|---:|---:|
| [0.02, 20] | 2.6328e-02 | 3.3964e-02 |
| [0.05, 20] | 2.5082e-02 | 3.5564e-02 |
| [0.10, 50] | 2.4321e-02 | 3.6959e-02 |

The choice of positive rate dictionary therefore remains an independent experimental variable.

## Interpretation

For this reference long-memory kernel, fixed-rate positive NNLS gives a much cleaner finite-SOE representation as the candidate dictionary grows, with the most pronounced improvement appearing from L=4 onward.

This does not establish universal superiority over Prony or optimality of the chosen rate range. It establishes a controlled baseline in which the known positive-measure structure of a completely monotone kernel is enforced explicitly.

The previous Prony stress test remains the unconstrained baseline. The next comparison should test whether the behavior persists under Hankel-SVD / matrix-pencil / ESPRIT and vector fitting before adaptive rate refinement or hierarchical MIN.