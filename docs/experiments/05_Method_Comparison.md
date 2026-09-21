# Experiment 05 — Matched SOE-identification comparison

Date: 2026-09-21

## Reference problem

\[
k(t)=\left(1+\frac{t}{\tau}\right)^{-\alpha},
\qquad \alpha=0.7,\quad\tau=0.5.
\]

Sampling: \(\Delta t=0.02\), \(T=8\), 401 samples, with
\[
x(t)=\sin(2\pi0.4t)+0.35\sin(2\pi1.7t).
\]

Methods: classical Prony, Hankel matrix pencil, ESPRIT, scalar Laplace-domain vector fitting, and fixed positive-rate NNLS.

## Results

| Method | L | E_K | E_M | Stable | Real | Nonnegative | Positive-real SOE |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|
| Prony | 2 | 4.0466e-01 | 3.0038e-01 | yes | yes | no | no |
| Prony | 4 | 1.9123e-01 | 1.2955e-01 | yes | yes | no | no |
| Prony | 8 | 9.9993e-01 | 1.0000e+00 | no | no | yes | no |
| Prony | 16 | — | — | no | no | no | no; numerical failure |
| Matrix Pencil | 2 | 2.0702e-02 | 2.9400e-02 | yes | yes | yes | yes |
| Matrix Pencil | 4 | 2.4597e-04 | 2.7794e-04 | yes | yes | yes | yes |
| Matrix Pencil | 8 | 2.5453e-08 | 2.5001e-08 | yes | yes | yes | yes |
| Matrix Pencil | 16 | 7.8716e-13 | 1.2293e-12 | yes | yes | yes | yes |
| ESPRIT | 2 | 2.0684e-02 | 2.9340e-02 | yes | yes | yes | yes |
| ESPRIT | 4 | 2.4473e-04 | 2.7722e-04 | yes | yes | yes | yes |
| ESPRIT | 8 | 2.5207e-08 | 2.4692e-08 | yes | yes | yes | yes |
| ESPRIT | 16 | 6.1033e-14 | 8.9272e-14 | yes | yes | yes | yes |
| Vector Fitting | 2 | 6.4669e-02 | 8.5538e-02 | yes | yes | yes | yes |
| Vector Fitting | 4 | 3.8682e-03 | 5.8382e-03 | yes | yes | yes | yes |
| Vector Fitting | 8 | 2.2370e-05 | 3.5241e-05 | yes | yes | yes | yes |
| Vector Fitting | 16 | 8.3084e-08 | 1.3750e-07 | yes | yes | no | no |
| NNLS | 2 | 3.8156e-01 | 5.1070e-01 | yes | yes | yes | yes |
| NNLS | 4 | 2.5082e-02 | 3.5564e-02 | yes | yes | yes | yes |
| NNLS | 8 | 1.5700e-04 | 1.7482e-04 | yes | yes | yes | yes |
| NNLS | 16 | 6.4615e-06 | 1.0235e-05 | yes | yes | yes | yes |

## Interpretation

This is a clean-data identification comparison, not a universal ranking.

Classical Prony provides the unconstrained baseline and reaches a failure regime as order rises. The guard added to the reference implementation prevents exponentially explosive nodes from overflowing the Vandermonde solve.

Matrix Pencil and ESPRIT use a truncated Hankel signal subspace and produce extremely accurate positive-real exponential fits for this target in the clean experiment.

Vector fitting improves strongly with order, but its order-16 residues are no longer all nonnegative, so that model is not a positive-real SOE even though it remains stable.

NNLS explicitly preserves positive rates and nonnegative weights, with accuracy improving systematically as the rate dictionary grows.

The next experiment is controlled noise plus observation-horizon variation. That will test whether the clean-data subspace behavior survives perturbation and where constrained NNLS becomes more robust.
