# Experiment 13C-3 — Environment–Kernel Model-Order and Rate-Support Mismatch

## Purpose

13C-3 tests whether the environment → kernel → temporal-state geometry mapping remains stable when the positive SOE dictionary is under-specified, over-dense, or shifted in rate support.

The experiment is task-agnostic: no BER, EVM, receiver, neural-network, or downstream task metric is introduced.

## Grid

Five controlled environments from 13C-1/13C-2 are reused, with four independent seeds and the same signal probes across dictionaries.

| Dictionary | Modes | Rate support (s^-1) | Role |
|---|---:|---:|---|
| under_low_8 | 8 | 0.5–10 | excludes faster support |
| under_high_8 | 8 | 5–100 | excludes slower support |
| matched_16 | 16 | 0.5–100 | reference |
| over_dense_32 | 32 | 0.25–200 | expanded support |
| shifted_16 | 16 | 0.75–150 | shifted support |

Positive coefficients are fitted with NNLS and normalized to unit sum.

## Comparisons

Because coefficient vectors from different dictionaries have different coordinates, they are not compared directly. Fitted kernels are compared in the common physical time domain.

Recorded GFE/GGFE metrics:
- M_cap
- M_scale
- M_res
- H_mem
- D_eff = L exp(H_mem)

Recorded geometry:
- weighted basis participation and entropy dimensions
- normalized-Gram participation and entropy dimensions
- maximum normalized basis coherence
- weighted MIN state participation and entropy dimensions
- rank-90% and rank-99%

Recorded errors:
- fitted-kernel relative L2 error
- fitted-kernel difference from the matched reference
- D_eff relative error
- weighted-basis participation relative error
- weighted-state participation relative error

## Dimension hierarchy

The experiment preserves the distinction:

L != D_eff != D_basis != D_state

D_eff is the GGFE entropy-effective count, not a realized temporal-state rank.

The normalized Gram control separates mode distinguishability from unequal finite-horizon mode energy, which is important for fast exponential modes.

## Interpretation boundary

13C-3 uses oracle environment covariance deliberately. This isolates dictionary/model mismatch from the finite-observation uncertainty studied in 13C-2.

Thus the tested chain is:

environment → kernel fit → geometry

rather than:

finite observation → covariance error → kernel error → geometry error.

Adaptive rate learning and joint model-order selection are deferred.

## Scientific role

13C-1 established an environment-dependent positive-kernel construction.

13C-2 established separate covariance, kernel-parameter, and representation-level errors under finite/noisy observation.

13C-3 tests robustness to the model class used to encode the environment. Stability of representation geometry despite parameter differences would support the distinction between identification fidelity and representation geometry; sharp geometry changes would identify a boundary for the kernel construction.

All results remain descriptive rather than causal.

## Reproducibility

Executable:
experiments/13C-3_environment_kernel_model_order_mismatch.py

Results are written to experiments/results/ and uploaded by the experiment-13c GitHub Actions workflow.
