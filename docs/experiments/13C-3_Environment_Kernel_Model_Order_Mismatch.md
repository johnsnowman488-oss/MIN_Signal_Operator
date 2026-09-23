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

## Executed result

GitHub Actions run: 35905567846.

- 41 tests passed.
- 13C-1 regression passed.
- 13C-2 regression passed.
- 13C-3 completed successfully.
- 300 result rows were generated.
- Artifact SHA-256: 37aac2653bfcfe67cd46f7c93412ddd3103bc99108ab6cc08fa15d8a1539a04f

### Main observations

Across the five environment families, the mean fitted-kernel relative L2 error was approximately:

| Dictionary | Mean kernel-fit relative L2 |
|---|---:|
| matched_16 | 0.1248 |
| over_dense_32 | 0.0096 |
| shifted_16 | 0.0534 |
| under_high_8 | 0.1533 |
| under_low_8 | 0.8496 |

The aggregate is strongly affected by the white-limit case and therefore should not be interpreted as a universal dictionary score.

For the multiscale environment, under_low_8 produced a weighted-state participation error of about 18.3%, while over_dense_32 was about 0.6% and shifted_16 about 0.4%. For the power-law environment, under_high_8 produced about 19.5% weighted-state participation error, compared with about 8.8% for over_dense_32 and 6.0% for shifted_16.

A useful control result is that normalized-Gram participation is dictionary-dependent but environment-independent in this protocol: it is determined by the fixed rate grid, whereas weighted basis/state geometry additionally reflects fitted kernel weights.

Across all 300 rows, weighted-basis relative error and weighted-state relative error were strongly correlated (Pearson approximately 0.958). This is expected in part because the same fitted weights are used to construct both quantities, so it is descriptive rather than an independent causal validation.

The correlation between kernel-fit error and weighted-state error was weak (Pearson approximately -0.08), reinforcing the earlier distinction between kernel identification fidelity and representation geometry.

### Interpretation

13C-3 supports a narrower statement than “more SOE modes are better”:

**Rate-support coverage is a critical part of the environment-to-kernel representation.**

An expanded dictionary can preserve representation geometry even when it contains more modes than the reference. Conversely, an under-specified dictionary can produce large kernel-fit error and, for environments whose relevant scales are excluded, materially alter the realized weighted temporal state geometry.

The results do not establish that mode count itself is unimportant. They show that nominal L must be interpreted together with rate support and fitted weight distribution.

The result also strengthens the dimension hierarchy:

L != D_eff != D_basis != D_state

In particular, changing L changes the scale of D_eff = L exp(H_mem) mechanically, so D_eff should not be used alone to infer realized MIN state dimension.

### Limitation

The environment covariance is oracle in 13C-3. Thus this experiment isolates dictionary/model mismatch rather than finite-sample uncertainty. Adaptive rate learning and joint model-order selection remain the next methodological extension if needed.
