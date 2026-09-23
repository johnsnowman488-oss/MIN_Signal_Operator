# Experiment 13C-2 — Finite Observation and Kernel Uncertainty

## Purpose

13C-2 removes the oracle-environment assumption of 13C-1.

The tested chain is

\[
\text{finite noisy environment realization}
\rightarrow \widehat C_{env}
\rightarrow \widehat K_{env}
\rightarrow \widehat{\{w_i\}}
\rightarrow \text{weighted basis geometry}
\rightarrow \text{MIN state geometry}.
\]

The oracle 13C-1 kernel remains available for each environment, allowing separate measurement of covariance, kernel, GFE/GGFE, basis, and state errors.

## Grid

- Environments: white-limit, short-memory, multiscale, power-law, squared-exponential.
- Independent environment seeds: 4.
- Observation lengths: 512, 2048, 8192 samples.
- Observation SNR: 0, 10, 20, 30 dB.
- Signal families used for the downstream state measurement: BPSK, QPSK, 16-QAM.
- Fixed positive SOE dictionary: 16 logarithmically spaced rates from 0.5 to 100 s^-1.
- Total rows: 720.
- Independent environment realizations: 240; signal families share each environment estimate and therefore are not counted as independent environmental evidence.

## Estimation

The environment realization is synthesized as a stationary Gaussian process with the prescribed 13C-1 covariance envelope. Measurement noise is then added at the selected SNR.

The covariance estimator is the unbiased sample autocovariance, normalized by its observed zero-lag value. All observation lengths are evaluated on the same 0.08 s physical lag window, so changing N does not silently change the covariance-error domain. A nonnegative least-squares fit then identifies the kernel weights on the fixed SOE dictionary.

This deliberately tests **environment estimation and kernel uncertainty**, not model-order mismatch.

## Error cascade

The experiment records:

1. covariance relative L2 error;
2. covariance integral error;
3. kernel fit error to the estimated covariance;
4. kernel-weight error against the oracle kernel;
5. GFE/GGFE descriptor changes, including D_eff;
6. weighted SOE basis geometry error;
7. weighted MIN state geometry error.

The important distinction is that kernel-fit quality and representation-geometry fidelity are separate measurements.

## Geometry controls

The weighted finite-horizon Gram matrix is

\[
G_w=W^{1/2}GW^{1/2},
\qquad
G_{ij}=\frac{1-e^{-(\gamma_i+\gamma_j)T}}{\gamma_i+\gamma_j}.
\]

The normalized Gram/correlation form is also recorded through weighted basis coherence. This helps distinguish mode distinguishability from unequal exponential-mode norms.

## Boundaries

13C-2 retains the fixed dictionary inherited from 13C-1. It does not introduce arbitrary kernel families, model-order selection, channel/receiver effects, BER/EVM, neural networks, or task optimization.

Dictionary/rate-support mismatch is reserved for 13C-3.

## Interpretation target

The central question is not whether finite noisy observations reproduce the oracle exactly. It is whether the mapping

\[
C_{env}\rightarrow K_{env}\rightarrow \text{MIN representation}
\]

remains quantitatively stable as observation length and observation SNR vary.

The four-seed environmental repetitions are the independent evidence for this question; the three signal families are representation probes sharing the same estimated environment.


## Execution result

Corrected CI execution:

- Workflow run: `35903753087`
- 41-test repository suite: passed.
- 13C-1 regression: passed.
- 13C-2 rows: 720.
- Independent environment realizations: 240.
- Artifact SHA-256: `995166dc7ef2dc8144adf3ae8c5158e6210d3693a92c5de5f503d68c23ee5e78`.

The covariance-error domain was corrected to a common physical lag horizon of 0.08 s before accepting the run. This avoids making longer observations appear worse simply because they are evaluated over a longer and noisier covariance tail.

Across the complete grid, mean covariance relative-L2 error decreased from approximately 0.623 at (N=512), to 0.350 at (N=2048), and 0.239 at (N=8192). By SNR, the corresponding means were approximately 0.639 at 0 dB, 0.360 at 10 dB, 0.311 at 20 dB, and 0.306 at 30 dB.

Mean kernel-weight relative-L2 error remained around 1.12, 1.04, and 0.86 for (N=512,2048,8192), respectively. Thus covariance recovery improves substantially without implying equally accurate recovery of the individual SOE weights.

The representation-level errors were smaller than the kernel-weight errors. Across observation lengths, mean weighted-basis participation error was approximately 0.086, 0.115, and 0.034, while weighted-state participation error was approximately 0.140, 0.150, and 0.058.

Across all 720 rows, covariance error correlated only weakly with weighted-state error (Pearson approximately 0.19), and kernel-weight error was essentially uncorrelated with weighted-state error (approximately -0.06). Weighted-basis error and weighted-state error were strongly coupled (approximately 0.87).

These results are descriptive, not causal. They support a specific distinction: **environment estimation error, kernel-parameter error, and realized temporal-representation error are not interchangeable quantities**. In particular, a kernel can have substantial coefficient error while producing comparatively stable weighted representation geometry.

The result also reinforces the GGFE distinction between (D_{eff}) and realized state/basis dimension. For example, the squared-exponential environment showed a large mean (D_{eff}) relative error (about 0.82) while its weighted-state participation error remained about 0.013. Therefore (D_{eff}) should not be interpreted as a direct prediction of realized MIN state dimension.
