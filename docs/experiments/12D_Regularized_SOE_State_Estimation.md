# Experiment 12D — Regularized SOE State Estimation and Inverse Conditioning

## Purpose

12D follows the diagnostic result of 12C. The question is no longer whether a
direct SOE inverse can be written down; 12C already demonstrated that the
matched noiseless sampled realization can encode the tested transformation.
12D asks why the inverse becomes unstable and whether regularized estimation
can exploit the compact SOE representation without requiring an unstable
direct inverse.

The primary object is the induced causal lower-triangular map

\[
y = H_K u.
\]

The central diagnostics are its singular spectrum,

\[
\sigma_1\ge\cdots\ge\sigma_n,
\]

minimum singular value, and condition number

\[
\kappa(H_K)=\frac{\sigma_1}{\sigma_n}.
\]

## Controlled cases

Five cases separate the sources of difficulty:

1. exact SOE model + exact channel + noiseless;
2. exact SOE model + exact channel + noise;
3. exact SOE model + estimated channel + noiseless;
4. exact SOE model + estimated channel + noise;
5. perturbed SOE model + estimated channel + noise.

The fifth case is a deliberately controlled receiver-side model mismatch. It is
not presented as a measurement of uncertainty in the physical environment.

The underlying signal/channel grid remains the 12C grid:

- BPSK, QPSK, 16-QAM;
- exponential and two-scale SOE memories;
- identity, flat-Rayleigh and normalized 3-tap multipath;
- 0, 10, 20 and 30 dB;
- five seeds;
- 128-symbol training prefix;
- 384 held-out symbols.

## Estimators

### Direct

The 12C baseline is retained as the unregularized triangular solve.

### Tikhonov / ridge

\[
\hat u_\lambda =
(H^\ast H+\lambda I)^{-1}H^\ast y.
\]

The experiment sweeps five dimensionless regularization factors,
scaled by \(\sigma_{\max}(H)^2\), rather than selecting one arbitrary
regularization value.

### Truncated SVD

\[
H=U\Sigma V^\ast,
\qquad
\hat u_\tau =
V\Sigma_\tau^+U^\ast y.
\]

Four relative singular-value cutoffs are tested. This exposes how much of
the inverse relies on small singular directions.

A Kalman/state-space estimator is intentionally not used as the primary
12D metric: the held-out symbols are deterministic QAM inputs rather than
a stochastic state process for which a particular process/measurement
covariance model is justified. 12D therefore uses the explicit SOE state
dimension and causal operator conditioning directly, avoiding an arbitrary
Kalman prior.

## Measurements

Every estimator records:

- \(\sigma_{\max}\);
- \(\sigma_{\min}\);
- condition number;
- numerical rank;
- a true-operator condition number for comparison;
- EVM;
- BER;
- held-out MSE;
- a noise-amplification proxy;
- SOE state dimension;
- parameter count;
- MACs/sample;
- estimated versus true channel norm;
- retained singular values for TSVD.

The condition-number measurements are primary scientific outputs. EVM/BER
are secondary task metrics showing the practical consequence of conditioning.

## Interpretation policy

12D must not be reduced to “regularization fixed MIN” or “regularization
failed to fix MIN.”

The intended separation is:

\[
\boxed{
\text{forward representation}
\neq
\text{inverse conditioning}
\neq
\text{task utility}
}
\]

A compact SOE realization may remain useful even if recovery of the original
waveform requires regularization or is intrinsically sensitive to noise.

Conversely, a numerically stable inverse would establish only a recovery
property for the tested sampled operator and receiver, not a general
communication or propagation advantage.

## Expected scientific output

12D should produce a conditioning map over:

\[
(\text{kernel},\text{channel},\text{SNR},\text{model mismatch})
\]

and show whether the singular directions responsible for direct inversion
are recoverable by regularized estimators.

This closes the 12C recovery branch sufficiently to move the main research
effort toward the refined MIN objective:

\[
\boxed{
\text{kernel}
\rightarrow
\text{temporal transform}
\rightarrow
\text{state representation}
\rightarrow
\text{propagation/manipulation}
\rightarrow
\text{task}
}
\]

rather than continuing to treat waveform recovery as the definition of a
useful MIN transform.
