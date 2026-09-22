# Experiment 12C — SOE state/input inversion

## Question

Does explicit realization of the SOE memory state provide a useful representation of a MIN-transformed signal, and what happens when that representation is forced through a direct noisy inverse?

12C is a deliberately diagnostic experiment. It does **not** assume that the purpose of a MIN transform is to preserve the original waveform, nor that a useful transform must be easy to undo with a conventional equalizer.

The broader research program began with three related questions:

1. **Transformation:** what temporal structure does MIN create in an information-bearing signal?
2. **Propagation:** can that transformed representation behave usefully or stably under a specified propagation process?
3. **Manipulation/recovery:** when the original signal is required, how efficiently can the transformation be inverted or otherwise decoded?

12C addresses only the third question, and only for a particular sampled SOE realization. Its negative recovery result must therefore not be interpreted as evidence that MIN is a poor transform.

## What changes from 12B

12B knew the SOE structure and constrained a transfer-function receiver.

12C instead constructs an explicit fixed-pole SOE state realization,

\[
x_{n+1}=Ax_n+Bu_n,\qquad y_n=Cx_n+Du_n,
\]

and uses that realization to form the causal lower-triangular input/output map.

The executed implementation then performs an **unregularized triangular solve** on that map. The file-level description should therefore not call this a "regularized batch state/input estimator": regularization is a 12D objective, not part of the 12C baseline.

This distinction matters:

\[
\text{SOE state representation}
\neq
\text{state estimation}
\neq
\text{regularized state estimation}.
\]

12C is primarily a **matched state-realization plus direct-inversion control**.

## Experimental design

- BPSK, QPSK, 16-QAM
- exponential and two-scale SOE memories
- identity, flat Rayleigh, normalized 3-tap multipath
- 0, 10, 20, 30 dB SNR
- five seeds
- 128-symbol training prefix
- 384 held-out symbols
- receivers: raw, FIR-7, IIR-2, SOE-state

Total:

\[
3\times2\times3\times4\times5\times4=\boxed{1440}.
\]

Unlike 12B, the numerical memory realization uses a 256-symbol kernel horizon. This reduces contamination from the arbitrary 32-symbol convolution cutoff used in the earlier communication experiments.

## State realization

For SOE poles

\[
r_j=e^{-\gamma_jT_s},
\]

the symbol-rate realization has

\[
x_{n+1}=Ax_n+Bu_n,\qquad
y_n=Cx_n+Du_n.
\]

The state parameters are fixed by the known SOE structure and the sampled rectangular-pulse impulse response.

The held-out input is then recovered through the corresponding causal lower-triangular input/output map.

The implementation therefore tests whether a compact state realization can represent the transformation and whether its direct inverse remains numerically usable after channel estimation and noise.

## Matched noiseless sanity check

In the sandbox, when the receiver used the same structured SOE realization that generated the observations, the held-out BPSK sequence was recovered to numerical precision (EVM on the order of \(10^{-15}\)).

This establishes the narrow control:

\[
\boxed{
\text{The tested sampled SOE realization can encode the transformation accurately.}
}
\]

It does **not** establish that:

- the continuous MIN operator is invertible;
- the inverse is well-conditioned;
- a conventional receiver is the intended use of MIN;
- MIN preserves the original waveform;
- MIN improves communication or propagation.

The matched-noiseless result is therefore a representation sanity check, not a utility claim.

## CI execution

The corrected experiment completed successfully in GitHub Actions with 1,440 rows.

At 30 dB, averaged over signals, channels, and five seeds:

| Memory | Receiver | Mean EVM | Mean BER |
|---|---|---:|---:|
| Exponential | raw | 98.32% | 0.4433 |
| Exponential | FIR-7 | 28.53% | 0.0237 |
| Exponential | IIR-2 | 28.82% | 0.0223 |
| Exponential | **SOE-state** | **74.80%** | **0.1378** |
| Two-scale SOE | raw | 99.44% | 0.4663 |
| Two-scale SOE | FIR-7 | 39.47% | 0.0618 |
| Two-scale SOE | IIR-2 | 42.48% | 0.0633 |
| Two-scale SOE | **SOE-state** | **99.41%** | **0.1965** |

The state receiver also produced overflow warnings in low-SNR cases. The workflow itself completed successfully; the warnings are numerical evidence of instability in the tested direct-inversion path rather than CI failures.

## Interpretation

### 1. Compact representation

For the tested exponential and two-scale kernels, an explicit SOE realization gives a low-dimensional representation of the sampled temporal transformation.

That is already different from saying that MIN is "just a bad filter." A transformation can deliberately redistribute information across temporal coordinates while still being represented by a compact dynamical state.

### 2. Direct inversion

The direct causal inverse is not robust under the tested noisy/channel-estimated conditions.

At 30 dB it performs substantially worse than FIR-7 and IIR-2, and lower-SNR cases can become numerically explosive.

The correct conclusion is therefore:

\[
\boxed{
\text{compact SOE representation}
\not\Rightarrow
\text{well-conditioned direct inverse}.
}
\]

This is a conditioning result, not a verdict on MIN as a transformation.

### 3. Representation versus objective

The receiver experiments have so far optimized one very specific objective:

\[
\hat{x}\approx x.
\]

That is appropriate for asking whether a conventional communications receiver can undo the transformation, but it is **not the only objective available to a time-series transform**.

A more general transform-domain objective is:

\[
x
\xrightarrow{\;M_K\;}
z
\xrightarrow{\;\text{propagation/manipulation}\;}
z'
\xrightarrow{\;D_K\;}
\hat{x},
\]

where the useful quantity may be the stability, separability, predictability, controllability, bandwidth occupancy, coding behavior, or robustness of \(z\), rather than the raw closeness of \(z\) to \(x\).

For this reason, a non-identity MIN output should be regarded first as a **transformed representation**. Calling it "distortion" is correct only relative to an objective that requires preservation of the original waveform.

### 4. A sharper research boundary

12C does not answer:

> "Is MIN better than conventional filtering?"

It answers:

> **Can an explicitly known SOE memory state represent the tested MIN transformation, and what numerical penalty appears when we insist on recovering the original input by direct inversion?**

That question is useful because it separates **transform representation** from **inverse recovery**.

## Kernel-selection implication

The two kernels tested so far are only a small part of the space of possible transformations.

An SOE form

\[
K(t)=\sum_{j=1}^{m}a_j e^{-\gamma_j t}
\]

is not merely a numerical approximation device. Its weights, rates, normalization, sign constraints, pole geometry, and number of modes determine what temporal structure the transform creates.

Consequently, the result

\[
\text{two tested SOE families}\rightarrow\text{poor direct recovery}
\]

must not be generalized to

\[
\text{all useful MIN kernels}\rightarrow\text{poor transform behavior}.
\]

The research should distinguish two searches:

\[
\boxed{
\text{kernel selection}
\quad\text{and}\quad
\text{receiver design}
}
\]

rather than choosing a small set of kernels and treating the resulting behavior as representative of the whole operator family.

A future kernel study should vary at least:

- decay-rate distribution \(\{\gamma_j\}\);
- mixture weights \(\{a_j\}\);
- number of SOE modes;
- positivity versus signed/oscillatory coefficients where mathematically admissible for the intended use;
- normalization and DC gain;
- temporal support/effective memory;
- spectral concentration and notches;
- conditioning of the forward map and inverse map;
- robustness of the associated state realization.

The important question is not simply "which kernel produces less EVM?" but:

\[
\boxed{
\text{Which kernel + realization produces a useful and controllable temporal representation for the target task?}
}
\]

## Continuous-to-discrete boundary

All 12C conclusions concern the chain

\[
\text{continuous kernel}
\rightarrow
\text{sampled rectangular-pulse realization}
\rightarrow
\text{communication channel}
\rightarrow
\text{training-based receiver}
\rightarrow
\text{direct inverse}.
\]

They do not establish properties of the continuous MIN operator independently of sampling, truncation, pulse shape, channel model, training length, or receiver parameterization.

## Next experiment: 12D

12D should be framed as a **conditioning and robust-estimation experiment**, not merely as an attempt to make 12C "work."

The key decomposition is:

1. perfect SOE model + perfect channel + noiseless observation;
2. perfect SOE model + perfect channel + noise;
3. perfect SOE model + estimated channel + noiseless observation;
4. estimated channel + noise.

For each case, measure:

- singular values of the induced causal operator;
- condition number and minimum singular value;
- noise amplification;
- direct inverse;
- Tikhonov/ridge;
- truncated-SVD;
- state-space/Kalman-style estimation where appropriate;
- BER, EVM and held-out MSE;
- state dimension and MACs/sample.

The central question becomes:

\[
\boxed{
\text{Can the compact SOE state be exploited without paying an unstable inverse?}
}
\]

A positive result would support an efficient **representation/receiver mechanism** for the tested task. A negative result would quantify a representation-versus-conditioning trade-off rather than invalidate the broader transform hypothesis.

## Longer-term research direction

The receiver ladder should remain only one branch of the larger MIN program:

\[
\boxed{
\text{kernel}
\rightarrow
\text{transform}
\rightarrow
\text{state representation}
\rightarrow
\text{propagation}
\rightarrow
\text{manipulation}
\rightarrow
\text{decoding/recovery}
}
\]

The eventual research question is therefore broader than whether MIN can be inverted after it has transformed a waveform:

\[
\boxed{
\text{Can deliberately engineered memory create a useful temporal representation or propagation state?}
}
\]

That question should be tested directly rather than inferred from conventional equalization alone.
