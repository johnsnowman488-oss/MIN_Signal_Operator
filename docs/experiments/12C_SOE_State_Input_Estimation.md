# Experiment 12C — SOE state/input estimation

## Question

Does explicit realization of the SOE memory state provide a better receiver mechanism than treating the memory as an unconstrained transfer function?

12C moves from the 12B fixed-denominator receiver to an explicit fixed-pole state realization. The receiver estimates the channel taps from the training prefix, carries the known training state into the held-out interval, and performs causal triangular state/input inversion.

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

[
3	imes2	imes3	imes4	imes5	imes4=oxed{1440}
]

Unlike 12B, the numerical memory realization uses a 256-symbol kernel horizon. This reduces contamination from the arbitrary 32-symbol convolution cutoff used in the earlier communication experiments.

## State realization

For SOE poles

[
r_j=e^{-gamma_jT_s},
]

the symbol-rate state realization has

[
x_{n+1}=Ax_n+Bu_n,qquad
y_n=Cx_n+Du_n.
]

The state parameters are fixed by the known SOE structure and the sampled rectangular-pulse impulse response. The held-out input is then recovered through the resulting causal lower-triangular input/output map.

This is intentionally a **state/input inversion test**, not another free-form IIR fit.

## Matched noiseless sanity check

In the sandbox, when the receiver used the same structured SOE realization that generated the observations, the held-out BPSK sequence was recovered to numerical precision (EVM on the order of (10^{-15})).

This is an important control:

[
oxed{	ext{The SOE state representation can encode the tested memory transformation.}}
]

It does **not** imply that the inverse is numerically useful under noise.

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

The state receiver also produced overflow warnings in low-SNR cases. The workflow itself completed successfully; the warnings are numerical evidence of instability rather than CI failures.

## Interpretation

The experiment separates two questions that had previously been easy to conflate.

### 1. Is the memory represented by a small state?

Yes, for the tested SOE family.

The matched noiseless realization can reproduce and invert the transformation accurately.

### 2. Does explicit state knowledge make inversion robust?

Not in this first causal implementation.

At 30 dB, the direct state/input inversion was substantially worse than FIR-7 and IIR-2. At lower SNR, some cases became numerically explosive.

Therefore:

[
oxed{
	ext{SOE has a compact state realization}

otRightarrow
	ext{the inverse is numerically well-conditioned}.
}
]

This is a useful result for the research program. It points toward **conditioning and regularization** as the next receiver problem rather than simply adding more state structure.

## Boundary

12C does not establish continuous-time MIN invertibility. It establishes behavior of a particular sampled SOE realization, communication channel, training procedure, and causal inversion algorithm.

The longer 256-symbol horizon also means its numerical memory realization is not directly identical to the earlier 32-symbol finite-convolution experiments; this was intentional to isolate state-representation behavior from an arbitrary short truncation.

## Next experiment

The natural next step is **12D: regularized/robust SOE state estimation**.

Candidate controls should include:

- ridge/Tikhonov input estimation,
- truncated-SVD or singular-value filtering,
- state-space/Kalman-style estimation,
- known-channel versus estimated-channel controls,
- condition-number/singular-spectrum diagnostics.

The key question becomes:

[
oxed{
	ext{Can the compact SOE state be exploited without paying an unstable inverse?}
}
]

That should be tested before making any claim of computational advantage.
