# Experiment 14E — Causal One-Step-Ahead Prediction

## Aim

14E asks whether the environment-informed MIN/SOE state retains information
that is useful for predicting a future observation, rather than only
reconstructing the current symbol.

The controlled chain is:

    past noisy observation → MIN/SOE memory state → future observation

The target is the noisy waveform at the next symbol center.

## Why this target

The BPSK, QPSK, and 16-QAM generators used in 14A–14D generate independent
symbols. Therefore, predicting the next symbol would not provide a legitimate
test of memory: the future symbol is independent of the past.

The next observed waveform sample is different. It contains a future symbol
component that is intentionally unpredictable plus colored environmental noise
whose temporal dependence can be carried across the one-symbol interval.
Consequently, the task asks whether the representation retains predictable
temporal information without pretending that the future symbol itself is
predictable.

## Protocol

The same 300 underlying cases as 14C/14D are used:

- BPSK, QPSK, 16-QAM;
- five controlled colored-noise environments;
- SNR = 0, 10, 20, 30 dB;
- five seeds;
- clustered, logspread, and wide 16-mode geometries.

For each case and geometry, the state at symbol center n predicts the observed
sample at center n+1. The state is computed causally from samples through n.
The final center has no target and is excluded.

Full-state readout uses PCA dimensions:

    d = 1, 2, 4, 8, 12, 16

and ridge strengths:

    λ = 0, 10^-8, 10^-6.

A secondary scalar-MIN path is evaluated at the same three ridge strengths.

The baseline is the current noisy center sample predicting the next noisy center
sample using λ=10^-6.

## Causality

For every prediction pair:

    state(y_0, ..., y_n) → y_{n+1}

No sample after the predictor center is used in state construction or PCA/readout
fitting. The held-out future sample is used only as an evaluation target.

## Measurements

Primary metric:

    NMSE_pred = E|ŷ_{n+1} - y_{n+1}|² / E|y_{n+1}|²

The main comparison is improvement relative to the current-observation baseline.

A predictive task dimension is also recorded:

    D_task,pred = min{d : NMSE_pred(d, λ=10^-6)
                         ≤ 1.10 NMSE_pred(16, λ=10^-6)}

This is operational and task-specific.

## Data accounting

There are:

- 300 underlying task cases;
- 3 geometries;
- 6 retained dimensions;
- 3 ridge strengths;

giving 16,200 full-state rows, plus 2,700 scalar-MIN rows.

## Interpretation boundary

14E is a temporal-information diagnostic, not a universal MIN performance
benchmark. Prediction error contains an irreducible component because the
future symbol is independent.

Evidence that MIN improves prediction over the current-observation baseline
would indicate that the state makes predictable temporal information more
accessible. Failure to improve the baseline would constrain that interpretation;
it would not by itself disprove the existence of useful memory for other tasks.

14E deliberately keeps the decoder linear and the task causal so that the
experiment remains comparable to 14C/14D.

## Relation to the program

13A: controlled kernel geometries.

13B: geometry → state structure.

13C: environment → kernel construction and mismatch.

14A: task-relevant information in the environment-informed full state.

14B: kernel alignment and scalar MIN exposure.

14C: geometry → task-relevant state structure.

14D: decoder accessibility and conditioning.

14E: causal future-observation prediction from the memory representation.


## Executed result

The authoritative final CI execution completed successfully:

- Experiment 14E workflow: 36306039350
- final experiment head: fcb58932e7081f0d1f6ccc54ea0ac5ad697bba9e
- result artifact: experiment-14e-results (artifact ID 10926978955)
- artifact SHA-256: 1cb0bfc6830cb640b265668e468ef30f6145bf492c310c5cdf229225c96eb267
- rows: 18,900 (16,200 full-state and 2,700 scalar-MIN)
- targeted/regression tests: 9 passed before the numerical sweep

The current-observation baseline has mean next-observation NMSE 0.9605 across
the 300 task cases.

At d=16 and lambda=10^-6, the full-state predictive NMSE was:

| Geometry | Mean | Median | Std. |
|---|---:|---:|---:|
| clustered | 0.9505 | 1.0028 | 0.1267 |
| logspread | 0.9885 | 1.0194 | 0.1655 |
| wide | 1.2845 | 1.1239 | 0.4854 |

Relative to the same raw baseline, the mean paired change at d=16,
lambda=10^-6 was approximately -0.0100 for clustered, +0.0281 for
logspread, and +0.3240 for wide. Clustered therefore gives only a modest
aggregate improvement, while the other two geometries do not improve the
simple current-observation predictor.

The predictive task dimension under the 10% criterion had means of
approximately 1.05, 1.30, and 1.16 for clustered, logspread, and wide,
respectively. These values are operational task dimensions, not physical
state dimensions.

The scalar-MIN path gave mean NMSE approximately 0.9559, 0.9674, and 0.9925
for clustered, logspread, and wide respectively.

These results constrain the interpretation of the MIN state. In this
controlled IID-symbol setting, the state does not provide a broad predictive
advantage over the current noisy observation. The small clustered improvement
is evidence that the memory representation can expose a limited amount of
predictable temporal structure, but the geometry dependence and the absence of
a consistent gain mean that 14E does not establish a general predictive
superiority of MIN.

A stronger temporal-memory test should therefore move to 14F with a
controlled process whose future is genuinely dependent on past state, rather
than relying on IID communication symbols.
