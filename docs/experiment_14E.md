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
