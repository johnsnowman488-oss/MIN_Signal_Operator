# Experiment 12A — Receiver Recoverability Ladder

## Objective

Experiment 12A asks whether the residual error observed in Experiment 11 is
primarily a consequence of limited receiver memory/model structure.

The receiver ladder is:

\[
\boxed{\mathrm{raw}\rightarrow\mathrm{FIR\!-\!7}\rightarrow\mathrm{FIR\!-\!15}
\rightarrow\mathrm{FIR\!-\!31}\rightarrow\mathrm{IIR\!-\!2}
\rightarrow\mathrm{IIR\!-\!4}\rightarrow\mathrm{IIR\!-\!8}}
\]

The same signal, memory transformation, channel realization, AWGN realization,
training prefix, and held-out symbols are used for every receiver in a case.

## Why add IIR?

A longer FIR represents memory by explicitly retaining more past samples. An
IIR receiver represents temporal dependence recursively:

\[
\hat x[n]=\sum_{k=0}^{p-1}b_k y[n-k]
+\sum_{k=1}^{q}a_k\hat x[n-k].
\]

Therefore a small feedback state can generate a much longer effective impulse
response.

This tests whether the large FIR residuals from Experiment 11 are at least
partly a representation mismatch rather than evidence of non-invertibility.

The IIR implementation is a deliberately simple linear ARX baseline. Fitted
feedback recursions are stabilized to a maximum pole radius of 0.98, and the
resulting pole radius is recorded.

## Conditions

The Experiment 11 conditions are retained:

- BPSK, QPSK, 16-QAM;
- identity, exponential, two-scale SOE, and power-law memory;
- identity, flat Rayleigh, and normalized 3-tap multipath;
- 0, 10, 20, 30 dB SNR;
- five deterministic seeds;
- 128-symbol training prefix;
- 384 held-out symbols.

This gives:

\[
3\times4\times3\times4\times5\times7=5040
\]

receiver-condition rows.

## Complexity accounting

| Receiver | Parameters | State dimension | MAC/sample | Explicit memory |
|---|---:|---:|---:|---:|
| raw | 0 | 0 | 0 | 0 |
| FIR-7 | 7 | 6 | 7 | 6 |
| FIR-15 | 15 | 14 | 15 | 14 |
| FIR-31 | 31 | 30 | 31 | 30 |
| IIR-2 | 4 | 2 | 4 | 2 |
| IIR-4 | 8 | 4 | 8 | 4 |
| IIR-8 | 16 | 8 | 16 | 8 |

The MAC count is an accounting convention for the Python implementation, not
a hardware benchmark.

## Metrics

Each held-out case records:

- BER;
- RMS EVM;
- held-out MSE;
- fitted parameter count;
- state dimension;
- MACs/sample;
- explicit receiver memory;
- IIR pole radius.

Primary plots:

\[
\boxed{\mathrm{BER/EVM}\ \mathrm{vs.}\ \mathrm{state\ dimension}}
\]

and

\[
\boxed{\mathrm{BER/EVM}\ \mathrm{vs.}\ \mathrm{MACs/sample}}.
\]

A useful secondary analysis is the smallest receiver complexity that reaches
predefined EVM or BER targets, when a target is reached.

## Required controls

1. **identity channel + AWGN** — establishes the noise/reference floor;
2. **identity channel + multipath + AWGN** — validates conventional equalization;
3. **MIN memory + identity channel + AWGN** — isolates the discrete memory transformation;
4. **MIN memory + multipath + AWGN** — tests the combined problem.

The explicit identity channel is included so the memory-only control is genuinely
separable from fading. Flat Rayleigh remains a controlled fading condition, while
3-tap multipath exposes ordinary ISI.

## Interpretation

### IIR improves long-memory recovery at low state dimension

This supports only the narrow conclusion that the tested FIR family was not a
sufficient representation of the memory.

It does not establish a communication advantage for MIN.

### IIR is similar to FIR at comparable arithmetic

This indicates that recursive representation does not automatically provide a
measurable advantage under the tested conditions.

### IIR requires strong stabilization

That is a meaningful conditioning/stability result: recursive inversion may
trade explicit memory for feedback sensitivity.

### All linear receivers remain poor

Check, in order:

1. discrete memory-filter invertibility;
2. finite-history/truncation effects;
3. training-prefix sufficiency;
4. channel/noise identifiability;
5. numerical conditioning;
6. a receiver explicitly constructed from the known memory kernel or SOE.

No conclusion about continuous-time MIN should be drawn from this experiment
alone.

## Scientific boundary

Experiment 12A does not test propagation superiority.

It asks:

\[
\boxed{\text{What receiver complexity is required to undo the tested discrete MIN-induced memory?}}
\]

The forward-propagation stability question remains Experiment 13.
