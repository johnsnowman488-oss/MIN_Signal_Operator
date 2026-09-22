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


## Executed run — 2026-09-21

The reproducible GitHub Actions run completed successfully on commit
`1f87d0d7c650b0b4cea6ba2dacdf587e5839dc72`. It produced 5,040 receiver-condition
rows and the aggregate analysis tables.

The following table averages over BPSK, QPSK, 16-QAM, all three channel
conditions, and five seeds at 30 dB SNR.

| Memory | Receiver | Mean EVM (%) | Mean BER |
|---|---|---:|---:|
| identity | IIR-2 | 3.288 | 0.000 |
| identity | FIR-7 | 3.346 | 0.000 |
| exponential | FIR-7 | 28.791 | 0.024 |
| exponential | IIR-2 | 29.078 | 0.024 |
| two-scale SOE | IIR-2 | 48.509 | 0.095 |
| two-scale SOE | FIR-7 | 48.610 | 0.099 |
| power-law | FIR-7 | 66.765 | 0.209 |
| power-law | IIR-2 | 68.283 | 0.199 |
| raw, identity memory | raw | 49.306 | 0.138 |
| raw, exponential memory | raw | 98.345 | 0.443 |
| raw, two-scale SOE | raw | 99.395 | 0.463 |
| raw, power-law | raw | 99.865 | 0.479 |

### Result interpretation

The central result is a separation between **ordinary channel equalization** and
**recovery of the tested memory transformations**.

For the identity-memory condition, a two-state IIR receiver already reaches about
3.3% EVM at 30 dB, with zero mean BER in this aggregate. Conventional
multipath/fading therefore remains recoverable with very small receiver state in
this experiment.

Once the memory transformation is introduced, increasing generic receiver
complexity does not restore the same error floor. At 30 dB, the exponential
condition reaches about 29% EVM, the two-scale SOE condition about 49% EVM, and
the power-law condition about 67% EVM. Moving from IIR-2 to IIR-8 or FIR-7 to
FIR-31 does not produce monotonic improvement; in several cases the larger
models are slightly worse.

The important negative result is therefore:

[
oxed{	ext{More generic linear receiver complexity did not, by itself,
remove the long-memory residual.}}
]

This is stronger evidence than Experiment 11 alone that the residual is not
simply a consequence of using a very short FIR. However, it is **not** evidence
of non-invertibility.

### What the run says about IIR

The IIR ladder demonstrates that recursive representation can compress temporal
dependence in the receiver, but it does not show a systematic advantage over
FIR at matched low complexity.

At 30 dB, IIR-2 and FIR-7 are nearly indistinguishable for the exponential and
two-scale memories. For the power-law memory, FIR-7 has lower EVM while IIR-2
has slightly lower BER. Thus there is no clean generic-IIR win.

This makes the next experiment more informative: the receiver should use the
**known SOE/MIN structure itself**, rather than merely replacing taps with
feedback coefficients.

### Important implementation observation

The memory stage in this experiment is a finite sampled convolution. The results
therefore establish properties of the tested discrete realization:

[
	ext{continuous memory model}
ightarrow
	ext{finite sampled convolution}
ightarrow
	ext{communication channel}
ightarrow
	ext{receiver}.
]

They do not establish a property of the continuous MIN operator independent of
sampling, truncation, or the particular discrete kernel.

### Next scientific test

Experiment 12B should construct a receiver from the known SOE parameters and
compare it against generic FIR/IIR receivers under matched state/arithmetic
budgets. The key question is:

[
oxed{	ext{Does knowledge of the MIN/SOE structure reduce recovery
complexity, conditioning, or error?}}
]

Only if that controlled comparison shows a structural effect should we move to
Experiment 13 on forward propagation stability.
