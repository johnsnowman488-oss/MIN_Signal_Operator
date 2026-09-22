# Experiment 12B — SOE/MIN-Structured Receiver

## Objective

Experiment 12B asks whether explicit knowledge of the tested SOE memory structure reduces receiver complexity or recovery error relative to generic FIR/IIR receivers.

The receiver is not a generic IIR with arbitrary feedback coefficients. For each known SOE memory, the symbol-rate recursive denominator is derived from the sampled rectangular-pulse memory realization. The denominator is then fixed; only the feedforward numerator is learned from the training prefix.

This separates generic recursive representation from receiver structure derived from the known SOE memory.

## Conditions

- BPSK, QPSK, 16-QAM
- exponential tau=0.10
- two-scale SOE 0.7 exp(-2t) + 0.3 exp(-30t)
- identity, flat Rayleigh, normalized 3-tap multipath
- 0, 10, 20, 30 dB SNR
- five deterministic seeds
- 128-symbol training prefix
- 384 held-out symbols

Total:

2 x 3 x 3 x 4 x 5 x 5 = 1800 receiver-condition rows.

The power-law memory is deliberately excluded: it is not an exact finite SOE model, so using an SOE receiver there would introduce model mismatch rather than test known-structure recovery.

## Receiver ladder

raw -> FIR-7 -> IIR-2 -> SOE-exact -> SOE-stable

SOE-exact uses the causal inverse denominator obtained from the known sampled SOE realization. SOE-stable reflects any analytically unstable inverse poles inside the unit circle before fitting the numerator. In the tested conditions the two variants are numerically identical, indicating that the derived inverse poles were already inside the unit circle.

The SOE receiver complexity is reported as total stored coefficients and separately as trainable coefficients. For one exponential: 3 total coefficients, 2 trainable feedforward coefficients, 1 recursive state, 3 MACs/sample. For the two-scale SOE: 5 total coefficients, 3 trainable feedforward coefficients, 2 recursive states, 5 MACs/sample.

## Executed run

The final reproducible run completed successfully on GitHub Actions:

- commit: cb9968c11d9d73b19791b9555cca6ae369686f83
- workflow run: 35745554935
- tests: 40 passed
- result rows: 1800
- Actions artifact: experiment-12b-results

A compact aggregate is stored in experiments/results/12B_summary_by_memory_snr.csv.

## Results

### 30 dB SNR

| Memory | Receiver | Mean EVM | Mean BER |
|---|---|---:|---:|
| exponential | FIR-7 | 28.791% | 0.0243 |
| exponential | IIR-2 | 29.078% | 0.0239 |
| exponential | SOE-exact | 32.816% | 0.0373 |
| two-scale | FIR-7 | 48.610% | 0.0988 |
| two-scale | IIR-2 | 48.510% | 0.0950 |
| two-scale | SOE-exact | 51.655% | 0.1131 |

The SOE-aware receiver therefore does not show a recovery advantage in the tested conditions. Its error remains close to the generic receivers but is consistently somewhat higher in the aggregate.

At 30 dB, the identity-channel control gives the same qualitative result: exponential memory is about 30.04% EVM for SOE-exact versus 29.23% for FIR-7 and 29.82% for IIR-2; the two-scale case is about 50.49% versus 48.90% and 48.86%, respectively.

## Interpretation

The result is a useful negative result:

MIN memory structure knowledge did not by itself reduce recovery error.

The SOE receiver is compact, however:

- exponential: 3 stored coefficients / 1 state
- two-scale: 5 stored coefficients / 2 states

That compactness should not be interpreted as a performance advantage. At matched or nearby arithmetic budgets, generic IIR-2 remains competitive or slightly better.

The result also indicates that memory structure knowledge is not equivalent to an optimal inverse. The receiver is derived from the known SOE realization, but it still operates under finite training, AWGN, symbol-rate observation, rectangular-pulse sampling, and—where present—an additional propagation channel.

## Scientific consequence

12B weakens the hypothesis that the special value of MIN/SOE memory in this communication task is simply that it gives a more compact inverse.

Current evidence:

MIN transforms signals does not imply that MIN has a receiver-complexity advantage.

The next useful test is therefore not another arbitrary equalizer order. The research should move toward identifying what property of the forward transformation could matter under propagation, if any.

That motivates Experiment 13: forward propagation stability, with signal preservation, receiver recoverability, and transform-domain stability measured as separate quantities.

## Boundary

12B tests the sampled SOE realization used by this repository. It does not prove or disprove invertibility of the continuous MIN operator, and it does not test whether MIN improves communication in general.
