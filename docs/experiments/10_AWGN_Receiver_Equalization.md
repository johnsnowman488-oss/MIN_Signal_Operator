# Experiment 10 — Controlled AWGN + Receiver Equalization

## Question

Experiment 09 showed that the tested MIN memory transformations create substantial temporal distortion at symbol sampling. Experiment 10 asks the next question:

> Can that distortion be compensated at the receiver, and how do BER and EVM scale with SNR?

This is a receiver study. It does not assume that MIN improves communication performance.

## Controlled setup

- Signals: BPSK, QPSK, 16-QAM
- Symbols/run: 512
- Samples/symbol: 16
- Symbol rate: 100 symbols/s
- Training prefix: first 128 symbols
- Evaluation: remaining 384 held-out symbols
- Seeds: 0–4
- SNR: 0, 5, 10, 15, 20, 25, 30 dB
- Memory cases: identity; exponential tau = 0.10 s; two-scale SOE 0.7 exp(-2t) + 0.3 exp(-30t); power-law alpha = 0.70, tau = 0.50 s

AWGN is scaled from the measured clean waveform average power. Noise is added before symbol-center sampling.

## Receiver cases

1. **raw** — no compensation.
2. **scalar** — complex scalar estimated from the training prefix.
3. **fir7** — causal 7-tap symbol-spaced complex FIR equalizer, trained only on the 128-symbol prefix with a small ridge term.

The equalizer is therefore not fitted on the held-out evaluation symbols.

## MIN numerical realization

For uniform sampling, the causal trapezoidal MIN integral can be evaluated as a discrete convolution plus the two endpoint half-weight corrections. Experiment 10 uses FFT convolution for this step so the AWGN sweep does not require the quadratic reference implementation for every trial.

## Metrics

For held-out symbols:
- EVM = sqrt(mean(|estimate - transmitted|^2) / mean(|transmitted|^2))
- BER using the deterministic constellation labeling of the signal generators.

## Reproducibility

Run `python experiments/10_awgn_receiver_equalization.py`.

It writes `experiments/results/10_awgn_receiver_equalization_results.csv` and `experiments/results/10_awgn_receiver_equalization_summary.json`.

The experiment contains 1,260 result rows: 3 signals × 4 memory cases × 3 receivers × 7 SNR values × 5 seeds.

## Interpretation rule

The central comparison is within each memory case, especially raw/scalar versus the held-out FIR equalizer as SNR changes. A reduction in BER/EVM after equalization shows compensable receiver-side memory distortion; it does not by itself establish a communication advantage for MIN.

The next experiment should introduce controlled multipath/fading and compare MIN against conventional FIR/IIR or memory-polynomial/Volterra baselines under matched conditions.