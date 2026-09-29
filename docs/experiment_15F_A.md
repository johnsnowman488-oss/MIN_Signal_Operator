# Experiment 15F-A — Synthetic Communications Validation

## Question

Does temporal information retained by a fixed MIN state translate into measurable receiver-relevant performance when the downstream objective is symbol recovery?

## Protocol

BPSK/QPSK → causal channel memory → complex AWGN → temporal representation → linear readout → held-out symbol estimates → BER/EVM/SER.

The controlled grid uses five seeds, four memory regimes, five SNR points, and state budgets 1/2/4/8/16.

Representations:
- current_1: current noisy observation.
- min_N: positive exponential MIN/SOE state bank centered on the known channel-memory timescale.
- iir_logspread_N: conventional log-spread exponential bank.
- fir_N: N-sample causal finite-history state.

The readout is fit only on the first 256 transmitted symbols and evaluated on the final 256. State construction is causal.

## Metrics

BER is computed after hard constellation decisions. SER is the symbol error rate after hard decisions. EVM is RMS symbol error normalized by RMS reference-symbol magnitude and reported as percent, matching the repository metric convention and standard symbol-point EVM definitions. citeturn2search0turn2search3

## Scope boundary

15F-A is synthetic validation. It can establish controlled task-level behavior, not real-IQ receiver performance. Real IQ remains reserved for the 16-series.

## Reproducibility

CI runs the tests, executes the experiment, and uploads the three result files as an artifact. Numerical claims should be made only from the retrieved artifact.
