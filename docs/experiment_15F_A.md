# Experiment 15F-A — Synthetic Communications Validation

## Purpose

15F-A is the task-transfer experiment at the end of the synthetic 15-series. It does not invent a new memory architecture. It asks whether the representation/budget behavior established in 15D and 15E survives when the downstream objective is symbol recovery.

## Lineage

- 15C: matched-timescale MIN-16 versus a conventional one-pole control.
- 15D: equal 16-state comparison of MIN, logarithmic IIR, dense-state coordinate control, and FIR.
- 15E: actual temporal-state budget sweep at N = 1, 2, 4, 8, 16 using those same four representation families.
- 15F-A: carries the four 15E families unchanged into a controlled complex-baseband communications task.

The 15F implementation imports the 15E representation constructors directly. This is intentional: a change in state construction must not be mistaken for a communications effect.

## Protocol

BPSK/QPSK rectangular-pulse baseband -> causal complex channel-memory mixture -> complex AWGN -> 15E temporal representation -> one linear symbol readout -> held-out symbol estimates.

The channel uses the same four 15E process families (short, multiscale, long, hidden_mix), their timescale parameters, and the same scale-factor parameterization. The first 15F-A run uses scale factors 4, 12, and 32 and SNR 0, 10, and 20 dB as a compact subset of the 15E envelope; 15F-B can widen the stress grid.

## Representations

For every N:
- min_N
- iir_logspread_N
- dense_ss_N
- fir_N

Plus current_1 as an observation-only baseline.

The dense state is retained because 15D/15E explicitly use it as a coordinate-control check. It is not treated as an independent memory mechanism.

## Train/test discipline

The linear readout is fitted only on the first 512 transmitted symbols and evaluated on the final 512. The temporal states are causal and are allowed to process the incoming held-out waveform, as an online receiver would; no held-out labels are used in state construction or fitting.

## Metrics

- BER: bit errors after hard BPSK/QPSK decisions.
- SER: symbol errors after hard constellation decisions.
- EVM: repository min.metrics.equalization.evm, reported as percent.
- NMSE: retained as a bridge metric to 15C–15E.

## What 15F-A can establish

It can test whether the 15E state-budget/representation relationship transfers to receiver-level metrics under a known synthetic ISI channel.

It cannot establish real-IQ receiver performance, universal communications superiority, or optimality of the MIN dictionary.

## CI authority

CI runs the tests, executes the experiment, and uploads the CSV/JSON artifact. GitHub Actions artifacts are persistent workflow outputs intended for later inspection, so the artifact—not a local rerun or console approximation—is the numerical record.
