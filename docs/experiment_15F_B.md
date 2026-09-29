# Experiment 15F-B — Communications Channel Stress

## Purpose

15F-B is the stress stage following 15F-A. It holds the temporal representation fixed at the 16-state budget and changes only the communications environment.

The question is:

> Does the representation relationship established by 15D/15E survive when the synthetic communications channel becomes harder?

## Lineage

- 15C: matched-timescale MIN-16 versus a one-pole control.
- 15D: equal-state comparison of MIN, logarithmic IIR, dense-state coordinate control, and FIR.
- 15E: actual temporal-state budgets 1, 2, 4, 8, 16.
- 15F-A: first transfer of those representations to controlled BPSK/QPSK symbol recovery.
- 15F-B: fixed N=16 stress test.

The representation constructors are imported directly from 15E. No second communications-specific implementation of MIN/IIR/FIR state construction is introduced.

## Fixed representation set

At every case:

- current_1 observation baseline, with the same linear-readout training discipline.
- min_16
- iir_logspread_16
- dense_ss_16
- fir_16

The dense representation remains a coordinate control.

## Stress axes

The grid varies:

- SNR: 0, 10, 20 dB.
- Memory family: the four 15E process families.
- Memory scale: 15E scale factors 4, 12, 32.
- Modulation: BPSK, QPSK, 16QAM.
- Channel stress: none, mild/strong multipath, mild/strong slow complex fading.

Multipath uses fixed causal delays of one and two symbols. Fading is a seeded slow complex Gauss-Markov process with two depths. The additional channel stress is independent of AWGN.

## Receiver protocol

Rectangular-pulse complex-baseband symbols are passed through the controlled temporal channel and AWGN. The resulting noisy stream is fed to the temporal representation.

The linear readout is fitted only on the first 512 transmitted symbols. The final 512 symbols are held out for BER, SER, EVM, and NMSE.

All temporal states are causal and no held-out labels are used.

## Why N=16 is fixed

15F-C is reserved for the explicit communications budget sweep 1 -> 2 -> 4 -> 8 -> 16.

Therefore 15F-B does not vary state budget. This keeps channel-stress effects separate from state-budget effects.

## Scope

15F-B tests robustness under controlled synthetic channel stress. It does not establish real-IQ receiver performance, universal communications superiority, or optimality of any representation.

## CI authority

CI runs the tests, executes the experiment, and uploads the CSV/JSON artifact. The retrieved CI artifact is the numerical record.
