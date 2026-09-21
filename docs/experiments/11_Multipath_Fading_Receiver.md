# Experiment 11 — Controlled Multipath/Fading + Receiver Equalization

## Purpose

Experiment 10 established that MIN memory distortion can be partly compensated by a short receiver equalizer under AWGN. Experiment 11 adds a controlled communication channel to separate memory distortion from ordinary propagation effects.

## Conditions

- BPSK, QPSK, 16-QAM
- 512 symbols/run; 16 samples/symbol; 100 symbols/s
- 128-symbol training prefix; 384 held-out evaluation symbols
- 5 seeds
- SNR = 0, 10, 20, 30 dB
- MIN cases: identity, exponential tau=0.10 s, two-scale SOE, power-law alpha=0.70
- Channels: normalized flat Rayleigh fading and normalized 3-tap multipath

## Channel models

Flat fading uses one unit-magnitude complex Rayleigh realization per run.

The 3-tap channel uses normalized taps of the form

`[1, 0.45 exp(j phi1), 0.25 exp(j phi2)]`.

The phases are independently seeded. Channel energy is normalized so comparisons do not include arbitrary channel-gain differences.

## Receiver comparison

- raw: no equalization
- fir7: 7-tap causal symbol-spaced FIR trained on the 128-symbol prefix
- fir15: 15-tap causal symbol-spaced FIR trained on the same prefix

Both equalizers are evaluated only on held-out symbols.

## Important control

The MIN memory filters are normalized to unit discrete kernel sum in this experiment. This prevents an arbitrary DC/power scaling of a memory kernel from being confused with channel performance.

This experiment therefore asks whether memory-induced temporal structure remains distinguishable from conventional multipath and whether increasing receiver memory helps compensate it.

## Output

`python experiments/11_multipath_fading_receiver.py` produces:

- `experiments/results/11_multipath_fading_receiver_results.csv`
- `experiments/results/11_multipath_fading_receiver_summary.json`

The sweep contains 1,440 cases: 3 signals × 4 memory regimes × 2 channels × 3 receivers × 4 SNR values × 5 seeds.

## Interpretation

A lower BER/EVM after FIR equalization means the observed distortion is more compensable by that receiver model. It does not establish that MIN is better than a conventional channel equalizer, nor does it establish a communication advantage.

The next benchmark should compare the same controlled channels against conventional FIR/IIR and Volterra or memory-polynomial baselines with matched parameter budgets.