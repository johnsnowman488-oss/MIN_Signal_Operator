# Experiment 09 — Digital signals through controlled MIN memory

Date: 2026-09-21

## Purpose

Experiment 09 moves from elementary deterministic signals to communication waveforms: BPSK, QPSK and 16-QAM.

The first question is not whether MIN improves BER. It is more basic:

> What waveform distortion and inter-symbol memory does each MIN kernel create before noise, propagation channels, equalization, or detection are introduced?

Each symbol is rectangularly held for 16 samples. Five independent seeds are used. The observation is sampled at the center of every symbol. A single complex scalar gain is fitted and removed before EVM is calculated.

## Kernels

- identity control
- exponential memory, `tau=0.10 s`
- positive two-scale SOE, `0.7 exp(-2t)+0.3 exp(-30t)`
- power-law memory, `(1+t/0.5)^(-0.70)`

## Results

| Signal | Identity | Exponential | Two-scale SOE | Power-law |
|---|---:|---:|---:|---:|
| BPSK | ~0% | 426.0% | 725.6% | 1766.2% |
| QPSK | ~0% | 404.0% | 676.6% | 1389.0% |
| 16-QAM | ~0% | 422.1% | 696.9% | 1554.3% |

Values are mean EVM over five seeds after only complex scalar gain fitting.

## Interpretation

The very large EVM values are not being interpreted as a communications failure of MIN in general. They show that the tested memory kernels are **not equivalent to a simple complex gain** for these symbol-rate settings.

In particular, the residual is consistent with temporal mixing/inter-symbol memory: the current sampled symbol depends on previous waveform history.

The experiment therefore establishes a useful baseline for the next stage:

[
	ext{MIN transformation}
\rightarrow
	ext{memory-induced ISI}
\rightarrow
	ext{possible equalization/detection problem}.
]

The result also shows why a later BER comparison must not compare raw MIN output directly against an unfiltered baseline. A fair communication experiment needs matched bandwidth/energy, a receiver model, and an equalizer or inverse appropriate to the tested transformation.

## Important limitation

The present experiment uses rectangular pulses and memory constants that are long relative to the 10 ms symbol period. It is intentionally a stress test of temporal memory, not a realistic wireless channel model.

No BER, SNR, fading, AWGN, matched filtering, equalization, or claim of communication benefit is made.

## Next step

Experiment 10 should introduce controlled AWGN and a receiver-side compensation layer. Before optimizing MIN, test whether the identified kernel can be inverted or equalized and measure BER/EVM versus SNR under a controlled channel.
