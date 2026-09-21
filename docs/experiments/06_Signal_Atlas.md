# Experiment 06 — Synthetic MIN signal atlas

Date: 2026-09-21

## Purpose

Characterize the direct action of MIN memory kernels on elementary deterministic signals before introducing communication channels or learned task objectives.

This experiment is descriptive rather than a performance contest.

The reference signals are:

- impulse;
- rectangular pulse;
- Gaussian pulse;
- sine;
- two-tone signal;
- linear chirp.

The memory cases are:

[
I,qquad
k(t)=e^{-t/0.10},qquad
k(t)=0.7e^{-2t}+0.3e^{-30t},
qquad
k(t)=left(1+t/0.5ight)^{-0.7}.
]

For each pair, record:

- output energy and energy ratio;
- output peak and peak ratio;
- input/output correlation;
- lag of maximum cross-correlation.

The experiment deliberately does not use output-vs-input NMSE as its primary metric: MIN is a transformation, so increased difference from the input is not itself evidence of failure.

## Scientific role

Experiment 06 establishes the signal-level baseline needed before task-oriented experiments.

The next signal stage should add frequency-response measurements, then digitally modulated signals, followed by controlled noise and channel models.
