# Experiment 14B — Kernel-condition task recovery

## Aim

Experiment 14B tests the environment-alignment question:

> Does using a kernel aligned with the observation environment change downstream digital-symbol recovery?

The observed task signal is held fixed across four representation conditions:

1. **informed** — oracle environment-derived 16-mode SOE/MIN kernel;
2. **estimated** — kernel estimated from an independent finite/noisy environment-only observation;
3. **mismatched** — oracle kernel belonging to a different declared environment;
4. **raw_center** — noisy observation at the symbol center, with no MIN/SOE state.

The first three use the full 16-dimensional modal state and the same complex linear readout. PCA is deliberately omitted because Experiment 14A already established the state-dimension question.

## Controlled grid

Three digital signal families (BPSK, QPSK, 16-QAM), five environment families, four SNR levels (0/10/20/30 dB), and five independent seeds give 300 underlying task cases. Four representation conditions produce 1,200 output rows.

The estimated kernel is derived from a separate 2,048-sample environment-only probe at the same SNR as the task observation. The mismatch is a fixed cyclic map between environment families, so it is not chosen using task performance.

The held-out 512-symbol target sequence is not used for kernel estimation or readout fitting.

## Primary quantities

The primary task metric is held-out normalized symbol reconstruction MSE (NMSE). The key comparison is the NMSE ratio to the informed condition. Kernel-fit error and state-geometry descriptors are recorded alongside task performance.

This experiment does not claim that any one representation is universally better. It tests whether changing kernel alignment produces a reproducible change under the declared task and decoder.

## Relation to 14A

14A asked how much of the full environment-informed MIN/SOE state is task-relevant.

14B holds that representation at the full 16-state level and changes only the relationship between the kernel and the true environment. Thus:

- 14A probes **task-relevant dimensionality**;
- 14B probes **environment-kernel alignment**.

A later 14C can hold kernel condition fixed while varying SOE geometry to isolate geometry-to-task effects.
