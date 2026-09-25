# Experiment 14B — Kernel-condition task recovery

## Aim

Experiment 14B tests the environment-alignment question:

> Does using a kernel aligned with the observation environment change downstream digital-symbol recovery?

The observed task signal is held fixed across four representation conditions:

1. **informed** — oracle environment-derived 16-mode SOE/MIN kernel;
2. **estimated** — kernel estimated from an independent finite/noisy environment-only observation;
3. **mismatched** — oracle kernel belonging to a different declared environment;
4. **raw_center** — noisy observation at the symbol center, with no MIN/SOE state.

The MIN conditions use the scalar operator output z=w^T q and the same complex linear readout. PCA is deliberately omitted because Experiment 14A already established the full-state task-dimension question. The scalar output is essential here: with a fixed gamma dictionary, an unweighted full modal state depends on the decay rates rather than the fitted kernel weights, so weight-only kernel changes would not constitute a meaningful alignment intervention.

## Controlled grid

Three digital signal families (BPSK, QPSK, 16-QAM), five environment families, four SNR levels (0/10/20/30 dB), and five independent seeds give 300 underlying task cases. Four representation conditions produce 1,200 output rows.

The estimated kernel is derived from a separate 2,048-sample environment-only probe at the same SNR as the task observation. The mismatch is a fixed cyclic map between environment families, so it is not chosen using task performance.

The held-out 512-symbol target sequence is not used for kernel estimation or readout fitting.

## Primary quantities

The primary task metric is held-out normalized symbol reconstruction MSE (NMSE). The key comparison is the NMSE ratio to the informed condition. Kernel-fit error and state-geometry descriptors are recorded alongside task performance.

This experiment does not claim that any one representation is universally better. It tests whether changing kernel alignment produces a reproducible change under the declared task and decoder.

## Relation to 14A

14A asked how much of the full environment-informed MIN/SOE state is task-relevant.

14B holds the downstream task and decoder fixed and changes the relationship between the MIN kernel and the true environment, using the scalar MIN output so that the kernel itself changes the representation. Thus:

- 14A probes **task-relevant dimensionality**;
- 14B probes **environment-kernel alignment**.

A later 14C can hold kernel condition fixed while varying SOE geometry to isolate geometry-to-task effects.
