# Experiment 15B-2 Analysis — Process-Specific Memory Alignment

## Purpose

15B-2 isolates temporal alignment from the generic 15A/15B geometry labels. For each generating process, three 16-mode, positive-uniform SOE kernels are compared:

- matched — rates concentrated around the process decay rate(s);
- displaced — the same number of modes and weights, but rate support shifted away;
- broad — fixed log-spread support from 0.5 to 100.

All three have the same nominal mode count and uniform-weight effective dimension, D_eff = 256. Observation noise, train/test split, PCA, linear readout and ridge regularization are held fixed within each paired case.

The implementation is in experiments/15B2_process_specific_alignment.py, with independent 15-series utilities in experiments/15_utils.py.

## Numerical execution

The numerical core was independently executed over:

- 4 processes: short, multiscale, long, hidden_mix
- 2 tasks: prediction, latent estimation
- 3 alignment roles
- 4 SNR levels: 0, 10, 20, 30 dB
- 5 seeds

This gives 480 case rows and 7,680 PCA-dimension rows in the full protocol.

A design issue was found during the first execution: the single-scale matched and displaced kernels initially duplicated eight rates. This was corrected before interpreting results; the corrected single-scale kernels use 16 unique rates.

## Corrected aggregate results

Mean held-out full-state NMSE across SNRs and seeds:

| Process | Task | Matched | Displaced | Broad |
|---|---|---:|---:|---:|
| short | prediction | 0.226 | 1.196 | 280.447 |
| short | latent estimation | 0.187 | 1.118 | 197.354 |
| multiscale | prediction | 0.883 | 1351.770 | 2244.723 |
| multiscale | latent estimation | 2.140 | 1951.198 | 7013.572 |
| long | prediction | 7.514 | 0.037 | 1179.583 |
| long | latent estimation | 7.431 | 0.033 | 1051.334 |
| hidden_mix | prediction | 0.392 | 4281.977 | 1700.791 |
| hidden_mix | latent estimation | 3.968 | 3250.763 | 82281.987 |

Mean task-relevant PCA dimension:

| Process | Task | Matched | Displaced | Broad |
|---|---|---:|---:|---:|
| short | prediction | 4.75 | 1.95 | 1.00 |
| short | latent estimation | 4.80 | 1.95 | 1.15 |
| multiscale | prediction | 6.40 | 1.00 | 1.00 |
| multiscale | latent estimation | 3.95 | 1.00 | 1.00 |
| long | prediction | 1.10 | 3.20 | 1.10 |
| long | latent estimation | 1.10 | 3.25 | 1.00 |
| hidden_mix | prediction | 5.85 | 1.00 | 1.00 |
| hidden_mix | latent estimation | 1.80 | 1.00 | 1.00 |

## Interpretation

The experiment does not support a universal rule that the kernel should simply match the process autocorrelation timescale.

There is process-specific structure:

- short: matched is substantially better than displaced and broad.
- multiscale: matched is dramatically better than displaced and broad.
- hidden_mix: matched is substantially better than the alternatives for both tasks.
- long: the apparently matched slow kernel is much worse than the deliberately displaced fast kernel.

The long-memory result is the critical counterexample. A kernel with the same decay timescale as the generating process applies another long low-pass/integral operation to an already slow state. For the prediction and latent-recovery tasks used here, a faster kernel can preserve more immediately useful variation than a same-timescale memory kernel.

The more defensible hypothesis is:

> Useful memory geometry depends on the interaction between process temporal structure and the downstream task, rather than on process timescale matching alone.

## Conditioning

State conditioning remains extreme for narrow rate clusters. Median state condition numbers were approximately:

- short: matched 1.54e16, displaced 1.06e16, broad 4.57e7
- multiscale: matched 7.66e12, displaced 3.59e15, broad 2.90e7
- long: matched 8.90e15, displaced 1.36e16, broad 1.69e8
- hidden_mix: matched 1.90e13, displaced 3.38e15, broad 7.28e7

The long-process result cannot be explained simply by the matched kernel having the worst condition number: its displaced alternative is also extremely ill-conditioned while achieving far lower task error.

## Relation to earlier experiments

15B-1 showed that fixed geometry families across different processes did not establish a clean alignment effect.

15B-2 provides the controlled intervention. Together with 13B and 14C, the evidence now supports a narrower chain:

kernel rate geometry -> realized state geometry -> task accessibility

but not:

kernel rate geometry -> universally better task performance

and not:

process timescale = optimal memory timescale.

## Next experiment

The next useful step is 15B-3: task-optimal timescale sweep.

For each process/task, continuously sweep a single characteristic kernel scale around the process scale while holding mode count and weights fixed. Measure task error, task-relevant dimension, state conditioning, kernel effective timescale, and overlap between process temporal spectrum and kernel transfer function.

This would determine whether the long-process counterexample reflects a systematic task-optimal offset between process memory and useful memory-filter timescale.

No universal MIN superiority is claimed by 15B-2.
