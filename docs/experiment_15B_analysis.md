# Experiment 15B — Process × Geometry Alignment: First Run

## Purpose

15B tests whether the functional usefulness of memory-kernel geometry depends on
the temporal scales present in the generating process.

The first implementation uses four controlled latent processes:

- short: AR timescale 0.025 s
- multiscale: 0.025 s + 0.50 s
- long: AR timescale 0.50 s
- hidden_mix: equal mixture of 0.025 s and 0.50 s latent components

Each process is observed with four SNR levels and represented with the same
three 16-mode SOE geometries used in 15A: clustered, logspread, and wide.
Two causal tasks are evaluated: one-step prediction and latent-state
estimation.

## Execution

The CI run completed successfully.

Grid:

- 4 processes
- 2 tasks
- 3 geometries
- 4 SNR levels
- 5 seeds
- 16 retained PCA dimensions

All tests passed, including numerical equivalence of the independent 15-series
SOE recurrence against the repository SOEMemory implementation.

## Main numerical observation

The first run does not support a simple rule that the broadest or apparently
timescale-matched geometry is automatically functionally superior.

Mean full-state NMSE over SNRs/seeds:

| Process | Task | Clustered | Logspread | Wide |
|---|---|---:|---:|---:|
| short | prediction | 0.578 | 4.985 | 276.686 |
| short | latent estimation | 0.551 | 3.917 | 177.870 |
| multiscale | prediction | 0.451 | 2.056 | 2318.448 |
| multiscale | latent estimation | 0.614 | 6.263 | 8062.618 |
| long | prediction | 0.113 | 0.342 | 910.861 |
| long | latent estimation | 0.107 | 0.329 | 789.581 |
| hidden_mix | prediction | 0.357 | 0.908 | 961.033 |
| hidden_mix | latent estimation | 0.720 | 25.017 | 80230.435 |

Clustered geometry is generally the strongest of the three in this first
controlled run; logspread is sometimes competitive, particularly for
multiscale prediction at higher SNR, while wide is frequently catastrophically
poor.

## Interpretation

This is not yet evidence for or against temporal alignment itself.

The reason is methodological: the three 15A geometries were retained unchanged
for 15B. They were designed to span different rate distributions, not to create
a process-specific matched-versus-mismatched intervention.

The results nevertheless establish an important constraint:

> Simply adding more separated memory timescales does not automatically make a
> representation more useful for a temporally structured process.

The extreme errors for the wide geometry also reinforce the 14D finding that
kernel geometry can strongly affect numerical conditioning and downstream
readout stability.

The state-participation dimensions increased from approximately 1.00 for
clustered to 1.16–1.30 for logspread/wide, but greater realized state
complexity did not imply better task performance.

## Consequence for the next 15B pass

A clean alignment experiment should therefore introduce process-specific
matched kernels while preserving mode count, positive uniform weights, and D_eff.

For each process, compare:

1. a kernel whose decay rates are centered around the process timescale(s);
2. a deliberately displaced/mismatched kernel;
3. a broad multiscale kernel.

That isolates temporal alignment from the pre-existing clustered/logspread/wide
geometry labels.

The current first run should be retained rather than overwritten because it is
a useful negative/control result: broad memory geometry is not automatically
beneficial even when the underlying process has long or multiscale dynamics.
