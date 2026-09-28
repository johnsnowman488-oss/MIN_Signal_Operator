# Experiment 15A — Geometry × Task Generalization

## Execution

- Grid: 3 signal families × 5 environments × 4 SNR levels × 5 seeds = 300 paired cases.
- Geometries: clustered, logspread, wide.
- Tasks: denoising, one-step prediction, symbol recovery.
- Each geometry uses 16 positive uniform SOE weights, so nominal mode count and D_eff are matched.
- Every geometry receives the identical noisy realization within a paired case.
- PCA/readout protocol is identical across geometries.
- CI execution completed successfully; the 15A unit tests also passed.

## Main result

The geometry effect seen in 14C generalizes strongly at the representation-complexity level.

Mean task-relevant dimension D_task:

| Task | Clustered | Logspread | Wide |
|---|---:|---:|---:|
| Denoising | 1.90 | 9.30 | 10.62 |
| Prediction | 1.46 | 8.16 | 8.93 |
| Symbol recovery | 2.04 | 10.20 | 11.70 |

Across the same paired cases, the median increase in D_task relative to clustered geometry was approximately +8 dimensions for logspread and +11 to +12 dimensions for wide, depending on task.

The realized state participation dimension also increased systematically:

- clustered ≈ 1.00
- logspread ≈ 1.16
- wide ≈ 1.25

This reproduces the qualitative 13B relationship between rate geometry and realized state structure while showing that the effect is not confined to the original 14C symbol-recovery task.

## Important negative result

Geometry-dependent representation structure does not imply universal task-performance improvement.

The raw/current observation baseline remains substantially better for these tasks overall. Mean full-state NMSE was:

| Task | Clustered | Logspread | Wide | Raw baseline |
|---|---:|---:|---:|---:|
| Denoising | 0.877 | 0.650 | 1.345 | 0.161 |
| Prediction | 0.908 | 0.735 | 1.420 | 0.262 |
| Symbol recovery | 0.857 | 0.500 | 0.716 | 0.159 |

Thus 15A does not establish that MIN is a better task representation than the raw observation.

The useful finding is narrower:

> Changing only the arrangement of memory timescales can systematically change the dimensional structure required for downstream task access, and this effect persists across multiple task definitions.

Performance remains task-dependent and can move differently from D_task. This is evidence against treating mode count or a single scalar effective dimension as a complete description of representation usefulness.

## Interpretation boundary

15A strengthens the hypothesis

kernel geometry → realized state geometry → task-relevant geometry

but does not yet establish

kernel geometry → universally better task performance.

The next useful question is therefore whether these geometry effects become task-relevant specifically when the underlying process has temporal structure matched to the memory timescales. That is the motivation for 15B.

## Reproducibility

The 15 series uses its own experiments/15_utils.py utility layer. It does not import 14-series experiment modules. The utility layer preserves the validated SOE state recurrence and checks numerical equivalence against the repository SOEMemory implementation before the experiment runs.

Full PCA-level and case-level outputs were retained as the CI artifact for the 15A run.
