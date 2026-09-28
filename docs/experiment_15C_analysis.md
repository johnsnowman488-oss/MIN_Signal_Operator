# Experiment 15C — Extended Timescale Sweep with One-Pole IIR Control

## CI execution

- Workflow: Experiment 15C
- Successful run: `36438562042`
- Head commit: `00f386b65ddff4e8a0a156fc081ff65f18d6a21d`
- Tests: passed
- Numerical sweep: passed
- Artifact: `experiment-15c-results`
- Case rows: 2,880
- PCA rows: 46,080
- Repository/SOE equivalence relative error: (1.44\times10^{-11})

Three earlier CI runs failed because of a prediction-alignment bookkeeping bug. Those runs produced no artifact and are not experimental evidence. The final run above contains the corrected implementation.

## Design

15C extends the 15B-3 timescale sweep into the regime suggested by 15B-2:

[
4,8,12,20,32,64\times.
]

For each process and factor, three representations use the same noisy observation:

1. **MIN-16:** 16 positive uniform exponential modes.
2. **One-pole control:** conventional single exponential/IIR state whose kernel (t_{50}) matches the MIN kernel.
3. **Observation:** current noisy sample.

The same train/test split, targets, linear readout and SNR/seed grid are used.

Processes: short, multiscale, long, hidden_mix.

Tasks: one-step prediction and latent estimation.

## Aggregate results

Values below average over all four SNRs and five seeds.

| Process | Task | Representation | Best factor | Best mean NMSE |
|---|---|---:|---:|---:|
| short | prediction | MIN-16 | 32x | **0.1010** |
| short | prediction | one-pole | 20x | 0.1265 |
| short | latent | MIN-16 | 32x | **0.0582** |
| short | latent | one-pole | 20x | 0.0849 |
| multiscale | prediction | MIN-16 | 64x | **0.0827** |
| multiscale | prediction | one-pole | 64x | 0.2067 |
| multiscale | latent | MIN-16 | 20x | **0.2218** |
| multiscale | latent | one-pole | 64x | 0.4296 |
| long | prediction | MIN-16 | 64x | **0.0266** |
| long | prediction | one-pole | 64x | 0.0374 |
| long | latent | MIN-16 | 64x | **0.0234** |
| long | latent | one-pole | 64x | 0.0341 |
| hidden_mix | prediction | MIN-16 | 64x | **0.0669** |
| hidden_mix | prediction | one-pole | 64x | 0.1534 |
| hidden_mix | latent | MIN-16 | 20x | **0.3508** |
| hidden_mix | latent | one-pole | 64x | 0.6493 |

MIN-16 therefore has lower best-achieved NMSE than the one-pole control in **all 8 process/task combinations**.

Relative to the best one-pole control, the MIN reductions in NMSE are approximately:

- short prediction: 20%
- short latent estimation: 31%
- multiscale prediction: 60%
- multiscale latent estimation: 48%
- long prediction: 29%
- long latent estimation: 31%
- hidden_mix prediction: 56%
- hidden_mix latent estimation: 46%

These are comparisons of the best scale found for each representation, so they should not be interpreted as a universal complexity-normalized superiority claim.

## Timescale result

15C changes the interpretation of 15B-3.

There is **not** a universal optimum at the fastest tested factor:

- short tasks peak around 32x;
- multiscale latent estimation peaks around 20x;
- hidden_mix latent estimation peaks around 20x;
- long prediction/latent estimation continue improving through 64x;
- multiscale and hidden_mix prediction also continue improving through 64x.

Thus the useful scale is both **process- and task-dependent**.

The earlier 15B-2 long-memory reversal is therefore not an isolated matched/displaced artifact. The extended sweep shows a smooth progression toward faster useful kernels, at least through 64x for several tasks.

At the same time, the presence of interior optima for short and multiscale/hidden latent tasks demonstrates that “faster is always better” is also false.

## Strongest result

The most important new observation is not simply that MIN performs well.

It is that the 16-mode representation retains a task-relevant advantage over a conventional one-pole exponential state when the latter is matched to the same effective temporal (t_{50}).

For example:

- multiscale prediction: MIN 0.0827 vs one-pole 0.2067;
- hidden_mix prediction: 0.0669 vs 0.1534;
- multiscale latent estimation: 0.2218 vs 0.4296;
- hidden_mix latent estimation: 0.3508 vs 0.6493.

This is evidence that the observed effect is not reducible to simply choosing a better single exponential time constant.

## What remains unresolved

The one-pole control is useful but not yet a complete complexity-matched baseline.

MIN-16 contains 16 state coordinates while the control contains one. Therefore 15C establishes:

> richer exponential state geometry can outperform a single-pole temporal representation at matched effective temporal scale.

It does **not** yet establish:

> MIN is more computationally efficient than conventional state-space/IIR filtering.

The next equal-budget comparison should therefore control state dimension, parameter count, or operations/sample.

A second issue is the very fast-kernel regime. At 64x, the representation becomes increasingly local. We should test whether the performance eventually converges toward the instantaneous observation or toward a conventional short-memory filter.

## Relation to previous experiments

15B-3 showed that process-timescale matching was not sufficient and that performance often improved toward faster kernels.

15C adds the crucial control:

[
	ext{MIN-16} quad vs quad 	ext{one-pole IIR}
]

at approximately matched kernel (t_{50}).

Together, 13B, 14C, 15B-2, 15B-3 and 15C support the narrower hypothesis:

[
oxed{
	ext{temporal kernel geometry}
ightarrow
	ext{realized state geometry}
ightarrow
	ext{task accessibility}
}
]

and provide evidence that **multi-mode geometry can carry task-relevant information beyond a single exponential timescale**.

They do not yet prove a universal MIN advantage over FIR/IIR/state-space models.

## Recommended 15D

The next experiment should be an **equal-state/computational-budget comparison**.

For example:

- MIN-16
- 16 independent conventional first-order IIR states
- conventional low-order state-space models with comparable state dimension
- finite-history/FIR representation with comparable feature budget

Then hold the same task, observation, train/test split and readout fixed.

Measure:

1. NMSE
2. state dimension
3. parameter count
4. approximate multiply/add operations per sample
5. memory footprint
6. conditioning
7. task-relevant dimension.

This would determine whether the 15C multi-mode advantage is merely a consequence of giving MIN more state coordinates, or whether its particular exponential rate geometry provides a useful representation-per-compute tradeoff.
