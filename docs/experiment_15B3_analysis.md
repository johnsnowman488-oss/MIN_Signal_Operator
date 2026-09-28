# Experiment 15B-3 — Task-Optimal Timescale Sweep

## Status

**CI execution completed successfully.**

- Workflow: `Experiment 15B-3`
- Run: `36435852512`
- Head commit: `351e843346ce42aeb05a2d757cb8af772a5600e9`
- Commit message: `Trigger 15B-3 CI execution`
- Job: `run-15b3`
- Tests: passed
- Experiment execution: passed
- Artifact upload: passed
- Case rows: 1,120
- PCA rows: 17,920
- Repository/SOE state equivalence relative error: `1.439e-11`

## Purpose

15B-3 swept a multiplicative memory-timescale factor around each process' characteristic decay scale while holding:

- nominal modes (L=16)
- positive uniform weights
- observation realization
- train/test split
- PCA procedure
- linear readout
- task definition

fixed.

Scale factors were:

`0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0`.

A factor below one gives a slower memory kernel; a factor above one gives a faster kernel.

The intended question was whether task-optimal memory timescales coincide with, or are systematically displaced from, process timescales.

## Aggregate held-out full-state NMSE

Means are over the four SNR values and five seeds.

| Process | Task | 0.25x | 0.5x | 0.75x | 1x | 1.5x | 2x | 4x |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| short | prediction | 0.476 | 0.321 | 0.259 | 0.221 | 0.181 | 0.162 | **0.131** |
| short | latent estimation | 0.446 | 0.287 | 0.221 | 0.182 | 0.142 | 0.122 | **0.089** |
| multiscale | prediction | 34.118 | 10.546 | 2.445 | 0.897 | 0.255 | 0.163 | **0.112** |
| multiscale | latent estimation | 74.691 | 23.816 | 3.168 | 2.042 | 0.970 | 0.602 | **0.307** |
| long | prediction | 610.820 | 107.699 | 27.254 | 6.856 | 1.762 | 0.862 | **0.114** |
| long | latent estimation | 608.657 | 107.173 | 27.106 | 6.783 | 1.727 | 0.835 | **0.107** |
| hidden_mix | prediction | 53.732 | 18.881 | 1.758 | 0.486 | 0.171 | 0.123 | **0.088** |
| hidden_mix | latent estimation | 239.662 | 58.025 | 7.575 | 4.402 | 2.326 | 1.360 | **0.539** |

Every process/task pair improved monotonically across the tested scale grid; the minimum was always at the fastest tested factor, 4x.

## Main result

The 15B-2 long-memory counterexample is **not a two-point anomaly**.

The full sweep shows a smooth and very strong preference for faster kernels:

- long/prediction: 1x → 4x gives about **60.3x lower NMSE**
- long/latent estimation: about **63.2x lower**
- multiscale/prediction: about **8.0x lower**
- multiscale/latent estimation: about **6.7x lower**
- hidden_mix/prediction: about **5.5x lower**
- hidden_mix/latent estimation: about **8.2x lower**
- short/prediction: about **1.68x lower**
- short/latent estimation: about **2.04x lower**

The effect is present at every tested SNR: for every process/task/SNR combination, the 4x factor had the lowest mean across the five seeds.

## What this does to the working hypothesis

The original 15B-3 hypothesis was that there might be a task-specific interior optimum, systematically displaced from the process timescale.

That specific prediction is **not supported yet**.

Instead, the present evidence says:

> In these pointwise/one-step tasks, useful temporal memory can be substantially faster than the characteristic process memory, and the tested optimum may lie beyond the current sweep boundary.

Thus:

`process timescale != automatically useful representation timescale`.

The result is compatible with a stronger statement:

`useful kernel geometry is task-mediated, rather than determined directly by process autocorrelation timescale`.

But the second step needs another control before it can be made confidently.

## Critical confound revealed by 15B-3

Increasing the scale factor simultaneously makes the exponential memory shorter and the representation more local in time.

Therefore a faster kernel may simply behave more like a low-pass/denoising version of the current observation, rather than exposing a special long-memory representation.

The current experiment therefore cannot distinguish:

1. a genuinely useful memory-state geometry,
2. ordinary bandwidth/lag optimization of an exponential filter,
3. movement toward an almost instantaneous observation representation.

This is especially important because all 16 rates within a single-scale process are tightly clustered. Their internal diversity is small, so the experiment is primarily sweeping **memory depth/bandwidth**, not introducing a qualitatively different 16-mode geometry.

## Conditioning

State condition numbers remain extremely large, especially for clustered single-scale kernels.

For the long process, for example, median conditioning changes from roughly:

- 1x: `4.20e15`
- 2x: `3.28e15`
- 4x: `2.27e15`

while NMSE changes from about 6.86 → 0.86 → 0.114 for prediction.

Therefore conditioning is correlated with the improvement but **cannot by itself explain the magnitude of the long-process effect**. The performance change is much larger than the modest reduction in condition number.

Across the seven scale factors, log-condition and log-NMSE are positively correlated for every process/task pair, but this should be interpreted as a coupled numerical/representation effect rather than a proof that conditioning causes the task result.

## Task-relevant dimension

The dimensionality result is not uniform.

Examples:

- short prediction: mean (D_{task}) decreases from 4.75 at 1x to 3.35 at 4x
- multiscale prediction: 7.30 → 6.45
- hidden_mix prediction: 6.15 → 6.45
- long prediction: 1.20 → 3.80

So lower task error does **not** require lower task-relevant dimension.

This reinforces the distinction:

`L != D_eff != D_basis != D_state != D_task`.

Timescale tuning changes both task accessibility and the geometry through which that accessibility is achieved.

## Relation to 15B-2

15B-2 compared process-specific matched and displaced kernels. For the long process, the displaced kernel was centered around a much faster rate and substantially outperformed the process-timescale-matched kernel.

15B-3 now shows that the improvement begins well before the 15B-2 displaced point and continues through 4x.

This suggests that the 15B-2 result may be part of a broader fast-kernel trend rather than an isolated matched-vs-mismatched reversal.

The 15B-2 displaced long kernel was approximately 20x faster than the process characteristic rate, so the current 4x sweep has not yet tested the region containing that earlier optimum candidate.

## Current conclusion

15B-3 supports:

> **The temporal scale of the process is not, by itself, the temporal scale that should be used by the representation.**

It does **not** yet establish:

> **There is a universal task-optimal displaced timescale.**

It also does not establish that the advantage is unique to MIN rather than a conventional exponential-memory/IIR bandwidth effect.

The strongest empirical chain currently remains:

`kernel rate geometry -> realized state geometry -> task accessibility`.

The 15B branch now adds:

`process temporal structure -> task-dependent useful memory scale`

as a working hypothesis, but the identity/filter confound must be separated before this becomes a central claim.

## Recommended next experiment

### 15B-4 — Extended scale sweep + conventional exponential control

Extend the scale factor above 4x, including the 15B-2 long-process displaced regime (approximately 20x), while retaining the same paired design.

At minimum test:

`4, 8, 12, 20, 32, 64x`.

For each factor compare:

- 16-mode MIN/SOE state
- equivalent single exponential/IIR state
- current noisy observation baseline

using the same observation, split, readout and task.

This separates:

`memory-timescale benefit`

from

`ordinary one-pole filtering / approach-to-identity behavior`.

A second useful diagnostic is to record the correlation between each state representation and the instantaneous observation, so the sweep can explicitly show whether performance gains arise as the representation becomes increasingly local.

## Interpretation discipline

15B-3 is a positive result for **timescale sensitivity**, not a positive result for universal MIN superiority.

The most important finding is actually methodological: process autocorrelation scale cannot be used as a sufficient proxy for task-useful memory scale, and the experiment exposes the need for conventional filter controls before attributing the effect to structured memory geometry.
