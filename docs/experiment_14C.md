# Experiment 14C — Geometry → Task Relevance

## Aim

14C isolates the geometry-to-task link:

    kernel rate geometry → MIN/SOE state structure → digital-symbol recovery

The task environment is held fixed within each paired case. The representation
kernel is varied across the three rate geometries established in 13A/13B:

- clustered: 16 rates from 9.51–10.51 s^-1;
- logspread: 16 rates from 2–30 s^-1;
- wide: 16 rates from 0.5–100 s^-1.

All three use positive uniform weights. Therefore nominal mode count L=16,
weight entropy H_mem=ln(16), and GGFE D_eff=L exp(H_mem)=256 are identical;
the intervention is temporal rate geometry and its derived descriptors such as
M_scale.

## Controlled protocol

The grid is:

- BPSK, QPSK, 16-QAM;
- five fixed colored-noise environments;
- SNR = 0, 10, 20, 30 dB;
- five seeds;
- three representation geometries.

This gives 900 underlying task cases. Each case reuses the same noisy
y_train/y_test realization for all three geometry conditions.

The held-out 512-symbol target sequence is not used to fit the kernel, PCA
basis, or readout.

## Two representation paths

### Full MIN/SOE state

Each geometry produces the 16 modal state q=(q_1,...,q_16). Complex PCA
subspaces of dimension d=1,...,16 are fit on training-symbol centers and
evaluated on held-out symbols.

The operational task dimension is

    D_task = min{d : NMSE_d <= 1.10 * NMSE_full}

for that geometry and task case. This is the same empirical definition used in
14A.

### Scalar MIN output

The scalar projection

    z = sum_j w_j q_j

is decoded by the same complex linear readout. It is recorded as a secondary
representation path because 14B showed that scalar projection can expose
different task behavior from the full state.

## Structural measurements

Recorded alongside task NMSE:

- D_task and PCA explained energy at D_task;
- center-state participation and entropy dimensions;
- M_scale, M_cap, H_mem and D_eff;
- kernel t50/t90.

## Scientific boundary

14C is not a geometry leaderboard. It asks whether controlled changes in
temporal rate geometry alter task-relevant structure under otherwise matched
conditions.

The design intentionally keeps L and D_eff constant so that a difference in
D_task or full-state task NMSE cannot be explained by nominal mode count or the
GGFE entropy-effective count alone.

The full-state PCA results are the primary geometry diagnostic. Scalar-MIN
results are secondary because they combine geometry with one-dimensional
projection/compression.

The experiment remains task-specific. It does not establish a universal
optimal kernel geometry, and D_task is not a physical state dimension or an
information-theoretic measure.

## Relation to the program

13A established controlled kernel geometries.

13B connected kernel geometry to SOE basis and signal-driven state structure.

13C tested environment-to-kernel construction and dictionary mismatch.

14A tested whether an environment-informed full state contains task-relevant
digital-symbol information.

14B tested environment/kernel alignment using scalar MIN.

14C now tests whether temporal rate geometry itself changes the task-relevant
structure when the observation is held fixed.

## Executed result

The authoritative pull-request CI execution completed successfully:

- GitHub Actions Experiment 14C run: 36209760102
- repository CI run: 36209760100
- full Python-version test matrix: 36209760135
- authoritative head: 11ddbf56c60c71c93397e66e8a9480655d642091
- merged to main as commit: 3d7fb804f47ddf1c2f305de17403cd48039dc123
- result artifact: experiment-14c-results
- artifact SHA-256: 83dd70619c1ecc5dbe518a0c9b65eb4cc64a5b9550c3d93d1a261f18f413f019

The run generated 300 underlying task cases, 14,400 full-state PCA rows, and
900 scalar-MIN rows. The repository-equivalence self-check between the
accelerated experiment path and SOEMemory.state_trajectory was 1.61e-11
relative error.

### Aggregate task results

| Geometry | Mean full-state NMSE | Mean D_task | Mean scalar-MIN NMSE | Mean center-state participation |
|---|---:|---:|---:|---:|
| clustered | 0.8574 | 2.04 | 0.9669 | 1.0004 |
| logspread | 0.5003 | 10.20 | 0.9794 | 1.1605 |
| wide | 0.7156 | 11.70 | 0.9991 | 1.2457 |

Because the raw observation is an unchanged control, scalar-MIN did not beat raw
center samples in any of the 900 paired geometry cases.

### SNR interaction

Mean full-state NMSE by SNR was:

| SNR | clustered | logspread | wide |
|---:|---:|---:|---:|
| 0 dB | 0.9442 | 0.7170 | 1.4215 |
| 10 dB | 0.8494 | 0.4982 | 1.3190 |
| 20 dB | 0.8203 | 0.4001 | 0.0970 |
| 30 dB | 0.8158 | 0.3859 | 0.0248 |

The wide geometry therefore shows a strong task-condition interaction: its
full-state readout is substantially worse at 0–10 dB but becomes very low-NMSE
at 20–30 dB. The wide condition also has the largest variance, driven by a
small number of low-SNR outliers. These should not be converted into a general
claim that wide geometry is superior or inferior.

The scalar path is much less geometry-sensitive: the paired mean NMSE changes
were approximately +0.0125 for logspread relative to clustered and +0.0322 for
wide relative to clustered. This is consistent with scalar projection masking
much of the geometry-dependent state structure.

### Task-dimensionality result

D_task changes markedly even though L and D_eff are identical:

- clustered: mean 2.04, median 2;
- logspread: mean 10.20, median 11;
- wide: mean 11.70, median 14.

The associated interpretation is not that clustered is intrinsically more
useful. Its nearly rank-one state reaches its own full-state performance with
very few dimensions, while that full-state performance remains substantially
higher in NMSE than the logspread state. Thus 14C separates representation
compression from task accuracy.

This is an important extension of 13B: temporal rate geometry can change both
the amount of usable task performance and the number of state dimensions needed
to expose that performance, without changing nominal mode count or D_eff.

### Paired geometry comparisons

Across the 300 paired task cases:

- logspread minus clustered full-state NMSE: mean -0.3571, median -0.4041;
  logspread was lower in 296/300 cases.
- wide minus clustered: mean -0.1419, median -0.7298; wide was lower in
  260/300 cases, but a small set of large low-SNR outliers raises the mean.
- wide minus logspread: mean +0.2153, median -0.3323; wide was lower in
  245/300 cases, while logspread was lower in 55/300.

The disagreement between mean and median for wide-versus-logspread is itself a
useful result: geometry effects are not well summarized by a single aggregate
score and depend strongly on task condition.

## Interpretation

14C provides the first controlled task-level evidence in this series that
temporal rate geometry is not a neutral implementation detail. With the same
16 modes, the same uniform weights, the same D_eff, the same observed
waveforms, the same decoder, and the same held-out protocol, changing only the
rate geometry substantially changes the full-state task representation.

The strongest pattern is a separation between:

    clustered → low-dimensional but weak task exposure
    logspread → moderate-dimensional and consistently useful task exposure
    wide → high-dimensional and strongly condition-dependent task exposure

This should be read as a task-specific empirical observation, not a universal
geometry ranking.

The scalar MIN result reinforces the distinction between the state
representation and the one-dimensional MIN projection: much of the
geometry-dependent structure visible in the full state is lost when it is
collapsed to z=w^Tq.

## Limitations and next step

The readout is a fixed complex linear model, the environment families are the
five controlled synthetic cases inherited from 14A/14B, and the SOE rate
families are predetermined rather than learned.

The wide-geometry low-SNR outliers also motivate explicitly recording readout
conditioning and testing regularization sensitivity before drawing stronger
mechanistic conclusions.

The next useful step is therefore not a geometry leaderboard. It is to use
14C's paired geometry results to design the next task-level control, for example
prediction or environment identification, while retaining the same separation
between L, D_eff, D_state and D_task.
