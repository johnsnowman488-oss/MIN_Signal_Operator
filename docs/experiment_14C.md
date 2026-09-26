# Experiment 14C — Geometry → Task Relevance

## Aim

14C isolates the geometry-to-task link:

    kernel rate geometry → MIN/SOE state structure → digital-symbol recovery

The task environment is held fixed within each paired case. The representation
kernel is varied across the three rate geometries established in 13A/13B:

- **clustered**: 16 rates from 9.51–10.51 s^-1;
- **logspread**: 16 rates from 2–30 s^-1;
- **wide**: 16 rates from 0.5–100 s^-1.

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

14C now tests whether **temporal rate geometry itself changes the task-relevant
structure when the observation is held fixed**.
