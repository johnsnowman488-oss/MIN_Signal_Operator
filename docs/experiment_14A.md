# Experiment 14A — Task-Relevant MIN Subspace

## Aim

Experiment 14A is the first task-level evaluation in the MIN Signal Operator
program. It tests whether an environment-informed MIN/SOE state contains
recoverable information about an underlying digital symbol sequence and how
much of that state a simple linear readout actually requires.

The experiment is deliberately diagnostic rather than competitive. It does
not ask whether MIN is universally superior to conventional methods.

## Protocol

```
digital symbols
      ↓
colored environment noise
      ↓
observed waveform
      ↓
oracle environment-derived SOE kernel
      ↓
16-dimensional MIN/SOE state
      ↓
complex PCA
      ↓
linear symbol readout
      ↓
held-out symbol NMSE
```

The environment families and positive 16-rate SOE dictionary are inherited
from the 13C program. The environment covariance is oracle-known in 14A so
that kernel-estimation uncertainty does not confound the first task-level
measurement.

## Grid

- BPSK, QPSK, 16-QAM
- white-limit, short-memory, multiscale, power-law, squared-exponential environments
- 0, 10, 20, 30 dB SNR
- 5 seeds
- 512 training symbols + 512 held-out symbols
- 16 retained-state dimensions tested

This gives 300 task cases and 4,800 PCA/readout rows.

## Primary measurement

The primary metric is held-out normalized symbol reconstruction MSE:

```
NMSE = mean(|y_hat - y|²) / mean(|y|²)
```

No BER, EVM, neural network, or task-specific nonlinear optimizer is used.

For every case, PCA and the linear readout are fitted only on the training
symbols. The held-out sequence is used only for final evaluation.

## Task-relevant dimension

Define:

```
D_task(10%) =
  smallest k such that
  NMSE(k) <= 1.10 × NMSE(full state)
```

This is an operational task-specific quantity. It is not asserted to equal
any structural dimension.

The experiment therefore preserves:

```
L ≠ D_eff ≠ D_basis ≠ D_state ≠ D_task
```

## Controls

A raw center-sample linear readout is included as a simple observation-space
control.

The primary structural comparison is the held-out NMSE curve as retained PCA
dimension increases from 1 to the full 16-state representation.

## Interpretation boundary

A lower NMSE for a retained state subspace would establish only that the
tested representation contains information useful to this specified linear
symbol-reconstruction task under the tested environment/SNR conditions.

It does not by itself establish superiority of MIN, optimality of the kernel,
or general task universality.

## Outputs

- `experiments/results/14A_task_relevant_min_subspace_results.csv`
- `experiments/results/14A_task_relevant_min_subspace_cases.csv`
- `experiments/results/14A_task_relevant_min_subspace_summary.json`
- GitHub Actions artifact: `experiment-14a-results`
