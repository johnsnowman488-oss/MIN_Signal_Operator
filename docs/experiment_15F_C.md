# Experiment 15F-C — Communications Budget Scaling

## Purpose

15F-C measures how receiver performance changes as the available temporal-state budget increases through the actual 15E budgets:

N = 1, 2, 4, 8, 16

The question is not whether MIN is universally superior. It is whether the state-efficiency behavior observed in 15E transfers into a controlled communications receiver when every representation receives the same number of state variables.

## Lineage

- **15C:** matched-timescale MIN-16 versus one-pole control.
- **15D:** equal-state comparison of MIN, logarithmic IIR, dense coordinate control, and FIR.
- **15E:** actual state budgets 1, 2, 4, 8, 16.
- **15F-A:** first communications transfer.
- **15F-B:** fixed-N channel stress.
- **15F-C:** explicit communications budget scaling.

The temporal representation constructors are imported directly from `15E_state_budget_sweep.py`.

## Representations

For each budget:

- `min_N`
- `iir_logspread_N`
- `dense_ss_N`
- `fir_N`

A `current_1` observation-only baseline is recorded once per condition. It is not part of the four representation budget curves.

The dense-state representation remains a coordinate control: it is an orthogonal similarity transform of the same logarithmic IIR pole bank.

## Controlled communications environment

15F-C removes the extra stress axes from 15F-B so that budget is the central varying factor.

The grid uses:

- BPSK and QPSK.
- The four 15E process families.
- 15E-compatible scale factors: 4, 12, 32.
- SNR: 0, 10, 20 dB.
- The complete 15E seed set.

The transmitted rectangular-pulse complex-baseband waveform is passed through one causal temporal channel, then AWGN is added. The resulting noisy stream is identical for every representation and every budget within a case.

## Receiver protocol

The same linear readout is used for every representation. It is fitted only on the first 512 transmitted symbols. The final 512 symbols are held out for BER, SER, EVM, and NMSE.

No nonlinear receiver, adaptive kernel fitting, or test-set tuning is used.

## Primary outputs

### Case-level results

One row is emitted for each modulation, process, scale factor, SNR, seed, representation, and state budget. Metrics are BER, SER, EVM, and NMSE.

### Paired representation deltas

For each budget, MIN is paired against logarithmic IIR, dense coordinate control, and FIR. The deltas are `MIN - control` so positive values indicate higher error for MIN.

### Budget-transfer deltas

For each representation the following transitions are recorded: 1 -> 2, 2 -> 4, 4 -> 8, and 8 -> 16.

For an error metric, `improvement = metric_at_N_from - metric_at_N_to`. Positive values therefore indicate lower error after adding states.

## Interpretation discipline

15F-C can distinguish several empirical patterns without assigning a universal winner:

- Early improvement concentrated at small N would indicate that additional state capacity is useful at low budgets.
- Improvement delayed until larger N would indicate that the tested receiver needs more temporal capacity before the representation becomes effective.
- MIN/IIR convergence would weaken claims that the particular MIN pole placement matters for the tested task.
- MIN/IIR separation together with IIR/dense agreement would point toward differences in temporal construction rather than arbitrary state coordinates.
- FIR remaining different from the exponential-bank representations would indicate that finite recent history and distributed temporal scales are not interchangeable at equal state count.

## CI authority

CI runs the tests, executes the complete experiment, and uploads the CSV/JSON outputs. Numerical claims about 15F-C should be made only from the retrieved CI artifact for the successful run.
