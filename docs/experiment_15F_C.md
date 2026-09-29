# Experiment 15F-C — Communications Budget Scaling

## Status

15F-C completed successfully in CI.

- Run: `36572380465`
- Commit: `9a1dcef2d8bd937c6576cecf03afe693dd812363`
- Artifact: `experiment-15f-c-results`
- Artifact ID: `11034756731`
- Artifact SHA-256: `9905fb43b6128aea144b59ae0b34eb0abe7d8c3a1b65a47ca71e828df0732a7b`
- Repository SOE equivalence relative error: `1.4322306863936366e-11`

## Purpose

15F-C formalizes the communications budget-scaling analysis through the actual 15E state budgets:

`N = 1, 2, 4, 8, 16`

The central question is how receiver error changes as temporal-state capacity increases, under the same noisy complex-IQ stream and the same linear readout discipline.

15F-C is a **closure/analysis stage rather than a new environmental condition set**. The four representation families use the same communications grid already executed in 15F-A. The main new output is an explicit marginal budget-delta table for every family and every doubling transition.

## Lineage

- **15C:** matched-timescale MIN-16 versus one-pole control.
- **15D:** equal-state comparison of MIN, logarithmic IIR, dense coordinate control, and FIR.
- **15E:** actual state budgets 1, 2, 4, 8, 16.
- **15F-A:** first communications transfer across those budgets.
- **15F-B:** fixed-N=16 channel stress.
- **15F-C:** budget-scaling closure and marginal-state analysis.

The temporal representation constructors are imported directly from `15E_state_budget_sweep.py`.

## Representations

For each budget:

- `min_N`
- `iir_logspread_N`
- `dense_ss_N`
- `fir_N`

A `current_1` observation baseline is recorded once per condition. In 15F-C it uses the same one-step linear-readout discipline as the representation rows; it is not part of the four budget curves.

The dense-state representation remains the coordinate control: it is an orthogonal similarity transform of the same logarithmic IIR pole bank.

## Controlled communications environment

The grid uses:

- BPSK and QPSK.
- The four 15E process families: short, multiscale, long, hidden_mix.
- Scale factors 4, 12, 32.
- SNR 0, 10, 20 dB.
- The complete 15E seed set: 0, 1, 2, 3, 4.

The transmitted rectangular-pulse complex-baseband waveform is passed through one causal temporal channel, then AWGN is added. The same noisy stream is reused by every representation and every budget within a case.

The complete artifact contains:

- 7,560 case rows;
- 1,512 summary rows;
- 5,400 paired representation-delta rows;
- 5,760 budget-delta rows.

## Receiver protocol

The linear readout is fitted only on the first 512 transmitted symbols. The final 512 symbols are held out for BER, SER, EVM, and NMSE.

No nonlinear receiver, adaptive kernel fitting, or test-set tuning is used. States are causal.

## Aggregate CI results

Mean error across the full 360-case condition grid for each representation/budget is:

| Budget | MIN BER | IIR/Dense BER | FIR BER |
|---:|---:|---:|---:|
| 1 | 0.238504 | 0.238504 | 0.154115 |
| 2 | 0.212093 | 0.212093 | 0.131934 |
| 4 | 0.152591 | 0.166349 | 0.103049 |
| 8 | 0.121354 | 0.122786 | 0.064686 |
| 16 | 0.120763 | 0.120163 | 0.040275 |

For MIN, mean EVM was 68.0726%, 61.7243%, 49.9105%, 44.0021%, and 43.6168% at N=1,2,4,8,16 respectively. Mean NMSE was 0.556012, 0.498249, 0.369773, 0.310617, and 0.308271.

The current-observation baseline averaged BER 0.154115, EVM 60.6241%, and NMSE 0.437765 in the 15F-C protocol. The one-state FIR is the same current-sample representation, so its aggregate values coincide with this baseline.

## Marginal state value

For error metrics, the recorded budget improvement is:

`improvement = metric_at_N_from - metric_at_N_to`

Positive values therefore mean that the larger budget reduced error.

For MIN, the mean BER improvements were:

| Transition | ΔBER |
|---|---:|
| 1 -> 2 | +0.026411 |
| 2 -> 4 | +0.059502 |
| 4 -> 8 | +0.031236 |
| 8 -> 16 | +0.000591 |

The largest MIN marginal BER gain occurred at 2 -> 4. The gain from 8 -> 16 was small.

The corresponding IIR/dense mean BER improvements were +0.026411, +0.045744, +0.043563, and +0.002623. FIR continued to show sizeable gains through 8 -> 16: +0.022182, +0.028885, +0.038363, and +0.024411.

Thus the budget curves have different shapes: MIN concentrates more of its aggregate gain in the middle of the sweep, IIR gains more strongly from 4 -> 8 and 8 -> 16, and FIR continues to gain at high state count.

## MIN versus IIR

At N=1 and N=2, MIN and log-spread IIR are identical because the tested one-state/two-state constructions collapse to the same rate choices in the relevant process families.

The paired mean MIN-minus-IIR BER deltas were:

| Budget | Mean ΔBER (MIN - IIR) |
|---:|---:|
| 1 | 0.000000 |
| 2 | 0.000000 |
| 4 | -0.013759 |
| 8 | -0.001432 |
| 16 | +0.000600 |

At N=4, MIN's aggregate BER was about 8.27% lower than IIR's. At N=8, the difference was about 1.17% lower. At N=16, IIR was about 0.50% lower than MIN.

Short and long are single-timescale processes, so their MIN and log-spread rate sets coincide. The nontrivial separation comes from multiscale and hidden_mix.

The dense control tracked IIR to numerical precision. Across all cases, the maximum absolute IIR-versus-dense BER and SER differences were zero; the maximum EVM difference was about 4.0e-7 percentage points and the maximum NMSE difference about 2.3e-9.

## Relationship to 15F-A

Because 15F-A already used the same BPSK/QPSK, process, scale, SNR, seed, and state-budget grid for the four primary representations, the 15F-C representation rows are a controlled reproduction of that existing dataset rather than a statistically independent condition.

Direct artifact comparison found:

- 7,200 non-baseline representation rows matched.
- BER and SER matched exactly.
- Maximum absolute EVM difference was about 6.5e-7 percentage points.
- Maximum absolute NMSE difference was about 3.5e-9.

The current-observation baseline was intentionally changed in 15F-C to use the same linear-readout protocol as the state representations, so its values are not expected to match the raw-observation baseline in 15F-A.

The scientific contribution of 15F-C is therefore the explicit budget-transition analysis, not another independent communications dataset.

## Interpretation boundaries

15F-C supports the following narrower observations for this synthetic receiver:

1. Increasing state budget from 1 to 16 generally reduced error for all four tested representation families.
2. MIN showed its strongest aggregate marginal BER improvement at 2 -> 4 and little additional aggregate BER improvement at 8 -> 16.
3. MIN was below log-spread IIR in aggregate BER at N=4 and N=8, while the ordering reversed slightly at N=16.
4. IIR and its dense coordinate control remained effectively identical, supporting the interpretation that their results are determined by temporal construction rather than arbitrary state coordinates.
5. FIR had substantially lower aggregate BER than the exponential-bank families in this specific synthetic communications setup.

These are task- and protocol-specific results. They do not establish universal communications superiority or real-IQ receiver performance.

## CI authority

The successful CI artifact is the numerical record for 15F-C. Local reruns should be treated as development checks, not as replacements for the recorded artifact.
