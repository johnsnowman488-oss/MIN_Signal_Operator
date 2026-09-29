# Experiment 15E — Actual temporal-state budget sweep

## Question

When the representation itself is constrained to an actual temporal-state budget, how does MIN's process-adapted memory allocation compare with conventional exponential and finite-history representations?

## Budget

The primary sweep uses **1, 2, 4, 8, and 16 temporal states**.

Each representation has exactly the stated number of stored state values. The direct full-state readout is the primary performance measurement; a compact PCA diagnostic is retained only as a secondary view.

## Representations

- **MIN-N:** process-adapted positive exponential modes. For multi-timescale processes, states are allocated deterministically from the process mixture weights, with each timescale represented whenever the budget permits.
- **IIR-logspread-N:** N first-order exponential states spanning the same minimum-to-maximum rate interval as the corresponding MIN construction.
- **Dense-state-N:** orthogonal similarity transform of the IIR-logspread-N state, serving as a coordinate control.
- **FIR-N:** N-sample finite-history delay line.

The same observation, train/test split, task targets, process grid, SNR grid, scale factors, seeds, and linear readout are used for all representations.

## Primary outputs

The CI artifact contains:

- `15E_state_budget_sweep_cases.csv` — one direct full-state result per matched case.
- `15E_state_budget_sweep_paired_deltas.csv` — matched MIN-minus-IIR, MIN-minus-dense, MIN-minus-FIR, and IIR-minus-dense differences.
- `15E_state_budget_sweep_marginal_state_value.csv` — NMSE reduction when increasing the budget 1→2, 2→4, 4→8, and 8→16.
- `15E_state_budget_sweep_summary.csv` — grouped means, spread, effective task dimension diagnostic, and conditioning.
- `15E_state_budget_sweep_pca_results.csv` — secondary PCA diagnostics up to each actual budget.
- `15E_state_budget_sweep_summary.json` — protocol and integrity metadata.

## Interpretation boundary

15E tests **performance versus actual temporal-state budget**. A MIN advantage at small budgets would support efficient allocation of temporal state capacity. Similar performance would indicate that the 15D PCA result does not translate into an intrinsic low-state-budget advantage.

The dense-state control should agree with its diagonal IIR counterpart up to numerical noise. A material discrepancy would point to an implementation issue rather than a scientific effect.

15E does not establish universal optimality, universal computational superiority, or superiority over every possible state-space construction.
