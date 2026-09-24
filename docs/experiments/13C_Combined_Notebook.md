# 13C Combined Research Notebook

The repository now has one notebook-facing synthesis layer for 13C-1, 13C-2 and 13C-3. It loads the authoritative experiment CSV artifacts, produces focused plots, and can regenerate all three CSV result sets with a single RUN_EXPERIMENTS switch.

The notebook intentionally does not duplicate experiment mathematics. The scripts remain the numerical source of truth.

Plot set:

1. 13C-1 environment temporal scale versus fitted M_scale.
2. 13C-1 weighted basis versus weighted MIN state participation.
3. 13C-2 state representation error versus observation length and SNR.
4. 13C-2 kernel error versus observation length and SNR.
5. 13C-3 dictionary/rate-support mismatch versus state error.
6. 13C-3 kernel fidelity versus representation fidelity.

The notebook also exports experiments/results/13C_combined_notebook_summary.csv.

The dimension hierarchy remains L != D_eff != D_basis != D_state, with D_task reserved for a later task-relevance stage.