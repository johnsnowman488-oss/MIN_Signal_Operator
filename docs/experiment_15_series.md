# Experiment 15 Series — Reproducibility Chain

This page is the provenance index for the current **15A–15E** MIN representation branch. It is intentionally a map, not a replacement for the experiment-specific protocols and analyses.

## Reproducibility rule

For published numerical claims, the authoritative sequence is:

1. experiment implementation at the recorded commit;
2. dedicated CI workflow and tests;
3. CI-produced artifact;
4. experiment-specific analysis/protocol document;
5. claim limited to the controls actually tested.

Local reruns are useful for development, but should not replace the recorded CI artifact when reporting a result.

## Chain

| Experiment | Question | Implementation | Test | Workflow | Documentation | CI/artifact record |
|---|---|---|---|---|---|---|
| **15A** | Does memory-rate geometry change task-relevant representation structure across tasks? | `experiments/15A_geometry_task_generalization.py` | `tests/test_experiment_15a.py` | `.github/workflows/experiment-15a.yml` | `docs/experiment_15A_analysis.md` | `experiment-15a-results`; 300 paired cases |
| **15B** | Does the geometry effect interact with temporal process structure? | `experiments/15B_process_geometry_alignment.py` | `tests/test_experiment_15b.py` | `.github/workflows/experiment-15b.yml` | `docs/experiment_15B_analysis.md` | `experiment-15b-results`; 4 processes × 2 tasks × 3 geometries × 4 SNR × 5 seeds |
| **15B-2** | Does explicit process-specific rate alignment matter? | `experiments/15B2_process_specific_alignment.py` | `tests/test_experiment_15b2.py` | `.github/workflows/experiment-15b2.yml` | `docs/experiment_15B2_analysis.md` | `experiment-15b2-results`; corrected run uses 16 unique rates |
| **15B-3** | Where is the task-useful memory timescale relative to the process scale? | `experiments/15B3_task_optimal_timescale_sweep.py` | `tests/test_experiment_15b3.py` | `.github/workflows/experiment-15b3.yml` | `docs/experiment_15B3_analysis.md` | Run `36435852512`; `experiment-15b3-results`; 1,120 cases / 17,920 PCA rows |
| **15C** | Does the multi-mode result persist against a matched one-pole IIR control? | `experiments/15C_extended_timescale_iir_control.py` | `tests/test_experiment_15c.py` | `.github/workflows/experiment-15c.yml` | `docs/experiment_15C_analysis.md` | Run `36438562042`; `experiment-15c-results`; 2,880 cases / 46,080 PCA rows |
| **15D** | Does representation geometry matter at an equal 16-state budget? | `experiments/15D_equal_budget_comparison.py` | `tests/test_experiment_15d.py` | `.github/workflows/experiment-15d.yml` | `docs/experiment_15D.md` | Run `36486139461`; 3,840 cases / 61,440 PCA rows; artifact SHA-256 `5b6a17cf891e431a9ef566ea9b8402ac84e6c7dd5c32624432bbec7f99f77f1b` |
| **15E** | Does the conclusion survive when the actual temporal-state budget is swept? | `experiments/15E_state_budget_sweep.py` | `tests/test_experiment_15e.py` | `.github/workflows/experiment-15e.yml` | `docs/experiment_15E.md` | Run `36512822394`; artifact ID `11009583130`; SHA-256 `db3f1377d3e13cbd04da714aa61d8ec776539725747a953c64fc83b4ba27b702`; 19,200 cases / 119,040 PCA rows |

The shared numerical layer is **`experiments/15_utils.py`**. The 15-series experiments independently use this utility layer and validate its SOE-state equivalence against the repository implementation.

## Claim boundaries

### 15A
Supports the narrower statement that changing memory-rate geometry can change realized state structure and task-relevant dimension across multiple tasks. It does **not** establish that any geometry universally improves task performance.

### 15B / 15B-2 / 15B-3
These experiments establish increasingly controlled evidence that useful memory scale is task/process dependent. In particular, 15B-2 contains a corrected single-scale construction after duplicated-rate design error was detected. 15B-3 shows that the useful scale can lie substantially faster than the process characteristic scale, while also exposing a bandwidth/locality confound.

### 15C
Adds the matched one-pole control. Its strongest supported claim is that, in the tested grid, a 16-mode MIN representation can outperform a single exponential state when the one-pole control is matched by effective t50. It is **not** an equal-compute or equal-state comparison.

### 15D
Moves to an equal 16-state comparison among MIN, conventional log-spread IIR, dense-state coordinate control, and FIR. The dense-state control is essential: its agreement with the diagonal IIR representation means coordinate choice is not the intended explanation.

The evidence remains task-dependent. It should not be summarized as universal MIN superiority over all 16-state temporal representations.

### 15E
Makes actual temporal-state budget the independent variable: 1, 2, 4, 8, 16 states. The direct full-state readout is primary; PCA is secondary. The cleanest discriminatory region is around 4 states, while high-state conditioning and non-monotonic 16-state curves limit stronger claims.

## Artifact contents

The CI workflows write numerical outputs under `experiments/results/` and upload them as workflow artifacts. Generated result CSV/JSON files are **not required to be committed to Git**; the CI artifact is the recorded numerical output for each execution.

For 15E the expected outputs are:

- `15E_state_budget_sweep_cases.csv`
- `15E_state_budget_sweep_paired_deltas.csv`
- `15E_state_budget_sweep_marginal_state_value.csv`
- `15E_state_budget_sweep_summary.csv`
- `15E_state_budget_sweep_pca_results.csv`
- `15E_state_budget_sweep_summary.json`

The same naming/provenance convention is used by the earlier 15-series workflows.

## What remains to be cleaned later

This pass deliberately does **not** rewrite historical experiment analyses or regenerate numerical artifacts. Before publication, the paper-facing archive should additionally record the exact CI run/artifact identifiers for 15A, 15B, and 15B-2 where those identifiers are not already present in their analysis documents.

That is documentation hygiene, not a numerical gap.
