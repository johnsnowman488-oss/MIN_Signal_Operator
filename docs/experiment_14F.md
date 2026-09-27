# Experiment 14F — Hidden-Memory Markovization Test

## Aim

14F directly tests the stronger temporal-state postulate behind the MIN/GFE
program:

> A process whose observed output is non-Markovian may admit a more predictive
> finite-dimensional representation when the relevant hidden memory is made
> explicit.

The experiment therefore constructs a process with a known Markov hidden state
and a generally non-Markovian scalar observation.

The controlled chain is:

    hidden exponential modes → scalar observed process
                              ↓
                         noisy observation
                              ↓
                     predictor representation

The primary question is whether a causal MIN/SOE state makes future output
prediction more accessible than the current observation alone.

## Controlled latent process

For each geometry, 16 latent modes follow

    q_j(n+1) = a_j q_j(n) + sqrt(1-a_j^2) ε_j(n+1)

with

    a_j = exp(-γ_j Δt).

The clean observed scalar process is

    x_n = (1/16) Σ_j q_j(n).

The measurement is

    y_n = x_n + η_n,

where η_n is independent white measurement noise at the specified SNR.

The three source geometries use exactly the 14C rate families:

- clustered;
- logspread;
- wide.

The hidden modal vector q_n is Markov by construction, while the scalar mixture
x_n generally requires memory/history to predict the next value.

## Predictive task

At index n, all predictors may use samples through y_n only. The target is the
clean process value x_{n+1}.

This gives a useful decomposition:

- current-observation predictor: only y_n;
- finite-history predictors: 1, 4, 16, or 64 past observations;
- causal MIN/SOE state: full state with PCA dimensions 1, 2, 4, 8, 16;
- oracle hidden state: the true q_n, included only as a reference for predictive
  headroom.

The future target is never used in state construction, PCA fitting, or readout
fitting.

## Readout and controls

All observable predictors use the same stable linear complex ridge/SVD readout
family as 14D/14E.

Ridge values are:

    λ = 0, 10^-8, 10^-6

with λ normalized by the augmented training Gram scale.

The same clean process and noisy observation are reused across all MIN
geometries for a fixed source geometry/SNR/seed. A MIN geometry is called
**matched** when its 16 decay rates equal the source process geometry.

## Measurements

Primary metric:

    NMSE_1 = E|x̂_{n+1} - x_{n+1}|² / E|x_{n+1}|².

The main causal accessibility comparison is against the current-observation
baseline.

Additional diagnostics include:

- gap from the true hidden-state oracle;
- performance of explicit finite history;
- predictive task dimension D_task,pred;
- readout conditioning;
- matched versus mismatched source/MIN geometry.

D_task,pred remains an operational representation measure. It is not a physical
state dimension and is not identified with D_eff, D_basis, or D_state.

## Data accounting

There are 60 source/SNR/seed cases and 3 MIN geometries, giving:

- 180 source × MIN cases;
- 2,700 full-state MIN rows;
- 540 scalar-MIN rows;
- 720 finite-history rows (shared across MIN geometries);
- 60 oracle rows (shared across MIN geometries);
- 4,020 total rows.

## Interpretation

Several outcomes are informative.

If MIN materially improves on the current observation and approaches the hidden
state oracle, this supports the claim that a causal memory representation can
make non-Markovian predictive information more accessible.

If finite history reaches similar performance with fewer assumptions, that shows
the effect can be explained by explicit history rather than the particular MIN
construction.

If matched MIN geometries outperform mismatched ones, that links predictive
accessibility to alignment between represented memory timescales and the source
process.

If MIN does not improve on current observation or finite history, the stronger
Markovization claim is constrained for this controlled construction.

No universal predictor or geometry ranking is asserted.
