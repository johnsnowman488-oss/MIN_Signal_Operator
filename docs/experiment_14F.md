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


## Executed result

The authoritative Experiment 14F CI execution completed successfully:

- workflow run: 36335831319
- experiment head: 6bddd6540d93afa59db426f39ba2c29a869ba93a
- artifact: experiment-14f-results (artifact ID 10937361017)
- artifact SHA-256: ca8ce10f7901478ea5a7b61581f98b78b463c9d44ed1042a2acd9eac85607faf
- targeted/regression tests: 13 passed
- result rows: 4,020 (2,700 full-state, 540 scalar-MIN, 720 finite-history, 60 oracle)

At d=16 and lambda=10^-6, the full MIN state showed substantial dependence
on the chosen MIN rate geometry. Averaged over seeds and SNRs:

| Source process | MIN clustered | MIN logspread | MIN wide |
|---|---:|---:|---:|
| clustered | 0.3423 | 0.1428 | 0.0833 |
| logspread | 0.3295 | 0.1486 | 0.0939 |
| wide | 0.4006 | 0.2303 | 0.1540 |

Within this experiment the matched source/MIN geometry cells were not
systematically separated from mismatched cells. At the same d=16, lambda=10^-6
setting, the mean matched NMSE was 0.2150 versus 0.2134 for mismatched cells.
This means the present result should not be interpreted as evidence that
geometric matching alone determines predictive accessibility.

A more informative paired comparison is the best MIN geometry available for
each physical source/SNR/seed case. Its aggregate mean NMSE was 0.1101, versus
0.1639 for the current-observation baseline. The best explicit finite-history
predictor was lower at 0.0543, while the oracle true-hidden-state predictor was
0.0182.

The improvement pattern was strongly SNR-dependent. At 0 dB, the best MIN
representation reduced mean NMSE from about 0.512–0.495 to 0.125–0.198 across
the three source geometries. At 20–30 dB, the current observation itself was
already very predictive and MIN often introduced smoothing error; for example,
for the clustered source at 30 dB the current-observation mean was 0.0131 while
the best MIN mean was 0.0672.

The finite-history control also showed that predictive information can be
accessed explicitly from ordinary observation history. Across the 60 physical
source/SNR/seed cases, the best history length was 4 in 25 cases, 16 in 19
cases, and 64 in 16 cases.

The predictive dimension curves for the wide MIN geometry show a clear
compression/accessibility effect: at lambda=10^-6, mean NMSE for source
clustered/logspread/wide was respectively 0.8891/0.8873/0.8957 at d=1,
0.6648/0.6034/0.6566 at d=2, 0.2931/0.3282/0.4300 at d=4,
0.0949/0.1077/0.1721 at d=8, and 0.0833/0.0939/0.1540 at d=16.

### Interpretation

14F provides evidence that the controlled hidden-memory process contains
predictive information beyond the instantaneous observation, and that a
causal MIN/SOE state can expose part of that information. The result is
strongest at lower SNR, where the raw observation is noisier.

However, the finite-history baseline performs better on this particular
linear process, and the true hidden-state oracle remains substantially better
than any observable MIN representation. Therefore 14F does **not** establish
that MIN is a unique or optimal Markovization mechanism.

The experiment instead establishes a narrower and useful result:

    hidden temporal dependence → predictive gain from causal memory access

and shows that MIN can participate in that chain, with performance depending
on state geometry, retained dimension, and observation SNR.

The lack of a systematic matched-geometry advantage is also informative. In
the current construction, broad rate support can capture predictive structure
more effectively than simply matching the generating geometry. A future
experiment should therefore distinguish memory-support coverage from exact
kernel matching rather than treating them as the same hypothesis.

14F closes the controlled test of whether the temporal-state idea has a real
predictive target. A subsequent experiment can now test the separate postulate
under a different condition instead of reusing the same observable prediction
task.
