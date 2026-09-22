# Experiment 13B — Geometry → State Structure

## Objective

13B tests the bridge

    GFE/GGFE kernel geometry → SOE temporal-basis geometry → MIN state geometry

The experiment remains task-agnostic: no channel, noise, receiver, classifier,
predictor, or optimization objective is introduced.

## Controlled atlas

The 13A-1 grid is retained unchanged:

- BPSK, QPSK, 16-QAM
- SOE mode counts L = 1, 2, 4, 8, 16
- clustered, logspread, and wide decay-rate geometries
- uniform, slow-dominant, and fast-dominant positive weights
- five seeds

This produces 675 rows.

## GFE/GGFE descriptors

The same descriptors are carried forward:

M_cap = sum(w_i/gamma_i^2) / sum(w_i/gamma_i)

M_scale = log10(gamma_max/gamma_min)

M_res = L/M_scale

H_mem = -sum(w_tilde_i log(w_tilde_i))

and the documented GGFE quantity

D_eff = L exp(H_mem).

No spectral/fractal dimension d_s is assigned to this finite positive-SOE atlas.

## Intermediate SOE basis geometry

For phi_i(t) = exp(-gamma_i t), 13B evaluates the finite-horizon Gram matrix

G_ij = (1 - exp(-(gamma_i + gamma_j)T)) / (gamma_i + gamma_j).

Its spectrum provides basis participation dimension, entropy dimension,
90%/99% ranks, effective rank at 1e-12, condition number, maximum normalized
coherence, and extremal eigenvalues.

This measures how the finite-horizon exponential basis is represented in the
chosen L2 geometry.

## Empirical MIN state geometry

The sampled causal states satisfy

q_i[n] = exp(-gamma_i dt) q_i[n-1]
       + (1-exp(-gamma_i dt))/gamma_i * x[n].

The covariance spectrum provides state participation dimension, rank,
conditioning, and the entropy analogue

H_state = -sum(p_i log p_i)
D_state,entropy = exp(H_state).

A weighted-state covariance is also recorded so that kernel amplitude weighting
can be distinguished from the unweighted mode-state geometry.

Thus D_eff is compared with signal-driven state quantities without assuming
that the two are equal.

## Results

The 675-row CI execution completed successfully on 2026-09-22. Repository tests,
the 13B runner, and artifact upload all passed.

### Main relationships

| Relationship | Pearson r | Spearman rho |
|---|---:|---:|
| M_scale → basis participation | 0.765 | 0.874 |
| M_scale → basis entropy dimension | 0.776 | 0.874 |
| M_scale → basis coherence | 0.170 | 0.138 |
| M_scale → state participation | 0.813 | 0.875 |
| M_scale → state entropy dimension | 0.823 | 0.876 |
| D_eff → state participation | 0.239 | 0.546 |
| D_eff → state entropy dimension | 0.246 | 0.551 |
| basis participation → state participation | 0.966 | 0.967 |
| basis coherence → state participation | 0.409 | 0.188 |
| basis condition → state participation | 0.128 | 0.592 |

All reported n values are 675 except basis-condition relationships, where
non-finite condition-number cases are excluded (n = 585).

### Findings

1. **Basis geometry is very strongly coupled to realized state geometry.**
   Basis participation versus state participation gives Pearson r = 0.966 and
   Spearman rho = 0.967. This is the strongest bridge tested in 13B.

2. **Memory scale is a strong empirical descriptor of state geometry.**
   M_scale gives Pearson r = 0.813 with state participation and r = 0.823
   with state entropy dimension; Spearman rho is approximately 0.875 for both.
   This is consistent with the 13A-1 observation that memory-scale geometry is
   more informative about realized state structure than D_eff alone.

3. **D_eff is not interchangeable with realized state dimension.**
   Its Pearson relationship with state participation is only 0.239
   (Spearman 0.546). Therefore D_eff should remain a kernel-complexity
   descriptor, not be renamed as MIN state dimension.

4. **Nominal mode count is insufficient.** At L = 16, mean participation
   dimensions are approximately 1.00/1.00 for clustered rates,
   1.20/1.19 for logspread, and 1.28/1.32 for wide rates
   (basis/state respectively). Adding modes therefore does not imply a
   proportional increase in realized dimensionality.

5. **Rate geometry matters.** Clustered rates produce nearly redundant
   exponential directions over the finite horizon. More separated rate
   geometries produce more distinguishable directions and larger realized
   state dimensions.

### Important methodological boundary

The raw finite-horizon Gram matrix uses the ordinary L2 inner product on
phi_i(t)=exp(-gamma_i t). Its eigenvalue participation therefore reflects both
mode distinguishability and the large norm differences associated with decay
rates. In the wide-rate cases, normalized coherence can be relatively low while
participation remains small because fast exponentials have much smaller L2 norm.

Consequently, 13B does **not** establish that low participation dimension by
itself means strong temporal collinearity. The Gram analysis establishes a
finite-horizon geometry bridge, but a normalized Gram/correlation basis is a
natural control if this distinction becomes important in a later experiment.

A second boundary is that the primary D_state is the **unweighted** covariance
geometry of the per-mode causal states. The experiment separately records
weighted-state geometry so that kernel amplitude weighting is not confused with
mode excitation. Thus D_eff and weighted state structure remain distinct
objects.

## Dimension hierarchy

    L != D_eff != D_basis != D_state

where D_task is reserved for later task-level evaluation.

These quantities answer different questions: specified modes, weighted kernel
complexity, finite-horizon basis geometry, and signal-excited state geometry.

## Interpretation boundary

A larger geometric dimension is not itself a utility claim. 13B establishes
the geometry/mechanism layer; 13C can test whether the resulting representation
is useful for a defined task.

The supported working hypothesis is:

> Realized MIN temporal dimensionality depends on the geometric distinguishability
> of the kernel's temporal modes under the sampling and signal excitation process,
> rather than on nominal SOE mode count alone.

This remains an empirical hypothesis, not a general theorem.

## Outputs

The CI artifact 'experiment-13b-results' contains:

- experiments/results/13B_geometry_to_state_structure_results.csv
- experiments/results/13B_geometry_to_state_structure_summary.csv
- experiments/results/13B_geometry_to_state_structure_summary.json

Artifact SHA-256:
'c50300ad2cca29edf09b799f1fbf666b3c72ecfd22d1441b11a6e4c3a0709320'.
