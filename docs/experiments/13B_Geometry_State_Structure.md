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

This measures whether nominal SOE modes are geometrically distinguishable over
the same finite temporal horizon used to excite the MIN states.

## Empirical MIN state geometry

The sampled causal states satisfy

q_i[n] = exp(-gamma_i dt) q_i[n-1]
       + (1-exp(-gamma_i dt))/gamma_i * x[n].

The covariance spectrum provides state participation dimension, rank,
conditioning, and the entropy analogue

H_state = -sum(p_i log p_i)
D_state,entropy = exp(H_state).

Thus D_eff is compared with a signal-driven state quantity without assuming
that the two are equal.

## Primary relationships

Both Pearson and Spearman relationships are recorded for:

1. M_scale → D_basis
2. M_scale → basis coherence and conditioning
3. M_scale → D_state
4. D_eff → D_state
5. D_basis → D_state
6. basis coherence/conditioning → D_state

## Dimension hierarchy

    L != D_eff != D_basis != D_state

where D_task is reserved for later task-level evaluation.

These quantities answer different questions: specified modes, weighted kernel
complexity, geometrically distinguishable temporal modes, and signal-excited
state dimensions respectively.

## Interpretation boundary

A larger geometric dimension is not itself a utility claim. 13B establishes
the geometry/mechanism layer; 13C can test whether the resulting representation
is useful for a defined task.

The working hypothesis is:

> Realized MIN temporal dimensionality depends on the geometric distinguishability
> of the kernel's temporal modes under the sampling and signal excitation process,
> rather than on nominal SOE mode count alone.

## Outputs

- experiments/results/13B_geometry_to_state_structure_results.csv
- experiments/results/13B_geometry_to_state_structure_summary.csv
- experiments/results/13B_geometry_to_state_structure_summary.json
