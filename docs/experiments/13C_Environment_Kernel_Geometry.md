# Experiment 13C-1 — Environment → Kernel Geometry

## Objective

Experiment 13C-1 begins the next layer of the MIN representation program:

    environment
        ↓
    temporal covariance geometry
        ↓
    positive SOE kernel identification
        ↓
    GFE/GGFE kernel geometry
        ↓
    finite-horizon temporal basis geometry
        ↓
    MIN state geometry

The experiment is deliberately task-agnostic. It does not test BER, EVM, propagation advantage, receiver recovery, classification, prediction, or a neural model.

## Why this variant exists

13A-1 established a controlled kernel/SOE geometry atlas. 13B showed that finite-horizon basis geometry is strongly coupled to realized MIN state geometry. The unresolved upstream question is whether a kernel can be selected or approximated from an environment's temporal geometry rather than treated as an arbitrary independent design choice.

13C-1 therefore uses an **oracle environment covariance envelope** as the environmental object. A positive SOE approximation is identified from that envelope with a fixed logarithmic decay-rate dictionary using nonnegative least squares (NNLS). The resulting kernel is then passed through the same basis/state geometry machinery used in 13B.

This isolates the environment → kernel mapping before introducing covariance-estimation uncertainty.

## Controlled grid

- Signals: BPSK, QPSK, 16-QAM
- Environment families: white-limit, short-memory exponential, multiscale exponential, power-law, squared-exponential
- Seeds: 4
- Fixed SOE dictionary: 16 logarithmically spaced rates from 0.5 to 100 s⁻¹
- Total: **60 cases**

The environment envelopes are normalized so that C_env(0)=1.

The five profiles are:

- white_limit: exp(-t/0.005)
- short: exp(-t/0.05)
- multiscale: 0.65 exp(-t/0.03) + 0.35 exp(-t/0.7)
- powerlaw: (1+t/0.2)^(-0.7)
- squared_exp: exp(-(t/0.15)^2)

These are controlled covariance geometries, not claims that any one profile is a universal physical noise model.

## Kernel identification

For each environment, construct

A_ij = exp(-gamma_j t_i)

with the fixed dictionary gamma_j. Solve

min_w ||A w - C_env||_2

subject to

w_j >= 0.

The fitted weights are normalized to sum to one before the GFE/GGFE descriptors are computed.

The positive constraint is important because the present MIN kernel program is tracking completely-monotone/SOE-compatible memory structure. The identification step is not trained against a downstream task.

## Metrics

### Environment geometry

- correlation integral
- correlation centroid
- half-decay time
- squared-envelope area

### Kernel/GFE/GGFE

- M_cap
- M_scale
- M_res
- H_mem
- entropy effective count
- documented D_eff = L exp(H_mem)
- number of nonzero fitted modes
- relative L2 fit error
- R² and maximum absolute fit error

### Basis geometry

The same finite-horizon SOE Gram matrix from 13B is used:

G_ij = (1-exp(-(gamma_i+gamma_j)T))/(gamma_i+gamma_j).

Recorded descriptors include basis participation dimension, entropy dimension, rank thresholds, condition number, and maximum normalized coherence.

### MIN state geometry

The same causal exponential state realization is applied to the controlled digital signals. Recorded descriptors include state participation dimension, state entropy dimension, rank thresholds, conditioning, component collinearity, energy, trajectory radius/length, and weighted-state geometry.

The distinction remains:

L != D_eff != D_basis != D_state.

## Interpretation boundary

13C-1 is a **mechanism experiment**, not a utility experiment.

A good environment-to-kernel fit means only that the selected positive SOE dictionary can represent that particular controlled covariance envelope. It does not mean the fitted kernel is optimal for communication or any other task.

Likewise, a relationship between environment descriptors, D_eff, D_basis, and D_state does not establish causality or task usefulness.

The oracle envelope is intentional. 13C-2 should introduce finite-sample covariance estimation and determine how much the environment → kernel geometry mapping degrades under observation uncertainty.

## Expected continuation

The 13C branch should proceed in layers:

1. **13C-1:** oracle environment geometry → positive SOE kernel geometry.
2. **13C-2:** finite-sample/noisy environment observation → kernel identification uncertainty.
3. **13C-3:** environment → kernel → basis → state relationship atlas across model order and environment mismatch.
4. **13C-4:** only then introduce a narrowly defined task metric (D_task) to test whether environment-conditioned representation geometry has measurable utility.

The central research hypothesis remains:

> The appropriate MIN temporal representation can be related to the temporal geometry of the signal environment, with the environment influencing kernel geometry and the kernel geometry constraining the realized state representation.

This remains an empirical hypothesis, not a theorem.
