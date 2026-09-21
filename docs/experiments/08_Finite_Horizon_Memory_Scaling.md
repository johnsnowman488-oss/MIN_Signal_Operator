# Experiment 08 — finite-horizon memory scaling

Date: 2026-09-21

## Purpose

Experiment 07 showed that a finite observation window can differ substantially from an infinite-horizon frequency response for the tested power-law kernel. Experiment 08 asks the next quantitative question:

> How does that discrepancy decay as the observation horizon grows, and how does the decay depend on frequency and memory-tail exponent?

For a causal kernel,

\[
H_T(\omega)=\int_0^T K(u)e^{-i\omega u}\,du,
\qquad
H_\infty(\omega)=\int_0^\infty K(u)e^{-i\omega u}\,du.
\]

The primary metric is

\[
E_T(\omega)=\frac{|H_T(\omega)-H_\infty(\omega)|}{|H_\infty(\omega)|}.
\]

This is a frequency-specific finite-memory error, not a communication-performance metric.

## Experimental design

Horizon values:

\[
T\in\{0.5,1,2,4,8,16,32,64\}\;s.
\]

Frequencies:

\[
\omega\in\{0.25,0.5,1,2,4,8\}\;rad/s.
\]

Kernels:

\[
K_E(t)=e^{-t/0.10},
\]

and

\[
K_\alpha(t)=\left(1+\frac{t}{0.50}\right)^{-\alpha},
\qquad
\alpha\in\{0.50,0.70,0.90\}.
\]

All three power-law kernels have divergent DC integrals because \(\alpha\le1\). The experiment therefore compares only nonzero-frequency oscillatory transforms.

## Numerical method

The exponential finite-window transform is evaluated analytically:

\[
H_T(\omega)=
\frac{1-e^{-(1/\tau+i\omega)T}}{1/\tau+i\omega}.
\]

This removes a numerical-discretization floor from the horizon-scaling measurement.

For the power-law kernels, both \(H_T\) and \(H_\infty\) use SciPy weighted oscillatory quadrature. The finite-window calculation uses a finite interval; the infinite reference uses the corresponding semi-infinite oscillatory integral.

The result table also records magnitude error and phase error. For a practical horizon criterion, the code reports the first tested \(T\) at which \(E_T\le10\%\), \(5\%\), and \(1\%\), when such a horizon occurs within the tested grid.

## Results

The run produced 192 kernel/frequency/horizon points.

### Exponential short-memory control

For \(K_E\), the exact tail relation gives

\[
E_T=e^{-T/0.10}.
\]

At \(T=0.5\) s the error is about 0.674%, already below 1%. At 8 s and beyond the measured error is at numerical round-off level.

### Power-law \(\alpha=0.70\)

The long-memory case remains substantially horizon-dependent. Across the six frequencies, the median error at \(T=8\) s is about 19.2%, while the maximum error at \(T=64\) s is about 9.03%.

The first tested horizon reaching 10% or better depends strongly on frequency: it is 64 s at 0.25 and 0.5 rad/s, 32 s at 1 and 2 rad/s, and 16 s at 4 and 8 rad/s. A 5% criterion is reached only at 64 s for the tested frequencies 2, 4, and 8 rad/s; the 1% criterion is not reached within the tested horizon grid.

### Tail-exponent comparison

At \(T=8\) s, the median relative errors across the tested frequency grid are approximately:

| Kernel | Median \(E_T\) at 8 s |
|---|---:|
| exponential, \(\tau=0.10\) | effectively 0 |
| power-law, \(\alpha=0.50\) | 30.9% |
| power-law, \(\alpha=0.70\) | 19.2% |
| power-law, \(\alpha=0.90\) | 11.9% |

At \(T=64\) s, the corresponding maximum errors over the tested frequencies are about 18.5%, 9.03%, and 4.31% for \(\alpha=0.50,0.70,0.90\), respectively.

These results show a systematic dependence on the algebraic tail exponent in this experiment: smaller \(\alpha\) retains stronger finite-horizon effects.

## Interpretation

The experiment supports a precise distinction:

\[
\boxed{\text{finite-horizon memory error is kernel- and frequency-dependent}}
\]

rather than treating all MIN kernels as ordinary stationary filters with one practically reachable transfer function.

For the tested power-law family, increasing \(\alpha\) shortens the practical horizon, while the exponential control reaches its infinite-horizon response much faster.

This does **not** establish a communications advantage. It establishes an operator property that can now be carried into the next stage: communication signals can be tested under deliberately chosen memory horizons, with the horizon itself treated as an experimental variable.

## Next experiment

Experiment 09 should introduce BPSK, QPSK, and QAM signals while preserving the same controlled memory regimes. The first comparison should be transformation fidelity and channel robustness, before any claim of BER improvement is made.

## Reproducibility

From the repository root:

~~~bash
pip install -e ".[research]"
python experiments/08_finite_horizon_memory_scaling.py
pytest -q tests/test_finite_horizon_memory_scaling.py
~~~

Artifacts:

- \`experiments/results/08_finite_horizon_memory_scaling_results.csv\`
- \`experiments/results/08_finite_horizon_memory_scaling_summary.json\`
- \`notebooks/08_Finite_Horizon_Memory_Scaling.ipynb\`
