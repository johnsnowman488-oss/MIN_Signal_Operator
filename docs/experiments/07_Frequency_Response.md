# Experiment 07 — MIN frequency-response characterization

Date: 2026-09-21

## Purpose

Experiment 07 characterizes the causal MIN operator on complex exponentials,

\[
x(t)=e^{i\omega t},
\qquad
(M_Kx)(t)=\int_0^t K(u)e^{i\omega(t-u)}\,du.
\]

The experiment separates three effects that can otherwise be confused:

1. the reference frequency response \(H_K(\omega)\);
2. finite observation-horizon truncation;
3. startup/nonstationarity in a finite causal record.

For a conventional stationary response,

\[
H_K(\omega)=\int_0^\infty K(u)e^{-i\omega u}\,du.
\]

For the power-law kernel with \(\alpha=0.70\), the DC integral diverges. Nonzero-frequency values are therefore evaluated with weighted oscillatory quadrature rather than treating a finite time window as an exact infinite-horizon response.

## Kernels

\[
K_1(t)=1,
\qquad
K_2(t)=e^{-t/0.10},
\]

\[
K_3(t)=0.7e^{-2t}+0.3e^{-30t},
\qquad
K_4(t)=\left(1+\frac{t}{0.50}\right)^{-0.70}.
\]

The analytic non-identity references are

\[
H_2(\omega)=\frac{0.10}{1+i0.10\omega},
\]

\[
H_3(\omega)=\frac{0.7}{2+i\omega}+\frac{0.3}{30+i\omega}.
\]

Their DC gains are 0.10 and 0.365; the identity baseline has gain 1. The power-law case has no finite DC gain for \(\alpha=0.70\).

## Numerical protocol

- \(\Delta t=0.001\,\mathrm{s}\);
- observation horizon \(T=12\,\mathrm{s}\);
- 12 logarithmically spaced nonzero angular frequencies from \(0.5\) to \(40\,\mathrm{rad/s}\).

For each kernel/frequency pair, compute the finite-window transform

\[
H_{\mathrm{window}}(\omega)=\int_0^T K(u)e^{-i\omega u}\,du,
\]

and the causal response produced by the reference MIN discretization.

At the observation endpoint,

\[
H_{\mathrm{endpoint}}=y(T)e^{-i\omega T}.
\]

This is a direct check of the finite-window operator because the causal integral at \(T\) is exactly the truncated transform up to numerical quadrature error.

A tail estimator is also computed,

\[
H_{\mathrm{tail}}=\frac{1}{N_{\mathrm{tail}}}
\sum_{n\in\mathrm{tail}}y_ne^{-i\omega t_n},
\]

using tail starts of 1 s for the single exponential, 4 s for the two-scale SOE, and 6 s for the power-law. It is intentionally treated as an estimator, not assumed to be exact.

## Reference calculation

The exponential and SOE references are analytic. The power-law response at nonzero frequency is evaluated with SciPy's weighted oscillatory quadrature on the semi-infinite interval. This is preferable to selecting an arbitrary very long finite window because oscillatory algebraic-tail integrals can converge non-monotonically under naive truncation.

## Recorded metrics

The result CSV contains:

- reference, finite-window, endpoint, and tail magnitudes;
- reference phase;
- endpoint-vs-window relative error;
- tail-estimator-vs-window relative error;
- window-vs-reference relative error;
- endpoint-vs-reference relative error.

The accompanying JSON records settings, per-kernel summary statistics, and low/mid/high-frequency diagnostics.

## Results

The recorded artifacts are:

- `experiments/results/07_frequency_response_results.csv`
- `experiments/results/07_frequency_response_summary.json`

The causal endpoint estimator matches the finite-window transform to numerical precision across all 48 tested kernel/frequency pairs. The largest endpoint-vs-window relative error is about \(3.2\times10^{-14}\).

For the short-memory exponential and two-scale SOE kernels, the 12-second observation window tracks the analytic reference closely. The maximum window-vs-reference deviations are about \(1.42\times10^{-4}\) and \(1.52\times10^{-4}\), respectively. Their tail estimators remain very close to the finite-window response, with median relative errors of about \(3.8\times10^{-7}\) and \(8.3\times10^{-6}\).

The power-law case behaves differently. At \(\omega=0.5\,\mathrm{rad/s}\), the 12-second finite-window response differs from the nonzero-frequency reference by about 20.6%. Across the frequency grid, the median endpoint-vs-reference error is about 11.6%. The median tail-estimator-vs-window error is about 11.7%, and the maximum tail-model residual is about 25.3%.

These results do not demonstrate an application advantage for any kernel. They show that short-memory and long-memory MIN can have very different relationships between a finite observed record and an infinite-horizon transfer description.

## Scientific interpretation

The controlled experiment supports the distinction

\[
\boxed{
\text{operator discretization error}
\;\neq\;
\text{finite-horizon memory error}
\;\neq\;
\text{tail-estimation error}
}
\]

and provides evidence that the last two effects become substantial for the tested power-law kernel.

This matters for the next signal-processing phase. A measured spectral change in a finite record should not automatically be interpreted as a stationary transfer-function effect when the underlying MIN kernel retains long history.

## Reproducibility

From the repository root:

```bash
pip install -e ".[research]"
python experiments/07_frequency_response.py
```

The experiment is deterministic and uses no random variables.
