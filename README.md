# MIN Signal Operator

A research laboratory for the Memory-Integrated Network (MIN) signal operator.

## Scope

This repository begins with a transparent reference implementation of the causal memory operator

\[
(Mf)(t)=\int_0^t K(t-s)f(s)\,ds.
\]

The initial implementation is deliberately small. It is intended for verification and experiments, not production performance.

## Research progression

1. Verify the core memory operator.
2. Explore kernel families and spectral representations.
3. Add state-space / sum-of-exponentials (SOE) realizations.
4. Introduce hierarchical MIN as a controlled extension.
5. Build the synthetic signal atlas and characterize MIN on elementary waveforms.
6. Characterize MIN frequency response while separating finite-horizon and startup effects.
7. Quantify finite-horizon memory scaling across frequency and memory-tail regimes.
8. Add controlled noise/channels and benchmark against FIR, IIR, Volterra, and memory-polynomial baselines.
9. Test synthetic communication signals, then real IQ and SDR.
10. Profile and optimize only after experiments justify it.

## Repository structure

- min/ — core implementation, kernels, synthetic signals, and metrics
- tests/ — numerical correctness tests
- notebooks/ — executable research experiments
- experiments/ — reproducible benchmark runners and recorded result artifacts
- docs/ — theory-to-experiment research protocol
- cpp/ — future performance implementations

## Development

The reference implementation uses NumPy. Install the project with your preferred Python environment, then run the test suite with:

```bash
pytest
```

The frequency-response experiment uses SciPy's weighted oscillatory quadrature for the long-memory power-law reference. Install the research dependencies with:

```bash
pip install -e ".[research]"
```

The notebooks are experiments rather than the source of truth; reusable logic belongs in `min/` and tests should accompany numerical claims.

## Status

Early research laboratory. Experiments 01–05 establish the numerical MIN/SOE and identification foundation. Experiment 06 begins the synthetic signal atlas. Experiment 07 characterizes frequency response and explicitly separates operator discretization from finite-history effects. Experiment 08 measures the horizon required for short-memory and algebraic-tail kernels to approach their infinite-horizon response. Claims about signal-processing benefit, robustness, or communication performance remain experimental questions and are not assumed by the implementation.
