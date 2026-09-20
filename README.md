# MIN Signal Operator

A research laboratory for the Memory-Integrated Network (MIN) signal operator.

## Scope

This repository begins with a transparent reference implementation of the causal memory operator

(Mf)(t) = integral_0^t K(t-s) f(s) ds.

The initial implementation is deliberately small. It is intended for verification and experiments, not production performance.

## Research progression

1. Verify the core memory operator.
2. Explore kernel families and spectral representations.
3. Add state-space / sum-of-exponentials (SOE) realizations.
4. Introduce hierarchical MIN as a controlled extension.
5. Benchmark against convolution, Volterra, and memory-polynomial baselines.
6. Test synthetic and real IQ signals.
7. Profile and optimize only after experiments justify it.

## Repository structure

- min/ — core implementation and kernels
- tests/ — numerical correctness tests
- notebooks/ — executable research experiments
- cpp/ — future performance implementations

## Development

The reference implementation uses NumPy. Install the project with your preferred Python environment, then run the test suite with:

pytest

The notebooks are experiments rather than the source of truth; reusable logic belongs in min/ and tests should accompany numerical claims.

## Status

Early research prototype. Claims about signal-processing benefit, robustness, or communication performance are experimental questions and are not assumed by the implementation.
