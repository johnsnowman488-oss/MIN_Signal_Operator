import numpy as np

from min import MemoryOperator
from min.kernels import exponential_kernel
from min.metrics.equalization import (
    add_awgn,
    apply_fir_equalizer,
    apply_scalar_compensation,
    design_fir_equalizer,
    evm,
)
from experiments._exp10_helpers import apply_min_uniform


def test_scalar_compensation_removes_complex_gain():
    rng = np.random.default_rng(4)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    y = (1.7 - 0.8j) * x
    z = apply_scalar_compensation(y, x)
    assert np.max(np.abs(z - x)) < 1e-12


def test_fir_equalizer_recovers_delayed_fir():
    rng = np.random.default_rng(5)
    x = rng.normal(size=256) + 1j * rng.normal(size=256)
    h = np.array([0.8 + 0.1j, 0.25 - 0.2j, -0.1 + 0.05j])
    y = np.convolve(x, h, mode="full")[: x.size]
    c = design_fir_equalizer(y, x, taps=5, ridge=1e-10)
    z = apply_fir_equalizer(y, c)
    assert np.mean(np.abs(z[20:] - x[20:]) ** 2) < 1e-5


def test_awgn_is_reproducible_and_has_requested_power_ratio():
    x = np.ones(2000, dtype=complex)
    a = add_awgn(x, 10.0, np.random.default_rng(9))
    b = add_awgn(x, 10.0, np.random.default_rng(9))
    assert np.array_equal(a, b)
    measured = np.mean(np.abs(a - x) ** 2)
    assert np.isclose(measured, 0.1, rtol=0.08)


def test_fft_trapezoid_matches_reference_operator():
    n = 256
    dt = 0.01
    t = np.arange(n) * dt
    x = np.exp(1j * 1.7 * t) + 0.2 * np.sin(0.4 * t)
    reference = MemoryOperator(lambda lag: exponential_kernel(lag, 0.10)).apply(t, x)
    accelerated = apply_min_uniform("exponential_tau_0.10", t, x)
    assert np.max(np.abs(reference - accelerated)) < 1e-11


def test_evm_zero_for_exact_reference():
    x = np.array([1, -1, 1j, -1j], dtype=complex)
    assert evm(x, x) == 0.0
