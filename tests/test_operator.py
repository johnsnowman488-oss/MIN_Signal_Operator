import numpy as np

from min import MemoryOperator, exponential_kernel


def test_constant_input_exponential_kernel_converges():
    t = np.linspace(0.0, 5.0, 1001)
    tau = 0.8
    y = MemoryOperator(lambda lag: exponential_kernel(lag, tau)).apply(t, np.ones_like(t))
    exact = tau * (1.0 - np.exp(-t / tau))
    assert np.max(np.abs(y - exact)) < 2e-5


def test_complex_iq_is_supported():
    t = np.linspace(0.0, 1.0, 101)
    x = np.exp(1j * 2 * np.pi * 3 * t)
    y = MemoryOperator(lambda lag: exponential_kernel(lag, 0.1)).apply(t, x)
    assert np.iscomplexobj(y)
    assert np.isfinite(y).all()


def test_complex_kernel_with_real_signal_is_supported():
    t = np.linspace(0.0, 1.0, 101)
    x = np.ones_like(t)
    kernel = lambda lag: np.exp(-(1.0 + 0.5j) * lag)
    y = MemoryOperator(kernel).apply(t, x)
    assert np.iscomplexobj(y)
    assert np.isfinite(y).all()
