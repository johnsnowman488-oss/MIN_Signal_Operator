import numpy as np

from min.metrics.equalization import add_awgn, apply_fir_equalizer, design_fir_equalizer
from experiments._exp11_helpers import memory_filter, channel


def test_multipath_channel_is_normalized():
    x = np.ones(64, dtype=complex)
    y, h = channel(x, "multipath_3tap", np.random.default_rng(1))
    assert np.isclose(np.sum(np.abs(h)**2), 1.0)
    assert y.size == x.size


def test_flat_rayleigh_is_unit_magnitude():
    x = np.ones(64, dtype=complex)
    y, h = channel(x, "flat_rayleigh", np.random.default_rng(2))
    assert np.isclose(abs(h), 1.0)
    assert np.allclose(y, h*x)


def test_memory_filter_preserves_length():
    x = np.arange(128, dtype=float) + 1j
    for name in ("identity", "exponential_tau_0.10", "soe_two_scale", "powerlaw_alpha_0.70"):
        y = memory_filter(x, name, 16)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_equalizer_reduces_known_three_tap_channel():
    rng = np.random.default_rng(3)
    x = rng.normal(size=512) + 1j*rng.normal(size=512)
    h = np.array([1.0, .4+.1j, .2-.1j])
    y = np.convolve(x, h)[:x.size]
    c = design_fir_equalizer(y[:256], x[:256], taps=15, ridge=1e-5)
    z = apply_fir_equalizer(y, c)
    raw = np.mean(np.abs(y[64:] - x[64:])**2)
    eq = np.mean(np.abs(z[64:] - x[64:])**2)
    assert eq < raw
