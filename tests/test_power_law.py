import numpy as np
import pytest

from min import power_law_kernel


def test_power_law_kernel_is_finite_and_normalized():
    t = np.linspace(0.0, 10.0, 101)
    k = power_law_kernel(t, alpha=0.7, tau=0.5)
    assert np.isfinite(k).all()
    assert k[0] == pytest.approx(1.0)
    assert np.all(np.diff(k) < 0)


def test_power_law_kernel_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        power_law_kernel(np.array([0.0]), alpha=0.0)
    with pytest.raises(ValueError):
        power_law_kernel(np.array([0.0]), alpha=1.0, tau=0.0)
