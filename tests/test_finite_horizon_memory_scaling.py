import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "08_finite_horizon_memory_scaling.py"
spec = importlib.util.spec_from_file_location("experiment08", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_exponential_finite_horizon_matches_closed_form_tail():
    kernel = mod.KernelSpec("exp", "exponential", tau=0.1)
    omega = 2.0
    h_inf = mod.h_infinite(kernel, omega)
    h_t = mod.h_finite(kernel, omega, 0.5)
    expected = abs(np.exp(-(1.0 / kernel.tau + 1j * omega) * 0.5))
    assert np.isclose(abs(h_t - h_inf) / abs(h_inf), expected, rtol=1e-12)


def test_horizon_error_decreases_for_alpha_070_grid():
    kernel = mod.KernelSpec("pl", "powerlaw", tau=0.5, alpha=0.7)
    omega = 1.0
    h_inf = mod.h_infinite(kernel, omega)
    errors = [
        abs(mod.h_finite(kernel, omega, T) - h_inf) / abs(h_inf)
        for T in mod.HORIZONS
    ]
    assert errors[-1] < errors[0]


def test_summary_contains_all_kernel_frequency_thresholds():
    rows = []
    for kernel in mod.KERNELS:
        rows.extend(mod.rows_for_kernel(kernel))
    summary = mod.summarize(rows)
    assert len(rows) == len(mod.KERNELS) * len(mod.OMEGAS) * len(mod.HORIZONS)
    for kernel in mod.KERNELS:
        thresholds = summary[kernel.name]["frequency_specific_horizon_thresholds"]
        assert len(thresholds) == len(mod.OMEGAS)
        assert all("T_for_10pct_or_better_s" in item for item in thresholds.values())
