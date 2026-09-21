import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "07_frequency_response.py"
spec = importlib.util.spec_from_file_location("experiment07", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_closed_form_references():
    exp = mod.KERNELS[1]
    soe = mod.KERNELS[2]
    omega = 3.0
    assert np.isclose(mod.analytic_reference(exp, omega), 0.10 / (1 + 1j * 0.10 * omega))
    assert np.isclose(
        mod.analytic_reference(soe, omega),
        0.7 / (2 + 1j * omega) + 0.3 / (30 + 1j * omega),
    )


def test_endpoint_matches_finite_window():
    t = np.arange(0.0, 2.0 + mod.DT / 2.0, mod.DT)
    for kernel in mod.KERNELS:
        omega = 2.0 if kernel.kind == "powerlaw" else 5.0
        y = mod.direct_min(kernel, t, omega)
        endpoint = y[-1] * np.exp(-1j * omega * t[-1])
        if kernel.kind == "identity":
            window = 1.0 + 0.0j
        else:
            window = mod.trapezoid_transform(mod.kernel_value(kernel, t), t, omega)
        assert abs(endpoint - window) / max(abs(window), 1e-15) < 1e-10


def test_powerlaw_nonzero_reference_and_dc_statement():
    powerlaw = mod.KERNELS[-1]
    response, method = mod.frequency_response_reference(powerlaw, 1.0)
    assert method == "scipy_weighted_quadrature"
    assert np.isfinite(response.real)
    assert np.isfinite(response.imag)
    assert powerlaw.kind == "powerlaw"


def test_experiment_rows_are_finite_and_group_delay_is_defined():
    rows, summary = mod.build_rows()
    assert len(rows) == len(mod.KERNELS) * len(mod.OMEGAS)
    assert all(np.isfinite(row["operator_vs_window_rel_error"]) for row in rows)
    assert all(np.isfinite(row["reference_group_delay_s"]) for row in rows)
    assert summary["powerlaw_alpha_0.70_tau_0.50"]["max_operator_vs_window_rel_error"] < 1e-10
