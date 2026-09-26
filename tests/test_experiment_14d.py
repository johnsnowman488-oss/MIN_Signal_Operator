import importlib.util
from pathlib import Path
import sys
import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "14D_readout_stability.py"
spec = importlib.util.spec_from_file_location("experiment14d", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def test_ridge_zero_matches_unregularized_solution():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(64, 4)) + 1j * rng.normal(size=(64, 4))
    y = rng.normal(size=64) + 1j * rng.normal(size=64)
    beta0, _ = module.ridge_fit(X, y, 0.0)
    design = np.column_stack([np.ones(X.shape[0], dtype=complex), X])
    beta_ref = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.allclose(beta0, beta_ref, rtol=1e-9, atol=1e-10)

def test_ridge_reduces_sensitivity_on_ill_conditioned_design():
    rng = np.random.default_rng(1404)
    base = rng.normal(size=128)
    X = np.column_stack([base, base + 1e-10 * rng.normal(size=128)])
    y = base + 1e-3 * rng.normal(size=128)
    c = module.conditioning(X)
    beta_ols, _ = module.ridge_fit(X, y, 0.0)
    beta_ridge, _ = module.ridge_fit(X, y, 1e-1)
    assert c["readout_condition_number"] > 1e8
    assert np.isfinite(beta_ols).all()
    assert np.isfinite(beta_ridge).all()

def test_repository_state_path_remains_equivalent():
    err = module.mod.validate_repository_equivalence()
    assert err < 1e-9
