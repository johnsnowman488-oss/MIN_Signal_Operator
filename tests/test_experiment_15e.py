from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location(
    "exp15e", ROOT / "experiments" / "15E_state_budget_sweep.py"
)
M = module_from_spec(spec)
spec.loader.exec_module(M)


def test_15e_budgets_have_exact_dimensions():
    rng = np.random.default_rng(15)
    x = rng.normal(size=257)
    for budget in M.BUDGETS:
        reps, mg, spread = M.representation_specs("multiscale", 8.0, x, budget)
        assert all(v.shape == (257, budget) for v in reps.values())
        assert mg.shape == (budget,)
        assert spread.shape == (budget,)


def test_15e_multiscale_budget_allocation_is_deterministic():
    a = M.budget_rates("multiscale", 8.0, 8)
    b = M.budget_rates("multiscale", 8.0, 8)
    assert np.array_equal(a, b)
    assert len(a) == 8
    assert a[4] > a[5]


def test_15e_one_state_is_well_defined():
    rates = M.budget_rates("hidden_mix", 8.0, 1)
    assert rates.shape == (1,)
    assert np.isfinite(rates[0])


def test_15e_dense_control_preserves_energy():
    rng = np.random.default_rng(1515)
    x = rng.normal(size=257)
    reps, _, _ = M.representation_specs("short", 8.0, x, 4)
    a = reps["iir_logspread_4"]
    b = reps["dense_ss_4"]
    assert np.isclose(
        np.linalg.norm(a), np.linalg.norm(b), rtol=1e-10, atol=1e-10
    )


def test_15e_equal_actual_state_budget():
    for budget in M.BUDGETS:
        c = M.costs(budget)
        assert {v["state_dimension"] for v in c.values()} == {budget}
        assert {v["memory_values"] for v in c.values()} == {budget}
