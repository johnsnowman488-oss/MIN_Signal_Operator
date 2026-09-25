import importlib.util
from pathlib import Path

import numpy as np

from min import SOEMemory


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "14B_kernel_condition_task_recovery.py"
)

spec = importlib.util.spec_from_file_location("experiment14b", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fast_state_path_matches_repository_state_api():
    rng = np.random.default_rng(41)
    x = rng.normal(size=129) + 1j * rng.normal(size=129)
    gammas = np.geomspace(0.5, 100.0, 16)
    weights = np.full(16, 1.0 / 16.0)

    fast = module.exact_piecewise_linear_states(x, gammas, weights)
    reference = SOEMemory(weights, gammas).state_trajectory(
        np.arange(x.size, dtype=float) * module.DT,
        x,
    )
    assert np.allclose(fast, reference, rtol=1e-11, atol=1e-12)


def test_mismatch_map_is_nonidentical_and_covers_all_environments():
    assert set(module.MISMATCH_MAP) == set(module.ENVIRONMENTS)
    assert set(module.MISMATCH_MAP.values()) == set(module.ENVIRONMENTS)
    assert all(
        module.MISMATCH_MAP[environment] != environment
        for environment in module.ENVIRONMENTS
    )


def test_estimated_kernel_is_normalized_and_finite():
    gammas, weights, meta = module.estimate_environment_kernel(
        "short", 20.0, 12345
    )
    assert gammas.shape == weights.shape == (16,)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.isfinite(meta["estimation_kernel_fit_relative_l2"])
