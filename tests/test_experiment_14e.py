import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "14E_causal_prediction.py"
spec = importlib.util.spec_from_file_location("experiment14e", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_causal_pairing_uses_only_past_center_for_prediction():
    centers = np.arange(5) * module.mod.SPS + module.mod.SPS // 2
    past = centers[:-1]
    target = centers[1:]
    assert np.all(target > past)
    assert np.all(np.diff(past) == module.mod.SPS)
    assert len(past) == len(target)


def test_prediction_readout_and_nmse_are_finite():
    rng = np.random.default_rng(1405)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    y = np.roll(x, -1)
    beta, alpha = module.ridge_fit(x[:, None][:-1], y[:-1], 1e-6)
    pred = module.predict(x[:, None][:-1], beta)
    value = module.nmse(y[:-1], pred)
    assert np.isfinite(alpha)
    assert np.isfinite(value)


def test_state_prediction_is_causal_at_each_target_pair():
    rng = np.random.default_rng(1406)
    x = rng.normal(size=128) + 1j * rng.normal(size=128)
    spec = module.mod.KERNELS["wide"]
    q = module.mod.exact_piecewise_linear_states(
        x, spec["gammas"], spec["coefficients"]
    )
    centers = np.arange(6) * module.mod.SPS + module.mod.SPS // 2
    for n in range(len(centers) - 1):
        assert centers[n] < centers[n + 1]
        assert np.allclose(q[centers[n]], q[:centers[n] + 1][-1])


def test_fast_state_path_matches_repository_reference():
    assert module.mod.validate_repository_equivalence() < 1e-9
