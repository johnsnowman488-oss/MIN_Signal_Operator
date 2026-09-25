import importlib.util
import sys
from pathlib import Path

import numpy as np

from min import SOEMemory


def test_state_trajectory_weighted_output_matches_apply():
    t = np.linspace(0.0, 0.5, 101)
    x = np.exp(1j * 2.0 * np.pi * 3.0 * t) + 0.2 * np.sin(2.0 * np.pi * 1.5 * t)
    weights = np.array([0.7, 0.3])
    gammas = np.array([2.0, 30.0])

    model = SOEMemory(weights, gammas)
    states = model.state_trajectory(t, x)
    output = model.apply(t, x)

    assert states.shape == (t.size, gammas.size)
    assert np.iscomplexobj(states)
    assert np.allclose(states @ weights, output, atol=1e-12, rtol=1e-12)


def test_14a_raw_control_returns_observed_center_samples():
    module_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "14A_task_relevant_min_subspace.py"
    )
    spec = importlib.util.spec_from_file_location("experiment14a", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    clean = np.zeros(4 * 4, dtype=complex)
    observed = clean.copy()
    observed[[2, 6, 10, 14]] = np.array([1.0, 2.0, 3.0, 4.0])

    centers = module.raw_center_samples(observed, 4, 4)

    assert np.allclose(centers, [1.0, 2.0, 3.0, 4.0])
    assert not np.allclose(centers, module.raw_center_samples(clean, 4, 4))
