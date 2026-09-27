import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "14F_hidden_memory_markovization.py"
spec = importlib.util.spec_from_file_location("experiment14f", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_hidden_process_has_genuine_temporal_dependence():
    x, hidden = module.generate_hidden_process("wide", 4096, 1406)
    current = x[:-1]
    future = x[1:]
    corr = np.corrcoef(current, future)[0, 1]
    assert np.isfinite(corr)
    assert abs(corr) > 0.05
    assert hidden.shape == (4096, module.PROCESS_DIMS)


def test_history_matrix_is_strictly_causal():
    y = np.arange(20, dtype=float)
    X, target_indices = module.history_matrix(y, history=4, start=4)
    assert X.shape[1] == 4
    assert np.all(X[:, 0] == y[target_indices - 0])
    assert np.all(X[:, 1] == y[target_indices - 1])
    assert np.all(target_indices > np.arange(4, 20)[:len(target_indices)] - 1 + 0)


def test_future_observations_do_not_change_min_state_at_current_time():
    x = np.random.default_rng(14).normal(size=256).astype(complex)
    spec14 = module.mod.KERNELS["wide"]
    q1 = module.mod.exact_piecewise_linear_states(
        x, spec14["gammas"], spec14["coefficients"]
    )
    modified = x.copy()
    modified[129:] += 1000.0
    q2 = module.mod.exact_piecewise_linear_states(
        modified, spec14["gammas"], spec14["coefficients"]
    )
    assert np.allclose(q1[:129], q2[:129], rtol=1e-12, atol=1e-12)


def test_repository_state_equivalence_remains_valid():
    assert module.mod.validate_repository_equivalence() < 1e-9
