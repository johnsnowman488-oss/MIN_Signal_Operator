import importlib.util
from pathlib import Path
import sys

import numpy as np

from min import SOEMemory


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "14C_geometry_to_task.py"
spec = importlib.util.spec_from_file_location("experiment14c", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_geometry_control_matches_L_and_D_eff_but_changes_M_scale():
    rows = []
    for geometry in module.GEOMETRIES:
        gammas = module.rate_pattern(geometry)
        weights = module.uniform_weights()
        rows.append(module.kernel_descriptors(gammas, weights))
    assert all(r["dictionary_mode_count"] == 16 for r in rows)
    assert np.allclose([r["gfe_d_eff"] for r in rows], 256.0)
    assert len({round(r["gfe_m_scale_decades"], 10) for r in rows}) == 3


def test_geometry_changes_scalar_representation():
    rng = np.random.default_rng(14)
    x = rng.normal(size=257) + 1j * rng.normal(size=257)
    outputs = []
    weights = module.uniform_weights()
    for geometry in module.GEOMETRIES:
        spec = module.KERNELS[geometry]
        q = module.exact_piecewise_linear_states(x, spec["gammas"], spec["coefficients"])
        outputs.append(q @ weights)
    assert not np.allclose(outputs[0], outputs[1])
    assert not np.allclose(outputs[1], outputs[2])


def test_fast_state_path_matches_repository_soememory():
    rng = np.random.default_rng(1401)
    x = rng.normal(size=129) + 1j * rng.normal(size=129)
    spec = module.KERNELS["wide"]
    fast = module.exact_piecewise_linear_states(x, spec["gammas"], spec["coefficients"])
    reference = SOEMemory(spec["weights"], spec["gammas"]).state_trajectory(
        np.arange(x.size, dtype=float) * module.DT, x
    )
    assert np.allclose(fast, reference, rtol=1e-11, atol=1e-12)
