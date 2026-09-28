from pathlib import Path
import importlib.util
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UTIL_SPEC = importlib.util.spec_from_file_location(
    "min_exp15_utils", ROOT / "experiments" / "15_utils.py"
)
UTIL = importlib.util.module_from_spec(UTIL_SPEC)
sys.modules[UTIL_SPEC.name] = UTIL
UTIL_SPEC.loader.exec_module(UTIL)

EXP_SPEC = importlib.util.spec_from_file_location(
    "experiment15a", ROOT / "experiments" / "15A_geometry_task_generalization.py"
)
EXP = importlib.util.module_from_spec(EXP_SPEC)
sys.modules[EXP_SPEC.name] = EXP
EXP_SPEC.loader.exec_module(EXP)


def test_15_utils_state_matches_repository_soememory():
    assert UTIL.validate_repository_equivalence() < 1e-9


def test_15a_geometry_control_has_equal_mode_count_and_deff():
    descriptors = [UTIL.KERNEL_DESCRIPTORS[g] for g in UTIL.GEOMETRIES]
    assert {d["dictionary_mode_count"] for d in descriptors} == {16}
    assert max(d["gfe_d_eff"] for d in descriptors) - min(
        d["gfe_d_eff"] for d in descriptors
    ) < 1e-12


def test_15a_task_definitions_are_causal_and_distinct():
    assert EXP.TASKS == ("denoise", "prediction", "symbol_recovery")
    x = np.arange(32, dtype=float) + 1j * np.arange(32, dtype=float)
    y = x + 1.0
    symbols = np.array([1 + 0j, -1 + 0j])
    # SPS=16: two symbol centers.
    Xp, yp = EXP.task_arrays(x, y, symbols, "prediction", True)
    assert len(Xp) == 31 and len(yp) == 31
    assert np.all(Xp == y[:-1])
    assert np.all(yp == x[1:])
