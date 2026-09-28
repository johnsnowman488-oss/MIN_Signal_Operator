import importlib.util
from pathlib import Path

def load():
    p = Path(__file__).resolve().parents[1] / "experiments" / "15B3_task_optimal_timescale_sweep.py"
    spec = importlib.util.spec_from_file_location("exp15b3_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_scale_grid_and_kernel_shape():
    m=load()
    assert len(m.SCALE_FACTORS)==7
    for process in m.PROCESSES:
        for factor in m.SCALE_FACTORS:
            g=m.sweep_rates(process,factor)
            assert g.shape==(16,)
            assert len(set(g.tolist()))==16
            assert (g>0).all()

def test_repository_equivalence():
    m=load()
    assert m.U.validate_repository_equivalence() < 1e-9
