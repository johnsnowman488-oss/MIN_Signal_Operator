import importlib.util
from pathlib import Path

def load():
    p=Path(__file__).resolve().parents[1]/"experiments"/"15C_extended_timescale_iir_control.py"
    spec=importlib.util.spec_from_file_location("exp15c_test",p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_grid_and_shapes():
    m=load()
    assert m.SCALE_FACTORS==(4.0,8.0,12.0,20.0,32.0,64.0)
    for p in m.PROCESSES:
        for f in m.SCALE_FACTORS:
            g=m.sweep_rates(p,f)
            assert g.shape==(16,)
            assert len(set(g.tolist()))==16
            assert (g>0).all()

def test_one_pole_and_repository_equivalence():
    m=load()
    assert m.U.validate_repository_equivalence()<1e-9
    x=[0,1,0,-1,0]*20
    q=m.one_pole_state(__import__("numpy").asarray(x,float),2.0)
    assert q.shape==(100,1)
    assert __import__("numpy").isfinite(q).all()
