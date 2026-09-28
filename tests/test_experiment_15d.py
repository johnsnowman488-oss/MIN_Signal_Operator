from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("exp15d",ROOT/"experiments"/"15D_equal_budget_comparison.py")
M=module_from_spec(spec); spec.loader.exec_module(M)

def test_15d_representation_shapes():
    rng=np.random.default_rng(15)
    x=rng.normal(size=257)
    reps,mg,spread=M.representation_specs("multiscale",8.0,x)
    assert set(reps)=={"min_16","iir_logspread_16","dense_ss_16","fir_16"}
    assert all(v.shape==(257,16) for v in reps.values())
    assert mg.shape==(16,)
    assert spread.shape==(16,)

def test_15d_dense_similarity_preserves_energy():
    rng=np.random.default_rng(1515)
    x=rng.normal(size=257)
    reps,_,_=M.representation_specs("short",8.0,x)
    a=reps["iir_logspread_16"]
    b=reps["dense_ss_16"]
    assert np.isclose(np.linalg.norm(a),np.linalg.norm(b),rtol=1e-10,atol=1e-10)

def test_15d_equal_budget_costs():
    assert {v["state_dimension"] for v in M.COSTS.values()}=={16}
    assert {v["memory_values"] for v in M.COSTS.values()}=={16}
