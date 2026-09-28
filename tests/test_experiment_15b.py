from pathlib import Path
import importlib.util, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
u=importlib.util.spec_from_file_location("u",ROOT/"experiments/15_utils.py")
U=importlib.util.module_from_spec(u); sys.modules["u"]=U; u.loader.exec_module(U)
e=importlib.util.spec_from_file_location("e",ROOT/"experiments/15B_process_geometry_alignment.py")
E=importlib.util.module_from_spec(e); sys.modules["e"]=E; e.loader.exec_module(E)

def test_process_scales_are_distinct():
    assert E.PROCESS_PARAMS["short"]["taus"][0] < E.PROCESS_PARAMS["long"]["taus"][0]
    assert E.PROCESS_PARAMS["multiscale"]["taus"] == (0.025,0.50)

def test_15b_state_equivalence():
    assert U.validate_repository_equivalence() < 1e-9

def test_15b_grid_dimensions():
    assert len(E.PROCESSES)==4
    assert len(E.TASKS)==2
    assert len(E.GEOMETRIES)==3
