from pathlib import Path
import importlib.util, sys
ROOT = Path(__file__).resolve().parents[1]
u=importlib.util.spec_from_file_location("u",ROOT/"experiments/15_utils.py")
U=importlib.util.module_from_spec(u); sys.modules["u"]=U; u.loader.exec_module(U)
e=importlib.util.spec_from_file_location("e2",ROOT/"experiments/15B2_process_specific_alignment.py")
E=importlib.util.module_from_spec(e); sys.modules["e2"]=E; e.loader.exec_module(E)

def test_alignment_roles_have_equal_mode_count_and_d_eff():
    for process in E.PROCESSES:
        for role in E.ROLES:
            spec=U.ALIGNMENT_KERNELS[process][role]
            assert len(spec["gammas"]) == U.MODES
            assert len(spec["weights"]) == U.MODES
            assert abs(U.ALIGNMENT_KERNEL_DESCRIPTORS[process][role]["gfe_d_eff"] - U.MODES**1.0) < 1e-12

def test_matched_and_displaced_are_distinct():
    for process in E.PROCESSES:
        m=U.ALIGNMENT_KERNELS[process]["matched"]["gammas"]
        d=U.ALIGNMENT_KERNELS[process]["displaced"]["gammas"]
        assert not (m == d).all()

def test_15b2_state_equivalence():
    assert U.validate_repository_equivalence() < 1e-9

def test_15b2_grid():
    assert len(E.PROCESSES)==4
    assert len(E.TASKS)==2
    assert E.ROLES == ("matched","displaced","broad")
