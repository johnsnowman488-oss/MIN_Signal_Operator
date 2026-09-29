"""Tests for 15F-D closure experiment."""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
S=importlib.util.spec_from_file_location("exp15fd",ROOT/"experiments"/"15F_D_memory_geometry_closure.py")
M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

def test_channel_families_are_distinct_and_causal():
    x=np.zeros(2048,dtype=complex); x[0]=1
    hs=[M.channel_kernel(f,12.0) for f in M.CHANNEL_FAMILIES]
    assert all(np.isfinite(h).all() for h in hs)
    assert all(h[0] >= 0 for h in hs)
    assert len({tuple(np.round(h[:128],12)) for h in hs}) == len(hs)
    assert np.isclose(sum(hs[0]),1.0)

def test_continuous_kernels_are_not_symbol_tap_only():
    for f in ("diffuse_exp","diffuse_gamma","continuous_multiscale"):
        h=M.channel_kernel(f,12.0)
        between=h[1:M.SPS]
        assert np.count_nonzero(np.abs(between)>1e-14) > 0

def test_horizon_fir_is_deterministic():
    g=M.U15E.budget_rates("long",12.0,8)
    assert M.min_horizon_samples(g)==M.min_horizon_samples(g)
    assert M.min_horizon_samples(g) > 8

def test_case_shape_and_metrics():
    rows=M.run_case("BPSK","multiscale","diffuse_exp",12.0,10.0,0,4)
    assert len(rows)==5
    assert all(np.isfinite(r["ber"]) for r in rows)
    assert {r["state_budget"] for r in rows}=={4}
    assert any(r["representation"].startswith("fir_horizon95") for r in rows)
