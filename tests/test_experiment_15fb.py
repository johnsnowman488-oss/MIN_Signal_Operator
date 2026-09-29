"""Tests for 15F-B channel stress."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp15fb", ROOT / "experiments" / "15F_B_channel_stress.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_15f_b_uses_15e_set():
    assert MOD.BUDGET == 16
    reps, _, _ = MOD.U15E.representation_specs("short", 4.0, np.ones(128, dtype=complex), 16)
    assert set(reps) == {"min_16", "iir_logspread_16", "dense_ss_16", "fir_16"}

def test_channel_profiles_are_finite():
    x = np.ones(512, dtype=complex)
    for profile in MOD.CHANNEL_PROFILES:
        y = MOD.channel(x, "short", 4.0, profile, 0)
        assert y.shape == x.shape
        assert np.isfinite(y).all()
        assert np.mean(np.abs(y) ** 2) > 0.0

def test_16qam_bits_and_decision():
    s = np.asarray([(-3-3j), (-3+3j), (3-3j), (3+3j)]) / np.sqrt(10.0)
    assert np.allclose(MOD.decide("16QAM", s), s)
    assert MOD.bits("16QAM", s).shape == (4, 4)

def test_case_shape_and_metrics():
    rows = MOD.run_case("QPSK", "multiscale", 12.0, 10.0, "multipath_mild", 0)
    assert len(rows) == 5
    assert {r["state_budget"] for r in rows} == {1, 16}
    assert all(np.isfinite(r["ber"]) for r in rows)
    assert all(np.isfinite(r["ser"]) for r in rows)
    assert all(np.isfinite(r["evm_percent"]) for r in rows)
