"""Tests for corrected 15F-A and its 15E lineage."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp15f", ROOT / "experiments" / "15F_A_synthetic_communications.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_15f_imports_15e():
    assert MOD.U15E.__file__.endswith("15E_state_budget_sweep.py")
    assert tuple(MOD.BUDGETS) == (1, 2, 4, 8, 16)

def test_15e_representation_set_is_preserved():
    reps, _, _ = MOD.U15E.representation_specs("short", 4.0, np.ones(64, dtype=complex), 4)
    assert set(reps) == {"min_4", "iir_logspread_4", "dense_ss_4", "fir_4"}

def test_channel_is_causal_and_dc_preserving():
    x = np.ones(256, dtype=complex)
    y = MOD.channel_memory(x, "short", 4.0)
    assert np.allclose(y[-32:], 1.0, atol=1e-10)
    z = np.zeros_like(x); z[80:] = 1.0
    yz = MOD.channel_memory(z, "short", 4.0)
    assert np.allclose(yz[:80], 0.0)

def test_metrics_zero_for_perfect_decisions():
    b = np.array([1+0j, -1+0j, 1+0j, -1+0j])
    assert MOD.ber("BPSK", b, b) == 0.0
    assert MOD.ser("BPSK", b, b) == 0.0
    q = np.exp(1j*(np.pi/4 + np.arange(4)*np.pi/2))
    assert MOD.ber("QPSK", q, q) == 0.0
    assert MOD.ser("QPSK", q, q) == 0.0

def test_case_shape_and_holdout():
    rows = MOD.run_case("BPSK", "short", 4.0, 10.0, 0)
    assert len(rows) == 1 + 4 * len(MOD.BUDGETS)
    assert all(r["train_symbols"] == MOD.NUM_TRAIN_SYMBOLS for r in rows)
    assert all(r["test_symbols"] == MOD.NUM_TEST_SYMBOLS for r in rows)
