"""Tests for Experiment 15F-A communications helpers."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp15f", ROOT / "experiments" / "15F_A_synthetic_communications.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_channel_memory_is_causal_and_dc_preserving():
    x = np.ones(64, dtype=complex)
    y = MOD.channel_memory(x, 0.02)
    assert np.allclose(y[-10:], 1.0)
    z = np.zeros_like(x); z[20:] = 1.0
    yz = MOD.channel_memory(z, 0.02)
    assert np.allclose(yz[:20], 0.0)

def test_min_state_dimensions_and_finiteness():
    x = np.ones(128, dtype=complex)
    for budget in MOD.BUDGETS:
        state = MOD.min_states(x, 0.02, budget)
        assert state.shape == (128, budget)
        assert np.isfinite(state).all()

def test_decision_and_evm_metrics_are_well_defined():
    ref = np.array([1+0j, -1+0j, 1+0j, -1+0j])
    est = ref.copy()
    assert MOD.ber("BPSK", ref, est) == 0.0
    assert MOD.ser("BPSK", ref, est) == 0.0
    qref = np.ones(4, dtype=complex) * np.exp(1j*np.pi/4)
    assert MOD.evm("QPSK", qref, qref) == 0.0

def test_case_row_count_and_holdout():
    rows = MOD.run_case("BPSK", "tau_0.020", 10.0, 0)
    assert len(rows) == 1 + 3 * len(MOD.BUDGETS)
    assert all(r["train_symbols"] == MOD.TRAIN_SYMBOLS for r in rows)
    assert all(r["test_symbols"] == MOD.NUM_SYMBOLS - MOD.TRAIN_SYMBOLS for r in rows)
