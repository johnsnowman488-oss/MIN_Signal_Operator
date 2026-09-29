"""Tests for Experiment 15F-C communications budget scaling."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp15fc", ROOT / "experiments" / "15F_C_communications_budget_scaling.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_15f_c_inherits_15e_budget_set_and_representations():
    assert tuple(MOD.BUDGETS) == (1, 2, 4, 8, 16)
    reps, _, _ = MOD.U15E.representation_specs(
        "multiscale", 12.0, np.ones(128, dtype=complex), 8
    )
    assert set(reps) == {"min_8", "iir_logspread_8", "dense_ss_8", "fir_8"}


def test_channel_memory_is_causal_and_finite():
    x = np.ones(512, dtype=complex)
    y = MOD.channel_memory(x, "short", 4.0)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert np.allclose(y[-64:], 1.0, atol=1e-10)

    z = np.zeros_like(x)
    z[100:] = 1.0
    yz = MOD.channel_memory(z, "short", 4.0)
    assert np.allclose(yz[:100], 0.0)


def test_metrics_are_zero_for_perfect_decisions():
    b = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
    assert MOD.ber("BPSK", b, b) == 0.0
    assert MOD.ser("BPSK", b, b) == 0.0

    q = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    assert np.allclose(MOD.decide("QPSK", q), q)
    assert MOD.ber("QPSK", q, q) == 0.0
    assert MOD.ser("QPSK", q, q) == 0.0


def test_case_and_budget_delta_shapes():
    rows = MOD.run_case("QPSK", "multiscale", 12.0, 10.0, 0)
    assert len(rows) == 1 + 4 * len(MOD.BUDGETS)
    assert {r["state_budget"] for r in rows} == {1, 2, 4, 8, 16}
    assert all(np.isfinite(r["ber"]) for r in rows)
    assert all(np.isfinite(r["ser"]) for r in rows)
    assert all(np.isfinite(r["evm_percent"]) for r in rows)
    assert all(np.isfinite(r["nmse"]) for r in rows)

    deltas = MOD.budget_deltas(rows)
    assert len(deltas) == 4 * 4
    assert {tuple((d["n_from"], d["n_to"])) for d in deltas} == {
        (1, 2), (2, 4), (4, 8), (8, 16)
    }
