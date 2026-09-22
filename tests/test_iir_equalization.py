import numpy as np
import pytest

from min.metrics.equalization import apply_iir_equalizer, design_iir_equalizer, design_fixed_denominator_equalizer


def test_iir_round_trip_on_synthetic_stable_recursion():
    rng = np.random.default_rng(4)
    received = rng.normal(size=256) + 1j * rng.normal(size=256)
    b_true = np.array([1.1 + 0.1j, -0.2 + 0.05j])
    a_true = np.array([0.25 - 0.05j, -0.08 + 0.02j])
    desired = apply_iir_equalizer(received, b_true, a_true)

    b, a, radius = design_iir_equalizer(
        received[:192], desired[:192],
        feedforward=2, feedback=2, ridge=1e-10, pole_radius_limit=0.98,
    )
    # Preserve recursive state across the train/test boundary by applying the
    # fitted receiver to the complete stream before evaluating the held-out tail.
    estimate = apply_iir_equalizer(received, b, a)[192:]

    assert radius < 0.98 + 1e-10
    assert np.mean(np.abs(estimate - desired[192:]) ** 2) < 1e-5


def test_iir_validates_orders():
    x = np.ones(16, dtype=complex)
    with pytest.raises(ValueError):
        design_iir_equalizer(x, x, feedforward=0, feedback=2)
    with pytest.raises(ValueError):
        design_iir_equalizer(x, x, feedforward=2, feedback=0)


def test_iir_stabilizes_feedback_recursion():
    rng = np.random.default_rng(8)
    received = rng.normal(size=128) + 1j * rng.normal(size=128)
    desired = np.zeros_like(received)
    desired[0] = 1.0
    desired[1:] = 1.4 * desired[:-1] + 0.01 * received[1:]

    _, _, radius = design_iir_equalizer(
        received, desired, feedforward=2, feedback=1,
        ridge=1e-8, pole_radius_limit=0.95,
    )
    assert radius <= 0.95 + 1e-10


def test_fixed_denominator_equalizer_recovers_known_numerator():
    rng = np.random.default_rng(12)
    received = rng.normal(size=256) + 1j * rng.normal(size=256)
    feedback = np.array([0.35 + 0.05j])
    b_true = np.array([1.2 - 0.1j, -0.25 + 0.03j])
    desired = apply_iir_equalizer(received, b_true, feedback)
    b = design_fixed_denominator_equalizer(
        received[:192], desired[:192], feedback,
        numerator_order=2, ridge=1e-10,
    )
    estimate = apply_iir_equalizer(received, b, feedback)
    assert np.mean(np.abs(estimate[192:] - desired[192:]) ** 2) < 1e-6
