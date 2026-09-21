import numpy as np

from min.signals import (
    generate_gaussian_pulse,
    generate_impulse,
    generate_linear_chirp,
    generate_multitone,
    generate_rectangular_pulse,
    generate_sine,
    make_time_axis,
    unit_energy,
)
from min.metrics import correlation, delay_of_max_correlation, nmse


def test_time_axis_and_unit_energy():
    t = make_time_axis(1.0, 100.0)
    assert t.size == 101
    assert t[-1] == 1.0
    x = np.array([1.0, -2.0, 2.0])
    assert np.isclose(np.sum(np.abs(unit_energy(x)) ** 2), 1.0)


def test_elementary_generators_return_finite_records():
    records = [
        generate_impulse(1.0, 100.0),
        generate_rectangular_pulse(1.0, 100.0, start=0.2, width=0.2),
        generate_gaussian_pulse(1.0, 100.0, center=0.5, sigma=0.08),
        generate_sine(1.0, 100.0, frequency=5.0),
        generate_multitone(1.0, 100.0, frequencies=[5.0, 17.0], amplitudes=[1.0, 0.3]),
        generate_linear_chirp(1.0, 100.0, f0=3.0, f1=25.0),
    ]
    for record in records:
        assert record.samples.ndim == 1
        assert np.isfinite(record.samples).all()
        assert record.sample_rate == 100.0


def test_complex_baseband_generation():
    record = generate_sine(1.0, 1000.0, frequency=20.0, amplitude=1j)
    assert np.iscomplexobj(record.samples)


def test_signal_metrics_have_expected_invariants():
    x = np.array([1.0, 2.0, 3.0])
    assert nmse(x, x) == 0.0
    assert correlation(x, x) == 1.0
    assert delay_of_max_correlation(x, x) == 0
