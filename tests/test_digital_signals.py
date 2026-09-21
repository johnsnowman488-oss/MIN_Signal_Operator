import numpy as np

from min.signals import generate_16qam, generate_bpsk, generate_qpsk


def test_digital_generators_are_reproducible():
    for generator in (generate_bpsk, generate_qpsk, generate_16qam):
        a = generator(64, samples_per_symbol=8, seed=17)
        b = generator(64, samples_per_symbol=8, seed=17)
        assert np.array_equal(a.symbols, b.symbols)
        assert np.array_equal(a.record.samples, b.record.samples)


def test_constellation_average_power():
    for generator in (generate_bpsk, generate_qpsk, generate_16qam):
        signal = generator(10000, samples_per_symbol=2, seed=1)
        assert np.isclose(np.mean(np.abs(signal.symbols) ** 2), 1.0, atol=0.03)


def test_symbol_indices_are_inside_waveform():
    signal = generate_qpsk(20, samples_per_symbol=8)
    assert signal.symbol_indices[-1] < signal.record.samples.size
    assert np.all(signal.record.samples[signal.symbol_indices] == signal.symbols)
