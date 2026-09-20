import numpy as np

from min import MemoryOperator, SOEMemory, fit_prony


def test_prony_recovers_two_decay_rates():
    dt = 0.01
    t = np.arange(80) * dt
    true_weights = np.array([1.7, 0.45])
    true_gammas = np.array([2.0, 11.0])
    samples = sum(w * np.exp(-g * t) for w, g in zip(true_weights, true_gammas))

    fit = fit_prony(samples, dt, order=2)
    order = np.argsort(fit.gammas.real)

    assert np.allclose(fit.gammas.real[order], true_gammas, atol=1e-6)
    assert np.allclose(fit.weights.real[order], true_weights, atol=1e-6)


def test_soe_matches_direct_memory_operator():
    dt = 0.005
    t = np.arange(501) * dt
    weights = np.array([1.2, 0.35])
    gammas = np.array([3.0, 18.0])
    x = np.sin(2 * np.pi * 1.5 * t)

    direct = MemoryOperator(
        lambda lag: sum(w * np.exp(-g * lag) for w, g in zip(weights, gammas))
    ).apply(t, x)

    streaming = SOEMemory(weights, gammas).apply(t, x)

    assert np.max(np.abs(direct - streaming)) < 2e-4
