import numpy as np

from min import fit_esprit, fit_matrix_pencil, fit_nnls, fit_vector_fitting
from min.kernels.nnls import log_rate_grid


def two_exponential(t):
    return 1.7 * np.exp(-2.0 * t) + 0.45 * np.exp(-11.0 * t)


def test_matrix_pencil_recovers_two_exponentials():
    dt = 0.01
    t = np.arange(120) * dt
    fit = fit_matrix_pencil(two_exponential(t), dt, order=2)
    order = np.argsort(fit.gammas.real)
    assert np.allclose(fit.gammas.real[order], [2.0, 11.0], atol=1e-5)
    assert np.allclose(fit.weights.real[order], [1.7, 0.45], atol=1e-5)


def test_esprit_recovers_two_exponentials():
    dt = 0.01
    t = np.arange(120) * dt
    fit = fit_esprit(two_exponential(t), dt, order=2)
    order = np.argsort(fit.gammas.real)
    assert np.allclose(fit.gammas.real[order], [2.0, 11.0], atol=1e-5)
    assert np.allclose(fit.weights.real[order], [1.7, 0.45], atol=1e-5)


def test_nnls_preserves_positive_structure():
    t = np.linspace(0.0, 4.0, 201)
    target = (1.0 + t / 0.5) ** (-0.7)
    fit = fit_nnls(target, t, log_rate_grid(0.05, 20.0, 8))
    assert np.all(fit.weights >= 0)
    assert np.all(fit.gammas > 0)
    assert fit.residual_norm >= 0


def test_vector_fitting_recovers_known_laplace_model():
    s = np.geomspace(0.1, 50.0, 100)
    response = 1.0 / (s + 2.0) + 0.5 / (s + 5.0)
    fit = fit_vector_fitting(
        s, response, order=2, initial_gammas=np.array([1.0, 8.0]), iterations=3
    )
    order = np.argsort(fit.poles.real)
    assert np.allclose(fit.poles.real[order], [-5.0, -2.0], atol=1e-8)
    assert np.allclose(fit.residues.real[order], [0.5, 1.0], atol=1e-8)
