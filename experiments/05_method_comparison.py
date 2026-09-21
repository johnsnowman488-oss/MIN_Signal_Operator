"""Matched exponential-identification benchmark for MIN power-law memory."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
from scipy.special import gamma, gammaincc
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from min import (
    fit_esprit, fit_matrix_pencil, fit_nnls, fit_prony, fit_vector_fitting,
    log_rate_grid, power_law_kernel,
)


def power_law_laplace(s: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    q = np.asarray(s, dtype=float) * tau
    return tau * np.exp(q) * q ** (alpha - 1.0) * gamma(1.0 - alpha) * gammaincc(1.0 - alpha, q)


def evaluate_modes(t, weights, gammas):
    return np.sum(weights[:, None] * np.exp(-gammas[:, None] * t[None, :]), axis=0)


def trapezoid_convolution(t, x, kernel_values):
    dt = t[1] - t[0]
    conv = fftconvolve(kernel_values, x)[:t.size]
    return dt * (conv - 0.5 * kernel_values[0] * x - 0.5 * x[0] * kernel_values)


def mode_flags(weights, gammas):
    tol = 1e-8
    return {
        "stable": bool(np.all(np.real(gammas) > 0)),
        "real_modes": bool(np.max(np.abs(np.imag(gammas))) < tol),
        "nonnegative_weights": bool(
            np.max(np.abs(np.imag(weights))) < tol and np.min(np.real(weights)) >= -tol
        ),
    }


def run():
    dt = 0.02
    t = np.arange(401) * dt
    alpha, tau = 0.7, 0.5
    kernel = power_law_kernel(t, alpha=alpha, tau=tau)
    x = np.sin(2 * np.pi * 0.4 * t) + 0.35 * np.sin(2 * np.pi * 1.7 * t)
    reference = trapezoid_convolution(t, x, kernel)
    orders = [2, 4, 8, 16]
    s = np.geomspace(0.05, 30.0, 120)
    H = power_law_laplace(s, alpha, tau)
    results = []

    for method in ("prony", "matrix_pencil", "esprit", "vector_fitting", "nnls"):
        for order in orders:
            start = time.perf_counter()
            status, exc_text = "ok", ""
            try:
                if method == "prony":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = fit_prony(kernel, dt, order)
                elif method == "matrix_pencil":
                    fit = fit_matrix_pencil(kernel, dt, order)
                elif method == "esprit":
                    fit = fit_esprit(kernel, dt, order)
                elif method == "vector_fitting":
                    fit = fit_vector_fitting(
                        s, H, order,
                        initial_gammas=log_rate_grid(0.05, 30.0, order),
                        iterations=10,
                    )
                else:
                    fit = fit_nnls(kernel, t, log_rate_grid(0.05, 20.0, order))

                weights, gammas = np.asarray(fit.weights), np.asarray(fit.gammas)
                fitted_kernel = evaluate_modes(t, weights, gammas)
                ek = float(np.linalg.norm(kernel - fitted_kernel) / np.linalg.norm(kernel))
                fitted_output = trapezoid_convolution(t, x, fitted_kernel)
                em = float(np.linalg.norm(reference - fitted_output) / np.linalg.norm(reference))
                flags = mode_flags(weights, gammas)
                admissible = flags["stable"] and flags["real_modes"] and flags["nonnegative_weights"]
            except Exception as exc:
                status = "failed"
                exc_text = f"{type(exc).__name__}: {exc}"
                ek = em = float("nan")
                flags = {"stable": False, "real_modes": False, "nonnegative_weights": False}
                admissible = False

            results.append({
                "method": method, "order": order,
                "kernel_rel": ek, "operator_rel": em,
                "runtime_ms": (time.perf_counter() - start) * 1000.0,
                "stable": flags["stable"], "real_modes": flags["real_modes"],
                "nonnegative_weights": flags["nonnegative_weights"],
                "positive_real_soe": admissible,
                "status": status, "error": exc_text,
            })

    for row in results:
        print(
            f"{row['method']:15s} {row['order']:2d} "
            f"{row['kernel_rel']:11.4e} {row['operator_rel']:12.4e} "
            f"{row['status']}"
        )

    out = ROOT / "experiments" / "05_method_comparison_results.json"
    out.write_text(json.dumps({
        "reference": {
            "alpha": alpha, "tau": tau, "dt": dt,
            "horizon": float(t[-1]), "samples": int(t.size),
            "laplace_samples": int(s.size),
        },
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    run()
