"""Tests for toy objectives and optimizers."""

import numpy as np

from controlled_adam.objectives import (
    Ackley,
    AnisotropicQuadratic,
    Beale,
    Easom,
    GoldsteinPrice,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
    SixHumpCamel,
)
from controlled_adam.optimizers import controlled_adam, vanilla_adam


def test_quadratic_value_and_gradient() -> None:
    objective = AnisotropicQuadratic()
    x = np.array([2.0, 3.0])
    assert np.isclose(objective.value(x), 0.5 * (50.0 * 4.0 + 9.0))
    np.testing.assert_allclose(objective.gradient(x), np.array([100.0, 3.0]))


def test_rosenbrock_gradient_finite_difference() -> None:
    objective = Rosenbrock()
    x = np.array([-1.2, 1.1])
    g = objective.gradient(x)
    h = 1e-6
    fd = np.zeros_like(x)
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = h
        fd[i] = (objective.value(x + step) - objective.value(x - step)) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-5)


def test_benchmark_gradients_finite_difference() -> None:
    cases = [
        (Himmelblau(), np.array([-3.5, 0.5])),
        (Rastrigin(), np.array([3.3, 2.8])),
        (Beale(), np.array([1.0, 1.0])),
        (Ackley(), np.array([1.2, -0.7])),
        (SixHumpCamel(), np.array([0.4, -0.8])),
        (GoldsteinPrice(), np.array([0.2, -0.8])),
        (Easom(), np.array([2.8, 3.3])),
    ]

    h = 1e-6
    for objective, x in cases:
        fd = np.zeros_like(x)
        for i in range(len(x)):
            step = np.zeros_like(x)
            step[i] = h
            fd[i] = (objective.value(x + step) - objective.value(x - step)) / (2 * h)
        np.testing.assert_allclose(objective.gradient(x), fd, rtol=1e-5, atol=1e-5)


def test_vanilla_adam_reduces_quadratic() -> None:
    objective = AnisotropicQuadratic()
    x0 = np.array([2.0, 2.0])
    history = vanilla_adam(objective, x0, alpha=0.04, steps=100)
    assert history.fs[-1] < objective.value(x0)
    assert history.xs.shape[0] == 101
    np.testing.assert_allclose(history.xs[0], x0)


def test_controlled_adam_reduces_quadratic_and_keeps_alpha_positive() -> None:
    objective = AnisotropicQuadratic()
    x0 = np.array([2.0, 2.0])
    history = controlled_adam(
        objective,
        x0,
        alpha0=0.04,
        steps=100,
        kp=0.25,
        rho_star=0.5,
        alpha_max=0.5,
    )
    assert history.fs[-1] < objective.value(x0)
    assert np.all(history.alphas > 0)
    assert history.rhos is not None
    assert np.any(history.accepted)
    assert history.xs.shape[0] == 101
    np.testing.assert_allclose(history.xs[0], x0)
