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
    assert history.rhos_clipped is not None
    assert history.predicted_decreases_safe is not None
    assert np.any(history.accepted)
    assert history.xs.shape[0] == 101
    np.testing.assert_allclose(history.xs[0], x0)


def test_controlled_adam_uses_scaled_gradient_fallback_for_non_descent_momentum() -> None:
    objective = AnisotropicQuadratic(curvature_x=1.0, curvature_y=1.0)
    x0 = np.array([1.0, 0.0])
    history = controlled_adam(
        objective,
        x0,
        alpha0=1.5,
        steps=2,
        kp=0.0,
        rho_star=0.5,
        alpha_max=2.0,
        max_backtracks=0,
    )

    assert history.gradient_fallback_used is not None
    assert not history.gradient_fallback_used[0]
    assert history.gradient_fallback_used[1]
    assert history.accepted is not None
    assert history.accepted[1]
    assert history.descent_scores is not None
    assert history.descent_scores[1] > 0.0
    assert abs(history.xs[2, 0]) < abs(history.xs[1, 0])


def test_controlled_adam_uses_clipped_rho_for_alpha_update() -> None:
    objective = AnisotropicQuadratic(curvature_x=1.0, curvature_y=1.0)
    x0 = np.array([1.0, 0.0])
    history = controlled_adam(
        objective,
        x0,
        alpha0=0.1,
        steps=2,
        kp=1.0,
        rho_star=0.0,
        alpha_max=10.0,
        rho_clip_max=0.1,
        max_backtracks=0,
    )

    assert history.rhos is not None
    assert history.rhos_clipped is not None
    assert history.rho_was_clipped is not None
    assert history.rhos[0] > 0.1
    assert history.rhos_clipped[0] == 0.1
    assert history.rho_was_clipped[0]
    np.testing.assert_allclose(history.alphas[0], 0.1)
    np.testing.assert_allclose(history.alphas[1], 0.1 * np.exp(0.1))
