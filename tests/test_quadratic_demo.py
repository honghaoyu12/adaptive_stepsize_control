"""Basic tests for the adaptive step-size demo."""

import numpy as np

from adaptive_stepsize_control.objectives import (
    BealeObjective,
    HimmelblauObjective,
    QuadraticObjective,
    RastriginObjective,
    RosenbrockObjective,
)
from adaptive_stepsize_control.optimizers import (
    controlled_gradient_descent,
    stochastic_gradient_descent,
)


def test_quadratic_value_and_gradient() -> None:
    objective = QuadraticObjective.anisotropic_2d()
    x = np.array([2.0, 3.0])

    assert np.isclose(objective.value(x), 0.5 * (50.0 * 4.0 + 9.0))
    np.testing.assert_allclose(objective.gradient(x), np.array([100.0, 3.0]))


def test_controlled_gradient_descent_reduces_objective() -> None:
    objective = QuadraticObjective.anisotropic_2d()
    x0 = np.array([2.0, 2.0])

    history = controlled_gradient_descent(
        objective=objective,
        x0=x0,
        eta0=0.08,
        steps=60,
        kp=0.7,
        rho_star=0.8,
        eta_min=1e-8,
        eta_max=0.2,
        reject_bad_steps=True,
    )

    assert history.fs[-1] < objective.value(x0)
    assert np.all(history.etas > 0)
    assert history.rhos is not None


def test_stochastic_gradient_descent_is_reproducible() -> None:
    objective = QuadraticObjective.anisotropic_2d()
    x0 = np.array([2.0, 2.0])

    first = stochastic_gradient_descent(
        objective=objective,
        x0=x0,
        eta=0.02,
        steps=60,
        gradient_noise_scale=0.05,
        seed=123,
    )
    second = stochastic_gradient_descent(
        objective=objective,
        x0=x0,
        eta=0.02,
        steps=60,
        gradient_noise_scale=0.05,
        seed=123,
    )

    np.testing.assert_allclose(first.xs, second.xs)
    assert first.fs[-1] < objective.value(x0)


def test_benchmark_objective_gradients_match_finite_differences() -> None:
    cases = [
        (RosenbrockObjective(), np.array([-1.2, 1.0])),
        (HimmelblauObjective(), np.array([-3.5, 0.5])),
        (RastriginObjective(), np.array([3.3, 2.8])),
        (BealeObjective(), np.array([1.0, 1.0])),
    ]

    eps = 1e-6
    for objective, x in cases:
        finite_difference_gradient = np.zeros_like(x)
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = eps
            finite_difference_gradient[i] = (
                objective.value(x + step) - objective.value(x - step)
            ) / (2.0 * eps)

        np.testing.assert_allclose(
            objective.gradient(x),
            finite_difference_gradient,
            rtol=1e-5,
            atol=1e-5,
        )
