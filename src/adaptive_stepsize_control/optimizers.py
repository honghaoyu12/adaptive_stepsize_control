"""Gradient-descent optimizers and adaptive step-size controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_stepsize_control.objectives import ObjectiveFunction


@dataclass
class OptimizationHistory:
    """Stores the trajectory and diagnostic quantities from optimization."""

    xs: np.ndarray
    fs: np.ndarray
    etas: np.ndarray
    rhos: Optional[np.ndarray] = None
    predicted_decreases: Optional[np.ndarray] = None
    actual_decreases: Optional[np.ndarray] = None
    prediction_errors: Optional[np.ndarray] = None
    accepted: Optional[np.ndarray] = None


def fixed_gradient_descent(
    objective: ObjectiveFunction,
    x0: np.ndarray,
    eta: float,
    steps: int,
) -> OptimizationHistory:
    """Run gradient descent with a fixed learning rate.

    Parameters
    ----------
    objective:
        Objective with ``value`` and ``gradient`` methods.
    x0:
        Initial point.
    eta:
        Fixed learning rate.
    steps:
        Number of iterations.
    """
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    x = np.asarray(x0, dtype=float).copy()

    xs = [x.copy()]
    fs = [objective.value(x)]
    etas = []

    for _ in range(steps):
        g = objective.gradient(x)
        etas.append(eta)

        x = x - eta * g

        xs.append(x.copy())
        fs.append(objective.value(x))

    return OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        etas=np.asarray(etas),
    )


def stochastic_gradient_descent(
    objective: ObjectiveFunction,
    x0: np.ndarray,
    eta: float,
    steps: int,
    gradient_noise_scale: float = 0.1,
    seed: int | None = 0,
) -> OptimizationHistory:
    """Run gradient descent with noisy stochastic gradient estimates.

    The stochastic gradient is modeled as

        grad f(x_t) + noise_t,

    where ``noise_t`` is zero-mean Gaussian noise. The noise standard deviation
    scales with ``max(1, ||grad f(x_t)||)`` so the stochasticity remains visible
    near flat regions without overwhelming large-gradient regions.
    """
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if gradient_noise_scale < 0:
        raise ValueError("gradient_noise_scale must be non-negative.")

    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()

    xs = [x.copy()]
    fs = [objective.value(x)]
    etas = []

    for _ in range(steps):
        g = objective.gradient(x)
        noise_std = gradient_noise_scale * max(1.0, float(np.linalg.norm(g)))
        stochastic_g = g + rng.normal(loc=0.0, scale=noise_std, size=g.shape)

        etas.append(eta)

        x = x - eta * stochastic_g

        xs.append(x.copy())
        fs.append(objective.value(x))

    return OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        etas=np.asarray(etas),
    )


def controlled_gradient_descent(
    objective: ObjectiveFunction,
    x0: np.ndarray,
    eta0: float,
    steps: int,
    kp: float = 0.7,
    rho_star: float = 0.8,
    eta_min: float = 1e-8,
    eta_max: float = 1.0,
    reject_bad_steps: bool = True,
    eps: float = 1e-12,
) -> OptimizationHistory:
    """Run gradient descent with proportional feedback step-size control.

    The trial step is

        x_trial = x_t - eta_t * grad f(x_t).

    The first-order predicted decrease is

        eta_t * ||grad f(x_t)||^2.

    The actual-over-predicted decrease ratio is

        rho_t = actual_decrease / predicted_decrease.

    The learning rate is then updated by

        eta_{t+1} = eta_t * exp(kp * (rho_t - rho_star)).

    Parameters
    ----------
    objective:
        Objective with ``value`` and ``gradient`` methods.
    x0:
        Initial point.
    eta0:
        Initial learning rate.
    steps:
        Number of iterations.
    kp:
        Proportional gain.
    rho_star:
        Target actual-over-predicted decrease ratio.
    eta_min, eta_max:
        Bounds on the adaptive learning rate.
    reject_bad_steps:
        If True, reject trial steps that increase the objective.
    eps:
        Small positive number for numerical stability.
    """
    if eta0 <= 0:
        raise ValueError("eta0 must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if kp < 0:
        raise ValueError("kp must be non-negative.")
    if eta_min <= 0 or eta_max <= 0 or eta_min > eta_max:
        raise ValueError("eta bounds must satisfy 0 < eta_min <= eta_max.")

    x = np.asarray(x0, dtype=float).copy()
    eta = float(np.clip(eta0, eta_min, eta_max))

    xs = [x.copy()]
    fs = [objective.value(x)]
    etas = []
    rhos = []
    predicted_decreases = []
    actual_decreases = []
    prediction_errors = []
    accepted = []

    for _ in range(steps):
        f_t = objective.value(x)
        g = objective.gradient(x)

        x_trial = x - eta * g
        f_trial = objective.value(x_trial)

        predicted_decrease = eta * float(np.dot(g, g))
        actual_decrease = f_t - f_trial

        rho = actual_decrease / (predicted_decrease + eps)
        f_hat_trial = f_t - predicted_decrease
        prediction_error = f_trial - f_hat_trial

        step_is_accepted = (not reject_bad_steps) or (rho > 0)
        if step_is_accepted:
            x = x_trial

        xs.append(x.copy())
        fs.append(objective.value(x))
        etas.append(eta)
        rhos.append(rho)
        predicted_decreases.append(predicted_decrease)
        actual_decreases.append(actual_decrease)
        prediction_errors.append(prediction_error)
        accepted.append(step_is_accepted)

        eta = eta * float(np.exp(kp * (rho - rho_star)))
        eta = float(np.clip(eta, eta_min, eta_max))

    return OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        etas=np.asarray(etas),
        rhos=np.asarray(rhos),
        predicted_decreases=np.asarray(predicted_decreases),
        actual_decreases=np.asarray(actual_decreases),
        prediction_errors=np.asarray(prediction_errors),
        accepted=np.asarray(accepted),
    )
