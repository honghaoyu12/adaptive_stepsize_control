"""Vanilla Adam and outer-loop controlled Adam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from controlled_adam.objectives import Objective


@dataclass
class OptimizationHistory:
    """Optimization trajectory and diagnostics."""

    xs: np.ndarray
    fs: np.ndarray
    alphas: np.ndarray
    grad_norms: np.ndarray
    rhos: Optional[np.ndarray] = None
    predicted_decreases: Optional[np.ndarray] = None
    actual_decreases: Optional[np.ndarray] = None
    accepted: Optional[np.ndarray] = None
    descent_scores: Optional[np.ndarray] = None
    gradient_fallback_used: Optional[np.ndarray] = None
    rhos_clipped: Optional[np.ndarray] = None
    predicted_decreases_safe: Optional[np.ndarray] = None
    predicted_was_floored: Optional[np.ndarray] = None
    rho_was_clipped: Optional[np.ndarray] = None
    direction_types: Optional[np.ndarray] = None


def _validate_common(steps: int, alpha: float) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if alpha <= 0:
        raise ValueError("learning-rate parameter must be positive.")


def _measure_rho(
    *,
    actual_decrease: float,
    predicted_decrease_raw: float,
    loss_before_value: float,
    eps: float,
    absolute_predicted_floor: float,
    relative_predicted_floor: float,
    rho_clip_min: float,
    rho_clip_max: float,
) -> tuple[float, float, float, bool, bool]:
    """Return measured/clipped rho and denominator diagnostics."""
    predicted_floor = max(
        eps,
        absolute_predicted_floor,
        relative_predicted_floor * abs(loss_before_value),
    )
    predicted_decrease_safe = max(predicted_decrease_raw, predicted_floor)
    predicted_was_floored = predicted_decrease_safe > predicted_decrease_raw
    rho_measured = actual_decrease / predicted_decrease_safe
    rho_clipped = float(np.clip(rho_measured, rho_clip_min, rho_clip_max))
    rho_was_clipped = bool(rho_clipped != rho_measured)
    return (
        rho_measured,
        rho_clipped,
        predicted_decrease_safe,
        predicted_was_floored,
        rho_was_clipped,
    )


def vanilla_adam(
    objective: Objective,
    x0: np.ndarray,
    alpha: float = 1e-2,
    steps: int = 1000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> OptimizationHistory:
    """Run vanilla Adam with a fixed global learning rate.

    Parameters
    ----------
    objective:
        Deterministic objective with value and gradient methods.
    x0:
        Initial point.
    alpha:
        Global Adam learning rate.
    steps:
        Number of optimizer steps.
    beta1, beta2:
        Adam exponential moving average coefficients.
    eps:
        Numerical stabilizer in the denominator.
    """
    _validate_common(steps, alpha)

    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    xs = [x.copy()]
    fs = [objective.value(x)]
    alphas = []
    grad_norms = []

    for t in range(1, steps + 1):
        g = objective.gradient(x)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        p = -m_hat / (np.sqrt(v_hat) + eps)
        x = x + alpha * p

        xs.append(x.copy())
        fs.append(objective.value(x))
        alphas.append(alpha)
        grad_norms.append(float(np.linalg.norm(g)))

    return OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        alphas=np.asarray(alphas),
        grad_norms=np.asarray(grad_norms),
    )


def controlled_adam(
    objective: Objective,
    x0: np.ndarray,
    alpha0: float = 1e-2,
    steps: int = 1000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    kp: float = 0.2,
    rho_star: float = 0.5,
    rho_min: float = 0.0,
    alpha_min: float = 1e-8,
    alpha_max: float = 1.0,
    non_descent_shrink: float = 0.5,
    reject_bad_steps: bool = True,
    max_backtracks: int = 1,
    backtrack_shrink: float = 0.5,
    absolute_predicted_floor: float = 1e-12,
    relative_predicted_floor: float = 1e-8,
    rho_clip_min: float = -1.0,
    rho_clip_max: float = 3.0,
) -> OptimizationHistory:
    """Run Adam with an outer-loop global step-size controller.

    Adam proposes a direction

        p_t = -m_hat_t / (sqrt(v_hat_t) + eps).

    The outer loop takes a trial step

        x_trial = x_t + alpha_t p_t,

    evaluates the actual decrease, compares it with the first-order predicted
    decrease

        predicted_decrease = -alpha_t * grad_t^T p_t,

    and updates alpha_t using

        alpha_{t+1} = alpha_t * exp(kp * (rho_t - rho_star)).

    If the Adam momentum direction is not a descent direction for the current
    gradient, the trial direction falls back to the negative gradient rescaled
    to the Adam direction norm. The same ``kp`` is used for both increases and
    decreases, so the controller has one proportional gain to tune.

    This implementation optionally performs a small amount of backtracking if
    the first proposed step is rejected. This makes the method closer to a
    line-search/trust-region method while preserving the outer-loop controller.
    Acceptance uses measured rho; alpha control uses clipped rho after a
    predicted-decrease floor has stabilized the denominator.
    """
    _validate_common(steps, alpha0)
    if kp < 0:
        raise ValueError("kp must be non-negative.")
    if not (0.0 < alpha_min <= alpha_max):
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")
    if not (0.0 < non_descent_shrink < 1.0):
        raise ValueError("non_descent_shrink must be in (0, 1).")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative.")
    if not (0.0 < backtrack_shrink < 1.0):
        raise ValueError("backtrack_shrink must be in (0, 1).")
    if absolute_predicted_floor <= 0.0:
        raise ValueError("absolute_predicted_floor must be positive.")
    if relative_predicted_floor < 0.0:
        raise ValueError("relative_predicted_floor must be non-negative.")
    if rho_clip_min >= rho_clip_max:
        raise ValueError("rho_clip_min must be < rho_clip_max.")

    x = np.asarray(x0, dtype=float).copy()
    alpha = float(np.clip(alpha0, alpha_min, alpha_max))
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    xs = [x.copy()]
    fs = [objective.value(x)]
    alphas = []
    grad_norms = []
    rhos = []
    predicted_decreases = []
    actual_decreases = []
    accepted = []
    descent_scores = []
    gradient_fallback_used = []
    rhos_clipped = []
    predicted_decreases_safe = []
    predicted_was_floored = []
    rho_was_clipped = []
    direction_types = []

    for t in range(1, steps + 1):
        f_t = objective.value(x)
        g = objective.gradient(x)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        p = -m_hat / (np.sqrt(v_hat) + eps)

        # Predicted decrease per unit alpha. This must be positive for p to
        # be a descent direction according to the current gradient.
        descent_score = -float(np.dot(g, p))
        used_gradient_fallback = False
        direction_type = "adam"

        rho = np.nan
        rho_clipped = np.nan
        predicted_decrease = 0.0
        predicted_decrease_safe = 0.0
        was_floored = False
        was_clipped = False
        actual_decrease = 0.0
        step_accepted = False
        alpha_used = alpha

        if descent_score <= 0.0:
            g_norm = float(np.linalg.norm(g))
            p_norm = float(np.linalg.norm(p))
            if g_norm > 0.0 and p_norm > 0.0:
                p = -g * (p_norm / (g_norm + eps))
                descent_score = -float(np.dot(g, p))
                used_gradient_fallback = True
                direction_type = "gradient_fallback"

        if descent_score <= 0.0:
            direction_type = "degenerate_skip"
            # Degenerate case: neither Adam nor the raw gradient gives a usable
            # trial direction. Keep the old conservative shrink-and-stay-put
            # behavior.
            alpha = float(np.clip(alpha * non_descent_shrink, alpha_min, alpha_max))
        else:
            trial_alpha = alpha
            # Try the current alpha, then shrink if the step is rejected.
            for _ in range(max_backtracks + 1):
                x_candidate = x + trial_alpha * p
                f_candidate = objective.value(x_candidate)
                candidate_predicted = trial_alpha * descent_score
                candidate_actual = f_t - f_candidate
                (
                    candidate_rho,
                    candidate_rho_clipped,
                    candidate_predicted_safe,
                    candidate_was_floored,
                    candidate_was_clipped,
                ) = _measure_rho(
                    actual_decrease=candidate_actual,
                    predicted_decrease_raw=candidate_predicted,
                    loss_before_value=f_t,
                    eps=eps,
                    absolute_predicted_floor=absolute_predicted_floor,
                    relative_predicted_floor=relative_predicted_floor,
                    rho_clip_min=rho_clip_min,
                    rho_clip_max=rho_clip_max,
                )

                if (not reject_bad_steps) or (candidate_rho > rho_min):
                    predicted_decrease = candidate_predicted
                    predicted_decrease_safe = candidate_predicted_safe
                    actual_decrease = candidate_actual
                    rho = candidate_rho
                    rho_clipped = candidate_rho_clipped
                    was_floored = candidate_was_floored
                    was_clipped = candidate_was_clipped
                    alpha_used = trial_alpha
                    step_accepted = True
                    x = x_candidate
                    break

                # Save the last attempted diagnostics even if all attempts fail.
                predicted_decrease = candidate_predicted
                predicted_decrease_safe = candidate_predicted_safe
                actual_decrease = candidate_actual
                rho = candidate_rho
                rho_clipped = candidate_rho_clipped
                was_floored = candidate_was_floored
                was_clipped = candidate_was_clipped
                alpha_used = trial_alpha
                trial_alpha = max(alpha_min, trial_alpha * backtrack_shrink)

            if step_accepted:
                error = rho_clipped - rho_star
                alpha = alpha_used * float(np.exp(kp * error))
            else:
                # If every trial failed, keep x fixed and shrink alpha.
                alpha = alpha_used * backtrack_shrink

            alpha = float(np.clip(alpha, alpha_min, alpha_max))

        xs.append(x.copy())
        fs.append(objective.value(x))
        alphas.append(alpha_used)
        grad_norms.append(float(np.linalg.norm(g)))
        rhos.append(rho)
        predicted_decreases.append(predicted_decrease)
        actual_decreases.append(actual_decrease)
        accepted.append(step_accepted)
        descent_scores.append(descent_score)
        gradient_fallback_used.append(used_gradient_fallback)
        rhos_clipped.append(rho_clipped)
        predicted_decreases_safe.append(predicted_decrease_safe)
        predicted_was_floored.append(was_floored)
        rho_was_clipped.append(was_clipped)
        direction_types.append(direction_type)

    return OptimizationHistory(
        xs=np.asarray(xs),
        fs=np.asarray(fs),
        alphas=np.asarray(alphas),
        grad_norms=np.asarray(grad_norms),
        rhos=np.asarray(rhos),
        predicted_decreases=np.asarray(predicted_decreases),
        actual_decreases=np.asarray(actual_decreases),
        accepted=np.asarray(accepted),
        descent_scores=np.asarray(descent_scores),
        gradient_fallback_used=np.asarray(gradient_fallback_used),
        rhos_clipped=np.asarray(rhos_clipped),
        predicted_decreases_safe=np.asarray(predicted_decreases_safe),
        predicted_was_floored=np.asarray(predicted_was_floored),
        rho_was_clipped=np.asarray(rho_was_clipped),
        direction_types=np.asarray(direction_types, dtype=object),
    )
