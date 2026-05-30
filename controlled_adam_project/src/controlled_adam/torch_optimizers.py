"""PyTorch optimizer utilities for minibatch controlled Adam experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ControlledAdamStep:
    """Diagnostics from one controlled Adam minibatch step."""

    loss_before: float
    loss_after: float
    alpha: float
    rho: float
    predicted_decrease: float
    actual_decrease: float
    accepted: bool
    descent_score: float
    backtracks: int
    rho_ema: float
    alpha_next: float
    alpha_update_factor: float
    trust_region_expanded: bool
    used_gradient_fallback: bool = False
    rho_clipped: float = float("nan")
    predicted_decrease_safe: float = 0.0
    predicted_was_floored: bool = False
    rho_was_clipped: bool = False
    direction_type: str = "adam"
    trust_good_count: int = 0


class TorchControlledAdam:
    """Adam direction with an outer-loop actual-vs-predicted step controller.

    The training loop must compute gradients on one minibatch before calling
    :meth:`step`. The ``reevaluate_loss`` closure must evaluate the loss on that
    same minibatch after each trial parameter update, without calling backward.
    Acceptance uses measured rho; EMA and alpha control use clipped rho.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        alpha0: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        kp: float = 0.05,
        rho_star: float = 0.5,
        rho_min: float = 0.0,
        alpha_min: float = 1e-8,
        alpha_max: float = 1e-1,
        non_descent_shrink: float = 0.5,
        reject_bad_steps: bool = True,
        max_backtracks: int = 1,
        backtrack_shrink: float = 0.5,
        ratio_eps: float = 1e-12,
        absolute_predicted_floor: float = 1e-12,
        relative_predicted_floor: float = 1e-8,
        rho_clip_min: float = -1.0,
        rho_clip_max: float = 3.0,
        rho_beta: float = 0.9,
        use_rho_ema: bool = True,
        min_alpha_factor: float = 0.8,
        max_alpha_factor: float = 1.05,
        trust_region_expand: bool = True,
        trust_region_rho_threshold: float = 0.9,
        trust_region_alpha_threshold: float = 1e-4,
        trust_region_expand_factor: float = 1.5,
        trust_region_max_factor: float = 1.5,
        trust_region_patience: int = 2,
    ) -> None:
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive.")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1).")
        if eps <= 0 or ratio_eps <= 0:
            raise ValueError("eps and ratio_eps must be positive.")
        if absolute_predicted_floor <= 0.0:
            raise ValueError("absolute_predicted_floor must be positive.")
        if relative_predicted_floor < 0.0:
            raise ValueError("relative_predicted_floor must be non-negative.")
        if rho_clip_min >= rho_clip_max:
            raise ValueError("rho_clip_min must be < rho_clip_max.")
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
        if not (0.0 <= rho_beta < 1.0):
            raise ValueError("rho_beta must be in [0, 1).")
        if min_alpha_factor <= 0.0 or max_alpha_factor <= 0.0:
            raise ValueError("alpha update factors must be positive.")
        if min_alpha_factor > max_alpha_factor:
            raise ValueError("min_alpha_factor must be <= max_alpha_factor.")
        if trust_region_rho_threshold < 0.0:
            raise ValueError("trust_region_rho_threshold must be non-negative.")
        if trust_region_alpha_threshold <= 0.0:
            raise ValueError("trust_region_alpha_threshold must be positive.")
        if trust_region_expand_factor <= 1.0:
            raise ValueError("trust_region_expand_factor must be > 1.")
        if trust_region_max_factor <= 1.0:
            raise ValueError("trust_region_max_factor must be > 1.")
        if trust_region_patience < 1:
            raise ValueError("trust_region_patience must be >= 1.")

        self.params = [param for param in params if param.requires_grad]
        if not self.params:
            raise ValueError("TorchControlledAdam requires at least one parameter.")

        self.alpha = float(np.clip(alpha0, alpha_min, alpha_max))
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.kp = kp
        self.rho_star = rho_star
        self.rho_min = rho_min
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.non_descent_shrink = non_descent_shrink
        self.reject_bad_steps = reject_bad_steps
        self.max_backtracks = max_backtracks
        self.backtrack_shrink = backtrack_shrink
        self.ratio_eps = ratio_eps
        self.absolute_predicted_floor = absolute_predicted_floor
        self.relative_predicted_floor = relative_predicted_floor
        self.rho_clip_min = rho_clip_min
        self.rho_clip_max = rho_clip_max
        self.rho_beta = rho_beta
        self.use_rho_ema = use_rho_ema
        self.min_alpha_factor = min_alpha_factor
        self.max_alpha_factor = max_alpha_factor
        self.trust_region_expand = trust_region_expand
        self.trust_region_rho_threshold = trust_region_rho_threshold
        self.trust_region_alpha_threshold = trust_region_alpha_threshold
        self.trust_region_expand_factor = trust_region_expand_factor
        self.trust_region_max_factor = trust_region_max_factor
        self.trust_region_patience = trust_region_patience
        self.rho_ema: float | None = None
        self.trust_good_count = 0
        self.step_count = 0
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        """Clear parameter gradients."""
        for param in self.params:
            param.grad = None

    def step(
        self,
        loss_before: torch.Tensor,
        reevaluate_loss: Callable[[], torch.Tensor],
    ) -> ControlledAdamStep:
        """Take one controlled Adam step using the current minibatch gradients."""
        grads = []
        for param in self.params:
            if param.grad is None:
                grads.append(None)
            else:
                grads.append(param.grad.detach().clone())

        if all(grad is None for grad in grads):
            raise RuntimeError("No gradients are available for controlled Adam step.")

        self.step_count += 1
        directions: list[torch.Tensor | None] = []
        descent_score = 0.0
        grad_norm_sq = 0.0
        direction_norm_sq = 0.0

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                directions.append(None)
                continue

            grad_norm_sq += float(torch.sum(grad * grad).item())
            self.m[i].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            m_hat = self.m[i] / (1.0 - self.beta1**self.step_count)
            v_hat = self.v[i] / (1.0 - self.beta2**self.step_count)
            direction = -m_hat / (torch.sqrt(v_hat) + self.eps)
            directions.append(direction)
            descent_score -= float(torch.sum(grad * direction).item())
            direction_norm_sq += float(torch.sum(direction * direction).item())

        alpha_used = self.alpha
        loss_before_value = float(loss_before.detach().item())
        used_gradient_fallback = False
        direction_type = "adam"

        if descent_score <= 0.0:
            grad_norm = float(np.sqrt(grad_norm_sq))
            direction_norm = float(np.sqrt(direction_norm_sq))
            if grad_norm > 0.0 and direction_norm > 0.0:
                scale = direction_norm / (grad_norm + self.eps)
                directions = [
                    None if grad is None else -grad * scale
                    for grad in grads
                ]
                descent_score = scale * grad_norm_sq
                used_gradient_fallback = True
                direction_type = "gradient_fallback"

        if descent_score <= 0.0:
            direction_type = "degenerate_skip"
            self.trust_good_count = 0
            alpha_next = float(
                np.clip(
                    self.alpha * self.non_descent_shrink,
                    self.alpha_min,
                    self.alpha_max,
                )
            )
            alpha_update_factor = alpha_next / alpha_used
            self.alpha = alpha_next
            return ControlledAdamStep(
                loss_before=loss_before_value,
                loss_after=loss_before_value,
                alpha=alpha_used,
                rho=float("nan"),
                predicted_decrease=0.0,
                actual_decrease=0.0,
                accepted=False,
                descent_score=descent_score,
                backtracks=0,
                rho_ema=float("nan") if self.rho_ema is None else self.rho_ema,
                alpha_next=alpha_next,
                alpha_update_factor=alpha_update_factor,
                trust_region_expanded=False,
                used_gradient_fallback=used_gradient_fallback,
                direction_type=direction_type,
                trust_good_count=self.trust_good_count,
            )

        original_params = [param.detach().clone() for param in self.params]
        rho = float("nan")
        rho_clipped = float("nan")
        predicted_decrease = 0.0
        predicted_decrease_safe = 0.0
        predicted_was_floored = False
        rho_was_clipped = False
        actual_decrease = 0.0
        loss_after_value = loss_before_value

        for backtracks in range(self.max_backtracks + 1):
            trial_alpha = self.alpha * (self.backtrack_shrink**backtracks)
            self._set_trial_params(original_params, directions, trial_alpha)

            with torch.no_grad():
                loss_after = reevaluate_loss()
            loss_after_value = float(loss_after.detach().item())
            predicted_decrease = trial_alpha * descent_score
            actual_decrease = loss_before_value - loss_after_value
            (
                rho,
                rho_clipped,
                predicted_decrease_safe,
                predicted_was_floored,
                rho_was_clipped,
            ) = self._measure_rho(
                actual_decrease=actual_decrease,
                predicted_decrease_raw=predicted_decrease,
                loss_before_value=loss_before_value,
            )

            if (not self.reject_bad_steps) or (rho > self.rho_min):
                alpha_used = trial_alpha
                rho_control = self._update_rho_control(rho_clipped)
                alpha_next, alpha_update_factor, trust_region_expanded = (
                    self._next_alpha_after_trial(
                        alpha_used,
                        rho_control,
                        backtracks,
                        accepted=True,
                    )
                )
                self.alpha = alpha_next
                return ControlledAdamStep(
                    loss_before=loss_before_value,
                    loss_after=loss_after_value,
                    alpha=alpha_used,
                    rho=rho,
                    predicted_decrease=predicted_decrease,
                    actual_decrease=actual_decrease,
                    accepted=True,
                    descent_score=descent_score,
                    backtracks=backtracks,
                    rho_ema=rho_control,
                    alpha_next=alpha_next,
                    alpha_update_factor=alpha_update_factor,
                    trust_region_expanded=trust_region_expanded,
                    used_gradient_fallback=used_gradient_fallback,
                    rho_clipped=rho_clipped,
                    predicted_decrease_safe=predicted_decrease_safe,
                    predicted_was_floored=predicted_was_floored,
                    rho_was_clipped=rho_was_clipped,
                    direction_type=direction_type,
                    trust_good_count=self.trust_good_count,
                )

        self._restore_params(original_params)
        alpha_used = trial_alpha
        rho_control = (
            self._update_rho_control(rho_clipped)
            if np.isfinite(rho_clipped)
            else self.rho_ema
        )
        alpha_next, alpha_update_factor, trust_region_expanded = self._next_alpha_after_trial(
            alpha_used,
            rho_control if rho_control is not None else self.rho_star - 1.0,
            self.max_backtracks + 1,
            accepted=False,
        )
        self.alpha = alpha_next
        return ControlledAdamStep(
            loss_before=loss_before_value,
            loss_after=loss_after_value,
            alpha=alpha_used,
            rho=rho,
            predicted_decrease=predicted_decrease,
            actual_decrease=actual_decrease,
            accepted=False,
            descent_score=descent_score,
            backtracks=self.max_backtracks,
            rho_ema=float("nan") if rho_control is None else rho_control,
            alpha_next=alpha_next,
            alpha_update_factor=alpha_update_factor,
            trust_region_expanded=trust_region_expanded,
            used_gradient_fallback=used_gradient_fallback,
            rho_clipped=rho_clipped,
            predicted_decrease_safe=predicted_decrease_safe,
            predicted_was_floored=predicted_was_floored,
            rho_was_clipped=rho_was_clipped,
            direction_type=direction_type,
            trust_good_count=self.trust_good_count,
        )

    def _set_trial_params(
        self,
        originals: list[torch.Tensor],
        directions: list[torch.Tensor | None],
        alpha: float,
    ) -> None:
        with torch.no_grad():
            for param, original, direction in zip(self.params, originals, directions):
                if direction is None:
                    param.copy_(original)
                else:
                    param.copy_(original + alpha * direction)

    def _restore_params(self, originals: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for param, original in zip(self.params, originals):
                param.copy_(original)

    def _update_rho_control(self, rho: float) -> float:
        """Update and return the rho signal used by the alpha controller."""
        if not self.use_rho_ema:
            return rho
        if self.rho_ema is None:
            self.rho_ema = rho
        else:
            self.rho_ema = self.rho_beta * self.rho_ema + (1.0 - self.rho_beta) * rho
        return self.rho_ema

    def _next_alpha_after_trial(
        self,
        alpha_used: float,
        rho_control: float,
        backtracks: int,
        accepted: bool = True,
    ) -> tuple[float, float, bool]:
        error = rho_control - self.rho_star
        raw_factor = float(np.exp(self.kp * error))
        factor = float(np.clip(raw_factor, self.min_alpha_factor, self.max_alpha_factor))
        if not accepted:
            factor = min(factor, 1.0)

        if accepted and backtracks == 0 and rho_control >= self.trust_region_rho_threshold:
            self.trust_good_count += 1
        else:
            self.trust_good_count = 0

        trust_region_expanded = (
            self.trust_region_expand
            and accepted
            and backtracks == 0
            and rho_control >= self.trust_region_rho_threshold
            and alpha_used <= self.trust_region_alpha_threshold
            and self.trust_good_count >= self.trust_region_patience
        )
        if trust_region_expanded:
            factor = max(factor, self.trust_region_expand_factor)
            factor = min(factor, self.trust_region_max_factor)

        gamma_hard_max = self.max_alpha_factor
        if self.trust_region_expand:
            gamma_hard_max = max(gamma_hard_max, self.trust_region_max_factor)

        lower = self.min_alpha_factor
        upper = gamma_hard_max
        if not accepted:
            lower = min(lower, 1.0)
            upper = min(upper, 1.0)
        factor = float(np.clip(factor, lower, upper))

        alpha_next = float(np.clip(alpha_used * factor, self.alpha_min, self.alpha_max))
        alpha_update_factor = alpha_next / alpha_used
        return alpha_next, alpha_update_factor, trust_region_expanded

    def _measure_rho(
        self,
        *,
        actual_decrease: float,
        predicted_decrease_raw: float,
        loss_before_value: float,
    ) -> tuple[float, float, float, bool, bool]:
        """Return measured/clipped rho and denominator diagnostics."""
        predicted_floor = max(
            self.absolute_predicted_floor,
            self.ratio_eps,
            self.relative_predicted_floor * abs(loss_before_value),
        )
        predicted_decrease_safe = max(predicted_decrease_raw, predicted_floor)
        predicted_was_floored = predicted_decrease_safe > predicted_decrease_raw
        rho_measured = actual_decrease / predicted_decrease_safe
        rho_clipped = float(np.clip(rho_measured, self.rho_clip_min, self.rho_clip_max))
        rho_was_clipped = bool(rho_clipped != rho_measured)
        return (
            rho_measured,
            rho_clipped,
            predicted_decrease_safe,
            predicted_was_floored,
            rho_was_clipped,
        )
