"""PI-controlled Adam wrapper for PyTorch.

This module implements an outer-loop PI controller around an Adam-like
inner direction. Adam proposes a direction p_t, then the PI controller
updates a global step-size multiplier alpha_t using the same-batch
actual-vs-predicted decrease ratio.

Mathematical form
-----------------
Given parameters x_t and gradient g_t = grad f(x_t), Adam proposes

    p_t = - m_hat_t / (sqrt(v_hat_t) + eps_adam).

The wrapper takes a trial step

    x_trial = x_t + alpha_t p_t.

The first-order predicted decrease is

    Delta_hat_f_t = - alpha_t <g_t, p_t>.

The actual decrease is

    Delta_f_t = f(x_t) - f(x_trial).

The controller signal is

    rho_t = Delta_f_t / (Delta_hat_f_t + eps_pred),
    error_t = rho_bar_t - rho_star.

The PI controller updates log alpha:

    I_t = lambda_I I_{t-1} + error_t,
    log_alpha_{t+1} = log_alpha_t + Kp error_t + Ki I_t.

We use log alpha rather than alpha directly so that alpha remains positive.

Important use pattern
---------------------
The step method requires a closure that can be called twice on the same
minibatch:

    loss_before = closure(backward=True)   # forward + backward
    loss_after  = closure(backward=False)  # forward only, same minibatch

For stochastic neural-network training, the closure must reuse the exact same
batch for both calls. Otherwise rho_t is contaminated by minibatch variation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

import torch
from torch import Tensor


Closure = Callable[..., Tensor]


@dataclass
class PIAdamDiagnostics:
    """Diagnostics returned by each optimizer step."""

    loss_before: float
    loss_after: Optional[float]
    actual_decrease: Optional[float]
    predicted_decrease: Optional[float]
    rho: Optional[float]
    rho_bar: Optional[float]
    error: Optional[float]
    integral: float
    alpha: float
    log_alpha: float
    delta_log_alpha: float
    accepted: bool
    used_fallback_direction: bool
    backtracks: int = 0
    alpha_next: Optional[float] = None
    alpha_update_factor: Optional[float] = None
    trust_region_expanded: bool = False
    skipped_reason: Optional[str] = None


class PIAdam(torch.optim.Optimizer):
    """Adam direction with an outer-loop PI step-size controller.

    Parameters
    ----------
    params:
        Iterable of parameters to optimize.
    alpha0:
        Initial global step-size multiplier. This plays the role of Adam's
        learning rate, but is adapted by the PI controller.
    betas:
        Adam exponential averaging coefficients ``(beta1, beta2)``.
    adam_eps:
        Numerical epsilon in the Adam denominator.
    weight_decay:
        Decoupled AdamW-style weight decay, applied as parameter scaling during
        the trial/update step rather than folded into the Adam moments or the
        controller's predicted-decrease direction. For pure Adam, leave this at
        zero.
    rho_star:
        Target actual-over-predicted decrease ratio. Values around ``0.5`` are
        aggressive; values around ``0.8`` are more conservative.
    kp:
        Proportional gain for the PI controller.
    ki:
        Integral gain for the PI controller. Start much smaller than ``kp``.
    integral_decay:
        Leaky-integral coefficient. ``1.0`` gives a non-leaky integrator.
        Values such as ``0.95`` or ``0.99`` are safer.
    integral_clip:
        Bounds for the integral state, used for anti-windup.
    rho_smoothing:
        Exponential smoothing coefficient for rho. ``0.0`` means no smoothing.
        For minibatch training, values such as ``0.9`` or ``0.99`` are safer.
    alpha_min, alpha_max:
        Bounds for the global step-size multiplier.
    multiplicative_clip:
        Optional bounds on each multiplicative alpha update. For example,
        ``(0.8, 1.25)`` prevents a single step from changing alpha by more than
        -20% or +25%.
    predicted_decrease_eps:
        Numerical floor used in the rho denominator and descent-direction check.
    fallback_to_gradient:
        If Adam's momentum direction is not a descent direction, optionally use
        ``p_t = -g_t`` for that step.
    reject_bad_steps:
        If True, reject trial steps with ``rho <= rho_min`` and restore the old
        parameters. For stochastic neural-network training, this is usually
        False because minibatch rho is noisy.
    rho_min:
        Acceptance threshold when ``reject_bad_steps=True``.

    Notes
    -----
    This optimizer intentionally does **not** inherit the usual PyTorch Adam
    learning-rate field. The scalar ``alpha`` is the learning-rate-like global
    multiplier controlled by PI feedback.

    The closure must support ``backward=True`` and ``backward=False`` keyword
    arguments, or be a zero-argument closure that always computes gradients.
    The recommended pattern is shown in ``demo_toy_regression.py``.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        alpha0: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        rho_star: float = 0.8,
        kp: float = 0.05,
        ki: float = 0.001,
        integral_decay: float = 0.95,
        integral_clip: Tuple[float, float] = (-5.0, 5.0),
        rho_smoothing: float = 0.9,
        use_rho_ema: bool = True,
        alpha_min: float = 1e-7,
        alpha_max: float = 1e-1,
        multiplicative_clip: Optional[Tuple[float, float]] = (0.8, 1.25),
        predicted_decrease_eps: float = 1e-12,
        fallback_to_gradient: bool = True,
        reject_bad_steps: bool = False,
        rho_min: float = 0.0,
        non_descent_shrink: float = 0.5,
        max_backtracks: int = 4,
        backtrack_shrink: float = 0.5,
        trust_region_expand: bool = True,
        trust_region_rho_threshold: float = 0.9,
        trust_region_alpha_threshold: float = 1e-4,
        trust_region_expand_factor: float = 1.5,
    ) -> None:
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive.")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1).")
        if adam_eps <= 0:
            raise ValueError("adam_eps must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not (0.0 <= rho_smoothing < 1.0):
            raise ValueError("rho_smoothing must be in [0, 1).")
        if not (0.0 <= integral_decay <= 1.0):
            raise ValueError("integral_decay must be in [0, 1].")
        if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
            raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")
        if integral_clip[0] > integral_clip[1]:
            raise ValueError("integral_clip must be ordered as (min, max).")
        if multiplicative_clip is not None:
            if multiplicative_clip[0] <= 0 or multiplicative_clip[1] <= 0:
                raise ValueError("multiplicative_clip entries must be positive.")
            if multiplicative_clip[0] > multiplicative_clip[1]:
                raise ValueError("multiplicative_clip must be ordered as (min, max).")
        if not (0.0 < non_descent_shrink < 1.0):
            raise ValueError("non_descent_shrink must be in (0, 1).")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative.")
        if not (0.0 < backtrack_shrink < 1.0):
            raise ValueError("backtrack_shrink must be in (0, 1).")
        if trust_region_rho_threshold < 0.0:
            raise ValueError("trust_region_rho_threshold must be non-negative.")
        if trust_region_alpha_threshold <= 0.0:
            raise ValueError("trust_region_alpha_threshold must be positive.")
        if trust_region_expand_factor <= 1.0:
            raise ValueError("trust_region_expand_factor must be > 1.")

        defaults = dict(
            betas=betas,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.rho_star = float(rho_star)
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_decay = float(integral_decay)
        self.integral_min = float(integral_clip[0])
        self.integral_max = float(integral_clip[1])
        self.rho_smoothing = float(rho_smoothing)
        self.use_rho_ema = bool(use_rho_ema)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.predicted_decrease_eps = float(predicted_decrease_eps)
        self.fallback_to_gradient = bool(fallback_to_gradient)
        self.reject_bad_steps = bool(reject_bad_steps)
        self.rho_min = float(rho_min)
        self.non_descent_shrink = float(non_descent_shrink)
        self.max_backtracks = int(max_backtracks)
        self.backtrack_shrink = float(backtrack_shrink)
        self.trust_region_expand = bool(trust_region_expand)
        self.trust_region_rho_threshold = float(trust_region_rho_threshold)
        self.trust_region_alpha_threshold = float(trust_region_alpha_threshold)
        self.trust_region_expand_factor = float(trust_region_expand_factor)

        if multiplicative_clip is None:
            self.delta_log_min = -math.inf
            self.delta_log_max = math.inf
        else:
            self.delta_log_min = math.log(float(multiplicative_clip[0]))
            self.delta_log_max = math.log(float(multiplicative_clip[1]))

        clipped_alpha0 = min(max(float(alpha0), self.alpha_min), self.alpha_max)
        self.log_alpha = math.log(clipped_alpha0)
        self.integral = 0.0
        self.rho_bar: Optional[float] = None
        self.prev_error: Optional[float] = None
        self.last_diagnostics: Optional[PIAdamDiagnostics] = None

    @property
    def alpha(self) -> float:
        """Current global step-size multiplier."""
        return math.exp(self.log_alpha)

    def _call_closure(self, closure: Closure, *, backward: bool) -> Tensor:
        """Call a user closure with graceful fallback for no-arg closures."""
        try:
            return closure(backward=backward)
        except TypeError as exc:
            # Allows true no-arg closures without hiding TypeError raised
            # inside closures that do accept the backward keyword.
            import inspect

            try:
                inspect.signature(closure).bind(backward=backward)
            except TypeError:
                return closure()
            raise exc

    def _all_params(self) -> List[Tensor]:
        return [p for group in self.param_groups for p in group["params"] if p.requires_grad]

    @torch.no_grad()
    def _clone_params(self) -> List[Tensor]:
        return [p.detach().clone() for p in self._all_params()]

    @torch.no_grad()
    def _restore_params(self, saved: List[Tensor]) -> None:
        for p, old in zip(self._all_params(), saved):
            p.copy_(old)

    def _build_adam_direction(self) -> Tuple[List[Tuple[Tensor, Tensor, Tensor, float]], float, bool]:
        """Build Adam direction p_t and predicted-decrease numerator.

        Returns
        -------
        directions:
            List of ``(parameter, gradient, direction, weight_decay)`` tuples.
        minus_g_dot_p:
            The quantity ``-<g, p>`` before multiplication by alpha.
        used_fallback:
            Whether the Adam direction was replaced by a gradient direction.
        """
        directions: List[Tuple[Tensor, Tensor, Tensor, float]] = []
        minus_g_dot_p = 0.0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            adam_eps = group["adam_eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("PIAdam does not support sparse gradients.")

                state: MutableMapping[str, Any] = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                direction = -m_hat / (v_hat.sqrt() + adam_eps)

                directions.append((p, grad, direction, float(weight_decay)))
                minus_g_dot_p += float((-grad * direction).sum().detach().cpu())

        used_fallback = False

        # Adam with momentum can occasionally propose a non-descent direction.
        # The ratio rho_t only makes sense if -<g, p> > 0.
        if minus_g_dot_p <= self.predicted_decrease_eps and self.fallback_to_gradient:
            fallback_directions: List[Tuple[Tensor, Tensor, Tensor, float]] = []
            fallback_minus_g_dot_p = 0.0
            for p, grad, _direction, weight_decay in directions:
                direction = -grad
                fallback_directions.append((p, grad, direction, weight_decay))
                fallback_minus_g_dot_p += float((-grad * direction).sum().detach().cpu())
            directions = fallback_directions
            minus_g_dot_p = fallback_minus_g_dot_p
            used_fallback = True

        return directions, minus_g_dot_p, used_fallback

    @torch.no_grad()
    def _apply_direction(self, directions: List[Tuple[Tensor, Tensor, Tensor, float]], alpha: float) -> None:
        for p, _grad, direction, weight_decay in directions:
            if weight_decay != 0.0:
                p.mul_(1.0 - alpha * weight_decay)
            p.add_(direction, alpha=alpha)

    @torch.no_grad()
    def _set_trial_params(
        self,
        originals: List[Tensor],
        directions: List[Tuple[Tensor, Tensor, Tensor, float]],
        alpha: float,
    ) -> None:
        for original, (p, _grad, direction, weight_decay) in zip(originals, directions):
            trial = original
            if weight_decay != 0.0:
                trial = trial * (1.0 - alpha * weight_decay)
            p.copy_(trial + alpha * direction)

    @torch.no_grad()
    def _clone_direction_params(self, directions: List[Tuple[Tensor, Tensor, Tensor, float]]) -> List[Tensor]:
        return [p.detach().clone() for p, _grad, _direction, _weight_decay in directions]

    @torch.no_grad()
    def _restore_direction_params(
        self,
        originals: List[Tensor],
        directions: List[Tuple[Tensor, Tensor, Tensor, float]],
    ) -> None:
        for original, (p, _grad, _direction, _weight_decay) in zip(originals, directions):
            p.copy_(original)

    def _shrink_after_non_descent(self) -> Tuple[float, float]:
        alpha_before = self.alpha
        alpha_next = min(max(alpha_before * self.non_descent_shrink, self.alpha_min), self.alpha_max)
        self.log_alpha = math.log(alpha_next)
        return alpha_next, math.log(alpha_next / alpha_before)

    def _update_controller(
        self,
        rho: float,
        *,
        alpha_used: float,
        backtracks: int,
    ) -> Tuple[float, float, float, float, bool]:
        """Update PI controller and return rho signal, error, integral, log step, trust flag."""
        if self.use_rho_ema:
            if self.rho_bar is None:
                rho_bar = float(rho)
            else:
                rho_bar = self.rho_smoothing * self.rho_bar + (1.0 - self.rho_smoothing) * float(rho)
            self.rho_bar = rho_bar
        else:
            rho_bar = float(rho)
            self.rho_bar = rho_bar

        error = rho_bar - self.rho_star

        self.integral = self.integral_decay * self.integral + error
        self.integral = min(max(self.integral, self.integral_min), self.integral_max)

        delta_log = self.kp * error + self.ki * self.integral
        delta_log = min(max(delta_log, self.delta_log_min), self.delta_log_max)

        trust_region_expanded = (
            self.trust_region_expand
            and backtracks == 0
            and rho_bar >= self.trust_region_rho_threshold
            and alpha_used <= self.trust_region_alpha_threshold
        )
        if trust_region_expanded:
            delta_log = max(delta_log, math.log(self.trust_region_expand_factor))

        new_log_alpha = math.log(alpha_used) + delta_log
        new_log_alpha = min(max(new_log_alpha, math.log(self.alpha_min)), math.log(self.alpha_max))
        self.log_alpha = new_log_alpha

        self.prev_error = error
        return rho_bar, error, self.integral, delta_log, trust_region_expanded

    def step(self, closure: Closure) -> PIAdamDiagnostics:  # type: ignore[override]
        """Perform one PI-controlled Adam step.

        The closure should reuse the same minibatch for both calls. Recommended
        closure signature:

        .. code-block:: python

            def closure(backward: bool = True):
                optimizer.zero_grad(set_to_none=True)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                if backward:
                    loss.backward()
                return loss

        Returns
        -------
        PIAdamDiagnostics
            A dataclass containing loss, rho, alpha, and controller diagnostics.
        """
        # First evaluation: compute loss and gradients at current parameters.
        with torch.enable_grad():
            loss_before_tensor = self._call_closure(closure, backward=True)
        loss_before = float(loss_before_tensor.detach().cpu())

        alpha_before = self.alpha

        directions, minus_g_dot_p, used_fallback = self._build_adam_direction()

        if not directions:
            diagnostics = PIAdamDiagnostics(
                loss_before=loss_before,
                loss_after=None,
                actual_decrease=None,
                predicted_decrease=None,
                rho=None,
                rho_bar=self.rho_bar,
                error=None,
                integral=self.integral,
                alpha=alpha_before,
                log_alpha=self.log_alpha,
                delta_log_alpha=0.0,
                accepted=False,
                used_fallback_direction=used_fallback,
                alpha_next=alpha_before,
                alpha_update_factor=1.0,
                skipped_reason="no gradients available",
            )
            self.last_diagnostics = diagnostics
            return diagnostics

        if minus_g_dot_p <= self.predicted_decrease_eps:
            # Cannot define a meaningful predicted decrease. Shrink alpha.
            alpha_next, delta_log = self._shrink_after_non_descent()
            diagnostics = PIAdamDiagnostics(
                loss_before=loss_before,
                loss_after=None,
                actual_decrease=None,
                predicted_decrease=None,
                rho=None,
                rho_bar=self.rho_bar,
                error=None,
                integral=self.integral,
                alpha=alpha_before,
                log_alpha=self.log_alpha,
                delta_log_alpha=delta_log,
                accepted=False,
                used_fallback_direction=used_fallback,
                alpha_next=alpha_next,
                alpha_update_factor=alpha_next / alpha_before,
                skipped_reason="non-descent direction and fallback unavailable",
            )
            self.last_diagnostics = diagnostics
            return diagnostics

        original_params = self._clone_direction_params(directions)
        rho = float("nan")
        loss_after = loss_before
        predicted_decrease = 0.0
        actual_decrease = 0.0
        accepted = False
        alpha_used = alpha_before
        backtracks_taken = 0

        for backtracks in range(self.max_backtracks + 1):
            trial_alpha = alpha_before * (self.backtrack_shrink**backtracks)
            self._set_trial_params(original_params, directions, trial_alpha)

            # Second evaluation: same batch, no backward pass required.
            with torch.no_grad():
                loss_after_tensor = self._call_closure(closure, backward=False)
            loss_after = float(loss_after_tensor.detach().cpu())

            predicted_decrease = trial_alpha * minus_g_dot_p
            actual_decrease = loss_before - loss_after
            rho = actual_decrease / (predicted_decrease + self.predicted_decrease_eps)
            alpha_used = trial_alpha
            backtracks_taken = backtracks

            if (not self.reject_bad_steps) or (rho > self.rho_min):
                accepted = True
                break

        if not accepted:
            self._restore_direction_params(original_params, directions)

        rho_bar, error, integral, delta_log, trust_region_expanded = self._update_controller(
            rho if math.isfinite(rho) else self.rho_star - 1.0,
            alpha_used=alpha_used,
            backtracks=backtracks_taken if accepted else self.max_backtracks + 1,
        )
        alpha_next = self.alpha

        diagnostics = PIAdamDiagnostics(
            loss_before=loss_before,
            loss_after=loss_after,
            actual_decrease=actual_decrease,
            predicted_decrease=predicted_decrease,
            rho=float(rho),
            rho_bar=rho_bar,
            error=error,
            integral=integral,
            alpha=self.alpha,
            log_alpha=self.log_alpha,
            delta_log_alpha=delta_log,
            accepted=accepted,
            used_fallback_direction=used_fallback,
            backtracks=backtracks_taken,
            alpha_next=alpha_next,
            alpha_update_factor=alpha_next / alpha_used,
            trust_region_expanded=trust_region_expanded,
            skipped_reason=None,
        )
        self.last_diagnostics = diagnostics
        return diagnostics

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha:.3e}, rho_star={self.rho_star:.3f}, "
            f"kp={self.kp:.3e}, ki={self.ki:.3e}, integral={self.integral:.3e}"
        )
