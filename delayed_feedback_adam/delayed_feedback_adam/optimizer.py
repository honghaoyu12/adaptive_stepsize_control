"""Adam with a delayed actual-vs-predicted decrease controller.

The optimizer has two components:

1. An Adam-style inner optimizer that proposes a preconditioned direction.
2. A delayed outer controller that adapts a global learning-rate multiplier.

At iteration t, the controller compares the current loss f_t with the previous
loss f_{t-1}. This estimates how well the previous step performed:

    rho_{t-1} = (f_{t-1} - f_t) / predicted_decrease_{t-1}.

The current multiplier is then updated before applying the Adam step at the
current parameters. This avoids evaluating f(x_{t+1}) immediately after a trial
step, so it does not require an extra forward pass in the usual training loop.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple, Union, Dict, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

LossInput = Optional[Union[float, Tensor]]


class DelayedFeedbackAdam(Optimizer):
    """Adam with a delayed outer learning-rate controller.

    Parameters
    ----------
    params:
        Iterable of parameters to optimize.
    lr:
        Base Adam learning rate. The effective learning rate is
        ``lr * alpha``, where ``alpha`` is controlled online.
    betas:
        Adam exponential averaging coefficients.
    adam_eps:
        Adam denominator epsilon.
    weight_decay:
        Weight decay coefficient.
    decoupled_weight_decay:
        If True, use AdamW-style decoupled weight decay. If False, use coupled
        L2 regularization by adding ``weight_decay * parameter`` to the gradient.
    alpha_init:
        Initial global learning-rate multiplier.
    alpha_bounds:
        Minimum and maximum allowed alpha multiplier.
    rho_star:
        Target actual-over-predicted decrease ratio.
    kp, ki, kd:
        P/PI/PID controller gains. Set ``ki=kd=0`` for the recommended initial
        delayed P controller.
    rho_beta:
        Exponential smoothing coefficient for rho. Use 0 for deterministic
        problems and 0.9--0.99 for noisy minibatch training.
    rho_clip:
        Clamp raw rho to this interval before smoothing.
    multiplier_bounds:
        Clamp the per-step multiplicative alpha update to this interval. This
        prevents one noisy loss value from changing alpha too aggressively.
    integral_decay:
        Leaky-integral decay. Values near 1 retain longer memory.
    integral_bounds:
        Anti-windup bounds for the integral state.
    derivative_beta:
        Smoothing coefficient for the derivative term.
    min_predicted_decrease:
        If the predicted decrease from the previous step is too small, skip the
        controller update for that step.
    fallback_to_gradient:
        If Adam's direction is not descent-aligned with the current gradient,
        use ``-gradient`` for that parameter tensor's direction. This makes the
        predicted decrease meaningful even when momentum points uphill.

    Notes
    -----
    The typical training loop is:

    >>> optimizer.zero_grad()
    >>> loss = criterion(model(inputs), targets)
    >>> loss.backward()
    >>> optimizer.step(loss=loss.item())

    Passing ``loss`` is important. The optimizer uses it as the current loss
    measurement for delayed feedback. No extra forward pass is required.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        *,
        alpha_init: float = 1.0,
        alpha_bounds: Tuple[float, float] = (1e-3, 1e3),
        rho_star: float = 0.8,
        kp: float = 0.05,
        ki: float = 0.0,
        kd: float = 0.0,
        rho_beta: float = 0.95,
        rho_clip: Tuple[float, float] = (-1.0, 2.0),
        multiplier_bounds: Tuple[float, float] = (0.8, 1.25),
        integral_decay: float = 0.95,
        integral_bounds: Tuple[float, float] = (-5.0, 5.0),
        derivative_beta: float = 0.9,
        min_predicted_decrease: float = 1e-16,
        fallback_to_gradient: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        if alpha_bounds[0] <= 0 or alpha_bounds[0] > alpha_bounds[1]:
            raise ValueError("alpha_bounds must satisfy 0 < min <= max")
        if multiplier_bounds[0] <= 0 or multiplier_bounds[0] > multiplier_bounds[1]:
            raise ValueError("multiplier_bounds must satisfy 0 < min <= max")
        if not 0 <= rho_beta < 1:
            raise ValueError("rho_beta must be in [0, 1)")
        if not 0 <= integral_decay < 1:
            raise ValueError("integral_decay must be in [0, 1)")
        if not 0 <= derivative_beta < 1:
            raise ValueError("derivative_beta must be in [0, 1)")
        if min_predicted_decrease <= 0:
            raise ValueError("min_predicted_decrease must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

        alpha_init = float(max(alpha_bounds[0], min(alpha_init, alpha_bounds[1])))
        self.controller: Dict[str, Any] = {
            "alpha": alpha_init,
            "log_alpha": math.log(alpha_init),
            "alpha_min": float(alpha_bounds[0]),
            "alpha_max": float(alpha_bounds[1]),
            "rho_star": float(rho_star),
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            "rho_beta": float(rho_beta),
            "rho_min": float(rho_clip[0]),
            "rho_max": float(rho_clip[1]),
            "multiplier_min": float(multiplier_bounds[0]),
            "multiplier_max": float(multiplier_bounds[1]),
            "integral_decay": float(integral_decay),
            "integral_min": float(integral_bounds[0]),
            "integral_max": float(integral_bounds[1]),
            "derivative_beta": float(derivative_beta),
            "min_predicted_decrease": float(min_predicted_decrease),
            "fallback_to_gradient": bool(fallback_to_gradient),
            "prev_loss": None,
            "prev_predicted_decrease": None,
            "rho_bar": None,
            "prev_error": None,
            "integral": 0.0,
            "derivative": 0.0,
            "last_rho_raw": None,
            "last_rho_clipped": None,
            "last_error": None,
            "last_multiplier": 1.0,
            "last_predicted_decrease": None,
            "last_actual_decrease": None,
            "last_controller_applied": False,
            "last_non_descent_tensors": 0,
        }

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], Tensor]] = None,
        *,
        loss: LossInput = None,
    ) -> Optional[Tensor]:
        """Perform one optimization step.

        Parameters
        ----------
        closure:
            Optional callable that reevaluates the model and returns the loss.
            This follows the PyTorch optimizer convention. For typical training,
            it is simpler to pass the already-computed loss using ``loss=``.
        loss:
            Current loss value, preferably the same scalar used for the backward
            pass. Passing this enables the delayed controller update.

        Returns
        -------
        Optional[Tensor]
            The closure loss, if a closure was provided. Otherwise ``None``.
        """
        closure_loss = None
        if closure is not None:
            with torch.enable_grad():
                closure_loss = closure()
            if loss is None:
                loss = closure_loss

        loss_value = self._loss_to_float(loss)
        if loss_value is not None:
            self._update_controller_from_previous_step(loss_value)

        alpha = float(self.controller["alpha"])
        predicted_decrease = 0.0
        non_descent_tensors = 0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            adam_eps = group["adam_eps"]
            weight_decay = group["weight_decay"]
            decoupled_weight_decay = group["decoupled_weight_decay"]
            step_size = lr * alpha

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("DelayedFeedbackAdam does not support sparse gradients")

                if weight_decay != 0 and not decoupled_weight_decay:
                    grad_for_adam = grad.add(param, alpha=weight_decay)
                else:
                    grad_for_adam = grad

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_num = state["step"]

                exp_avg.mul_(beta1).add_(grad_for_adam, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_for_adam, grad_for_adam, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step_num
                bias_correction2 = 1.0 - beta2 ** step_num

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                adam_direction = -m_hat / (v_hat.sqrt() + adam_eps)

                # Check whether Adam's direction is descent-aligned with the current gradient.
                # This uses the gradient seen by the optimizer. For decoupled weight decay,
                # predicted decrease is still based on the data-loss gradient component.
                directional_derivative = torch.sum(grad_for_adam * adam_direction).item()
                if directional_derivative >= 0.0 and self.controller["fallback_to_gradient"]:
                    direction = -grad_for_adam
                    non_descent_tensors += 1
                else:
                    direction = adam_direction

                predicted_decrease += -step_size * torch.sum(grad_for_adam * direction).item()

                if weight_decay != 0 and decoupled_weight_decay:
                    param.mul_(1.0 - step_size * weight_decay)

                param.add_(direction, alpha=step_size)

        self.controller["last_non_descent_tensors"] = int(non_descent_tensors)
        self.controller["last_predicted_decrease"] = float(predicted_decrease)

        if loss_value is not None:
            self.controller["prev_loss"] = float(loss_value)
            self.controller["prev_predicted_decrease"] = float(predicted_decrease)

        return closure_loss

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return controller diagnostics as a plain dictionary."""
        keys = [
            "alpha",
            "rho_star",
            "rho_bar",
            "last_rho_raw",
            "last_rho_clipped",
            "last_error",
            "integral",
            "derivative",
            "last_multiplier",
            "last_predicted_decrease",
            "last_actual_decrease",
            "last_controller_applied",
            "last_non_descent_tensors",
        ]
        return {key: self.controller.get(key) for key in keys}

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        """Return optimizer state, including the global controller state."""
        state = super().state_dict()
        state["controller"] = dict(self.controller)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        """Load optimizer state, including the global controller state."""
        controller = state_dict.pop("controller", None)
        super().load_state_dict(state_dict)
        if controller is not None:
            self.controller.update(controller)

    @staticmethod
    def _loss_to_float(loss: LossInput) -> Optional[float]:
        if loss is None:
            return None
        if isinstance(loss, Tensor):
            return float(loss.detach().item())
        return float(loss)

    def _update_controller_from_previous_step(self, current_loss: float) -> None:
        prev_loss = self.controller["prev_loss"]
        prev_predicted = self.controller["prev_predicted_decrease"]

        self.controller["last_controller_applied"] = False
        self.controller["last_actual_decrease"] = None
        self.controller["last_rho_raw"] = None
        self.controller["last_rho_clipped"] = None
        self.controller["last_error"] = None
        self.controller["last_multiplier"] = 1.0

        if prev_loss is None or prev_predicted is None:
            return

        prev_predicted = float(prev_predicted)
        if not math.isfinite(prev_predicted) or prev_predicted <= self.controller["min_predicted_decrease"]:
            return

        actual_decrease = float(prev_loss) - float(current_loss)
        rho_raw = actual_decrease / prev_predicted
        rho_clipped = max(self.controller["rho_min"], min(rho_raw, self.controller["rho_max"]))

        rho_bar = self.controller["rho_bar"]
        if rho_bar is None:
            rho_bar = rho_clipped
        else:
            beta = self.controller["rho_beta"]
            rho_bar = beta * float(rho_bar) + (1.0 - beta) * rho_clipped

        error = rho_bar - self.controller["rho_star"]

        # PI term with leakage and anti-windup bounds.
        integral = self.controller["integral_decay"] * float(self.controller["integral"]) + error
        integral = max(self.controller["integral_min"], min(integral, self.controller["integral_max"]))

        # Smoothed derivative term.
        prev_error = self.controller["prev_error"]
        raw_derivative = 0.0 if prev_error is None else error - float(prev_error)
        derivative = (
            self.controller["derivative_beta"] * float(self.controller["derivative"])
            + (1.0 - self.controller["derivative_beta"]) * raw_derivative
        )

        log_multiplier = (
            self.controller["kp"] * error
            + self.controller["ki"] * integral
            + self.controller["kd"] * derivative
        )
        multiplier = math.exp(log_multiplier)
        multiplier = max(self.controller["multiplier_min"], min(multiplier, self.controller["multiplier_max"]))

        log_alpha = float(self.controller["log_alpha"]) + math.log(multiplier)
        log_alpha = max(math.log(self.controller["alpha_min"]), min(log_alpha, math.log(self.controller["alpha_max"])))
        alpha = math.exp(log_alpha)

        self.controller["rho_bar"] = float(rho_bar)
        self.controller["prev_error"] = float(error)
        self.controller["integral"] = float(integral)
        self.controller["derivative"] = float(derivative)
        self.controller["log_alpha"] = float(log_alpha)
        self.controller["alpha"] = float(alpha)
        self.controller["last_rho_raw"] = float(rho_raw)
        self.controller["last_rho_clipped"] = float(rho_clipped)
        self.controller["last_error"] = float(error)
        self.controller["last_multiplier"] = float(multiplier)
        self.controller["last_actual_decrease"] = float(actual_decrease)
        self.controller["last_controller_applied"] = True
