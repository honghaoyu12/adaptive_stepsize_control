"""Muon with a delayed actual-vs-predicted decrease controller.

This optimizer combines:

1. A Muon-style inner optimizer for matrix-like parameters. Muon forms a
   momentum update and approximately orthogonalizes it with Newton-Schulz
   iterations.
2. An AdamW-style auxiliary inner optimizer for non-matrix parameters such as
   biases and normalization gains.
3. A delayed outer controller that adapts a global learning-rate multiplier
   using the previous step's observed loss change.

At iteration t, the controller compares the current loss f_t with the previous
loss f_{t-1}. This estimates how well the previous step performed:

    rho_{t-1} = (f_{t-1} - f_t) / predicted_decrease_{t-1}.

The current multiplier is then updated before applying the Muon/AdamW step at
the current parameters. This avoids evaluating f(x_{t+1}) immediately after a
trial step, so it does not require an extra forward pass in the usual training
loop.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

LossInput = Optional[Union[float, Tensor]]


class DelayedFeedbackMuon(Optimizer):
    """Muon with a delayed outer learning-rate controller.

    Parameters
    ----------
    params:
        Iterable of parameters, or parameter groups. By default, 2D tensors use
        Muon and non-2D tensors use auxiliary AdamW. You may override this with
        the per-group option ``use_muon``.
    lr:
        Base learning rate. The effective learning rate is ``lr * alpha``
        times any Muon shape adjustment.
    weight_decay:
        Decoupled weight decay coefficient.
    momentum:
        Muon momentum coefficient.
    nesterov:
        If True, use PyTorch's Nesterov-style Muon update
        ``lerp(grad, buf, momentum)``.
    ns_steps:
        Number of Newton-Schulz iterations.
    ns_coefficients:
        Coefficients ``(a, b, c)`` for the Newton-Schulz polynomial
        ``aX + b(XX^T)X + c(XX^T)^2X``.
    muon_eps:
        Numerical epsilon for Newton-Schulz normalization.
    adjust_lr_fn:
        Shape-based Muon learning-rate adjustment. Supported values are
        ``"original"``, ``"match_rms_adamw"``, and ``None``. ``None`` and
        ``"original"`` apply ``sqrt(max(1, rows / cols))`` to matrix updates,
        matching ``torch.optim.Muon``.
        ``"match_rms_adamw"`` applies ``0.2 * sqrt(max(rows, cols))``.
    aux_betas, aux_eps:
        AdamW fallback hyperparameters for non-Muon tensors.
    use_muon:
        Per-group setting. ``"auto"`` means use Muon for 2D tensors and
        auxiliary AdamW otherwise. ``True`` forces Muon for eligible 2D tensors.
        ``False`` uses auxiliary AdamW for the group.
    alpha_init, alpha_bounds:
        Initial and bounded global controller multiplier.
    rho_star:
        Target actual-over-predicted decrease ratio.
    kp, ki, kd:
        P/PI/PID controller gains. Start with ``ki=kd=0``.
    rho_beta:
        Exponential smoothing coefficient for delayed ``rho``.
    rho_clip:
        Clamp raw delayed ``rho`` before smoothing.
    multiplier_bounds:
        Clamp the per-step multiplicative update to alpha.
    integral_decay, integral_bounds, derivative_beta:
        Optional PI/PID stabilizers.
    min_predicted_decrease:
        Skip controller update when previous predicted decrease is too small.
    fallback_to_gradient:
        If Muon or auxiliary AdamW proposes a non-descent direction for a tensor,
        fall back to the negative gradient for that tensor.

    Notes
    -----
    Typical usage:

    >>> optimizer.zero_grad()
    >>> loss = criterion(model(inputs), targets)
    >>> loss.backward()
    >>> optimizer.step(loss=loss.item())

    Passing ``loss`` is important. It provides the delayed feedback measurement
    without requiring an extra forward pass.
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        lr: float = 2e-2,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        muon_eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = "original",
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        use_muon: Union[str, bool] = "auto",
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
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 0 <= ns_steps < 100:
            raise ValueError("ns_steps must be in [0, 100)")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a 3-tuple")
        if muon_eps <= 0:
            raise ValueError("muon_eps must be positive")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError("adjust_lr_fn must be None, 'original', or 'match_rms_adamw'")
        if not 0 <= aux_betas[0] < 1 or not 0 <= aux_betas[1] < 1:
            raise ValueError("aux_betas must be in [0, 1)")
        if aux_eps <= 0:
            raise ValueError("aux_eps must be positive")
        if use_muon not in ("auto", True, False):
            raise ValueError("use_muon must be 'auto', True, or False")
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
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            muon_eps=muon_eps,
            adjust_lr_fn=adjust_lr_fn,
            aux_betas=aux_betas,
            aux_eps=aux_eps,
            use_muon=use_muon,
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
            "last_muon_tensors": 0,
            "last_aux_adamw_tensors": 0,
        }

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], Tensor]] = None,
        *,
        loss: LossInput = None,
    ) -> Optional[Tensor]:
        """Perform one optimization step."""
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
        muon_tensors = 0
        aux_adamw_tensors = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            use_muon_setting = group["use_muon"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if torch.is_complex(param) or torch.is_complex(param.grad):
                    raise RuntimeError("DelayedFeedbackMuon does not support complex parameters or gradients")
                if param.grad.is_sparse:
                    raise RuntimeError("DelayedFeedbackMuon does not support sparse gradients")

                grad = param.grad
                should_use_muon = self._should_use_muon(param, use_muon_setting)

                if should_use_muon:
                    direction, shape_lr_factor = self._muon_direction(param, grad, group)
                    muon_tensors += 1
                else:
                    direction, shape_lr_factor = self._aux_adamw_direction(param, grad, group)
                    aux_adamw_tensors += 1

                effective_lr = lr * alpha * shape_lr_factor

                directional_derivative = torch.sum(grad * direction).item()
                if directional_derivative >= 0.0 and self.controller["fallback_to_gradient"]:
                    direction = -grad
                    directional_derivative = torch.sum(grad * direction).item()
                    non_descent_tensors += 1

                # The denominator is the first-order predicted loss decrease
                # from the optimizer direction. Decoupled weight decay is applied
                # separately and intentionally not included in this diagnostic,
                # because the passed loss often excludes weight decay.
                predicted_decrease += -effective_lr * directional_derivative

                if weight_decay != 0.0:
                    # Match torch.optim.Muon: decoupled weight decay uses the
                    # base learning rate, not the shape-adjusted Muon LR.
                    param.mul_(1.0 - lr * alpha * weight_decay)

                param.add_(direction, alpha=effective_lr)

        self.controller["last_non_descent_tensors"] = int(non_descent_tensors)
        self.controller["last_muon_tensors"] = int(muon_tensors)
        self.controller["last_aux_adamw_tensors"] = int(aux_adamw_tensors)
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
            "last_muon_tensors",
            "last_aux_adamw_tensors",
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

    @staticmethod
    def _should_use_muon(param: Tensor, use_muon_setting: Union[str, bool]) -> bool:
        if use_muon_setting is True:
            return param.ndim == 2
        if use_muon_setting is False:
            return False
        return param.ndim == 2

    def _muon_direction(self, param: Tensor, grad: Tensor, group: Dict[str, Any]) -> Tuple[Tensor, float]:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        momentum = float(group["momentum"])
        buf = state["momentum_buffer"]
        # Match torch.optim.Muon:
        #   buf <- lerp(buf, grad, 1 - momentum)
        #   update <- lerp(grad, buf, momentum) if nesterov else buf
        buf.lerp_(grad, 1.0 - momentum)

        if bool(group["nesterov"]):
            update = grad.lerp(buf, momentum)
        else:
            update = buf

        matrix = update.reshape(update.shape[0], -1)
        ortho = _zeropower_newton_schulz(
            matrix,
            steps=int(group["ns_steps"]),
            coefficients=group["ns_coefficients"],
            eps=float(group["muon_eps"]),
        ).reshape_as(update)

        direction = -ortho.to(dtype=param.dtype)
        shape_lr_factor = self._shape_lr_factor(matrix.shape, group["adjust_lr_fn"])
        return direction, shape_lr_factor

    def _aux_adamw_direction(self, param: Tensor, grad: Tensor, group: Dict[str, Any]) -> Tuple[Tensor, float]:
        beta1, beta2 = group["aux_betas"]
        aux_eps = float(group["aux_eps"])

        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1
        step_num = state["step"]

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** step_num
        bias_correction2 = 1.0 - beta2 ** step_num

        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2
        direction = -m_hat / (v_hat.sqrt() + aux_eps)
        return direction, 1.0

    @staticmethod
    def _shape_lr_factor(shape: torch.Size, adjust_lr_fn: Optional[str]) -> float:
        rows, cols = int(shape[0]), int(shape[1])
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            return math.sqrt(max(1.0, rows / max(1, cols)))
        if adjust_lr_fn == "match_rms_adamw":
            return 0.2 * math.sqrt(max(rows, cols))
        raise ValueError(f"Unsupported adjust_lr_fn: {adjust_lr_fn}")

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

        integral = self.controller["integral_decay"] * float(self.controller["integral"]) + error
        integral = max(self.controller["integral_min"], min(integral, self.controller["integral_max"]))

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


def _zeropower_newton_schulz(
    matrix: Tensor,
    *,
    steps: int = 5,
    coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> Tensor:
    """Approximate the orthogonal factor of a matrix using Newton-Schulz.

    This computes an approximate zeroth power of ``matrix``. If ``matrix`` has
    SVD ``U S V^T``, the target orthogonalized update is approximately
    ``U V^T``. The implementation uses the common Muon trick of transposing
    tall matrices so the iteration multiplies the smaller side.
    """
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be 2D after reshaping")

    if matrix.numel() == 0:
        return matrix

    original_dtype = matrix.dtype
    X = matrix.float()
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    X = X / (X.norm() + eps)
    a, b, c = coefficients
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(dtype=original_dtype)
