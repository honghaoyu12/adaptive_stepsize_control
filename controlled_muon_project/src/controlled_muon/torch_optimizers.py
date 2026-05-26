"""PyTorch Muon optimizer utilities for minibatch benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from controlled_muon.orthogonalization import orthogonalize


@dataclass(frozen=True)
class ControlledMuonStep:
    """Diagnostics from one controlled Muon minibatch step."""

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


class MuonConfig:
    """Configuration for the educational PyTorch Muon implementation."""

    def __init__(
        self,
        momentum: float = 0.95,
        nesterov: bool = True,
        orthogonalizer: str = "newton_schulz",
        ns_steps: int = 5,
        update_scale: float = 1.0,
        shape_scale: bool = True,
    ) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1.")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive.")
        if update_scale <= 0:
            raise ValueError("update_scale must be positive.")
        self.momentum = momentum
        self.nesterov = nesterov
        self.orthogonalizer = orthogonalizer
        self.ns_steps = ns_steps
        self.update_scale = update_scale
        self.shape_scale = shape_scale


def default_muon_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    adamw_name_keywords: Sequence[str] = (
        "embed",
        "embedding",
        "lm_head",
        "head",
        "norm",
        "bias",
        "bn",
    ),
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split model parameters into official-style Muon and AdamW fallback params."""
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        excluded_by_name = any(keyword in name.lower() for keyword in adamw_name_keywords)
        if param.ndim == 2 and not excluded_by_name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


class TorchControlledMuon:
    """Muon direction with an outer-loop actual-vs-predicted step controller."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]],
        alpha0: float = 1e-2,
        config: MuonConfig | None = None,
        kp: float = 0.05,
        rho_star: float = 0.7,
        rho_min: float = 0.0,
        alpha_min: float = 1e-8,
        alpha_max: float = 1e-1,
        non_descent_shrink: float = 0.5,
        reject_bad_steps: bool = True,
        max_backtracks: int = 4,
        backtrack_shrink: float = 0.5,
        ratio_eps: float = 1e-12,
        rho_beta: float = 0.9,
        use_rho_ema: bool = True,
        min_alpha_factor: float = 0.8,
        max_alpha_factor: float = 1.05,
        trust_region_expand: bool = True,
        trust_region_rho_threshold: float = 0.9,
        trust_region_alpha_threshold: float = 1e-4,
        trust_region_expand_factor: float = 1.5,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive.")
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
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")

        if isinstance(params, tuple):
            muon_params, adamw_params = params
            self.muon_params = [param for param in muon_params if param.requires_grad]
            self.adamw_params = [param for param in adamw_params if param.requires_grad]
        else:
            self.muon_params = [param for param in params if param.requires_grad and param.ndim == 2]
            self.adamw_params = [param for param in params if param.requires_grad and param.ndim != 2]
        self.params = [*self.muon_params, *self.adamw_params]
        if not self.params:
            raise ValueError("TorchControlledMuon requires at least one parameter.")

        self.config = config or MuonConfig()
        self.alpha = float(np.clip(alpha0, alpha_min, alpha_max))
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
        self.rho_beta = rho_beta
        self.use_rho_ema = use_rho_ema
        self.min_alpha_factor = min_alpha_factor
        self.max_alpha_factor = max_alpha_factor
        self.trust_region_expand = trust_region_expand
        self.trust_region_rho_threshold = trust_region_rho_threshold
        self.trust_region_alpha_threshold = trust_region_alpha_threshold
        self.trust_region_expand_factor = trust_region_expand_factor
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.weight_decay = float(weight_decay)
        self.rho_ema: float | None = None
        self.step_count = 0
        self.momentum = {param: torch.zeros_like(param) for param in self.muon_params}
        self.adam_state: dict[torch.nn.Parameter, dict[str, torch.Tensor | int]] = {
            param: {} for param in self.adamw_params
        }

    @staticmethod
    def _tensor_to_matrix(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...]]:
        """Flatten a tensor to a 2D matrix for Muon-style orthogonalization."""
        shape = tuple(tensor.shape)
        if tensor.ndim == 2:
            matrix = tensor.detach().cpu().numpy()
        else:
            raise ValueError("Official-style Muon supports only 2D tensors.")
        return matrix, shape

    @staticmethod
    def _matrix_to_tensor(matrix: np.ndarray, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(matrix.reshape(shape)).to(device=device, dtype=dtype)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def _direction(self, grad: torch.Tensor, momentum_buffer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_momentum = momentum_buffer.lerp(grad, 1.0 - self.config.momentum)
        if self.config.nesterov:
            matrix_to_orthogonalize = grad.lerp(new_momentum, self.config.momentum)
        else:
            matrix_to_orthogonalize = new_momentum
        matrix, shape = self._tensor_to_matrix(matrix_to_orthogonalize)
        ortho_update = orthogonalize(
            matrix,
            method=self.config.orthogonalizer,
            ns_steps=self.config.ns_steps,
        )
        if self.config.shape_scale:
            rows, cols = matrix.shape
            ortho_update = ortho_update * np.sqrt(max(1.0, rows / max(cols, 1)))
        direction = -self.config.update_scale * self._matrix_to_tensor(
            ortho_update,
            shape,
            device=grad.device,
            dtype=grad.dtype,
        )
        return direction, new_momentum

    def _adamw_direction(
        self,
        grad: torch.Tensor,
        state: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        beta1, beta2 = self.adam_betas
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] = int(state["step"]) + 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        assert isinstance(exp_avg, torch.Tensor)
        assert isinstance(exp_avg_sq, torch.Tensor)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        step = int(state["step"])
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denom = exp_avg_sq.sqrt().div(np.sqrt(bias_correction2)).add_(self.adam_eps)
        return -exp_avg.div(bias_correction1).div(denom)

    def step(
        self,
        loss_before: torch.Tensor,
        reevaluate_loss: Callable[[], torch.Tensor],
    ) -> ControlledMuonStep:
        grads = []
        for param in self.params:
            grads.append(None if param.grad is None else param.grad.detach().clone())
        if all(grad is None for grad in grads):
            raise RuntimeError("No gradients are available for controlled Muon step.")

        self.step_count += 1
        directions: list[torch.Tensor | None] = []
        descent_score = 0.0
        for i, grad in enumerate(grads):
            if grad is None:
                directions.append(None)
                continue
            param = self.params[i]
            if torch.is_complex(param) or torch.is_complex(grad):
                raise RuntimeError("TorchControlledMuon does not support complex parameters or gradients.")
            if grad.is_sparse:
                raise RuntimeError("TorchControlledMuon does not support sparse gradients.")
            if param in self.momentum:
                direction, new_momentum = self._direction(grad, self.momentum[param])
                self.momentum[param] = new_momentum
            else:
                direction = self._adamw_direction(grad, self.adam_state[param])
            directions.append(direction)
            descent_score -= float(torch.sum(grad * direction).item())

        alpha_used = self.alpha
        loss_before_value = float(loss_before.detach().item())
        if descent_score <= 0.0:
            alpha_next = float(
                np.clip(self.alpha * self.non_descent_shrink, self.alpha_min, self.alpha_max)
            )
            alpha_update_factor = alpha_next / alpha_used
            self.alpha = alpha_next
            return ControlledMuonStep(
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
            )

        original_params = [param.detach().clone() for param in self.params]
        rho = float("nan")
        predicted_decrease = 0.0
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
            rho = actual_decrease / (predicted_decrease + self.ratio_eps)
            if (not self.reject_bad_steps) or (rho > self.rho_min):
                rho_control = self._update_rho_control(rho)
                alpha_next, alpha_update_factor, trust_region_expanded = self._next_alpha_after_trial(
                    trial_alpha,
                    rho_control,
                    backtracks,
                )
                self.alpha = alpha_next
                return ControlledMuonStep(
                    loss_before=loss_before_value,
                    loss_after=loss_after_value,
                    alpha=trial_alpha,
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
                )

        self._restore_params(original_params)
        rho_control = self._update_rho_control(rho) if np.isfinite(rho) else self.rho_ema
        alpha_next, alpha_update_factor, trust_region_expanded = self._next_alpha_after_trial(
            alpha_used,
            rho_control if rho_control is not None else self.rho_star - 1.0,
            self.max_backtracks + 1,
        )
        self.alpha = alpha_next
        return ControlledMuonStep(
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
                    trial = original
                    if self.weight_decay != 0.0:
                        trial = trial * (1.0 - alpha * self.weight_decay)
                    param.copy_(trial + alpha * direction)

    def _restore_params(self, originals: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for param, original in zip(self.params, originals):
                param.copy_(original)

    def _update_rho_control(self, rho: float) -> float:
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
    ) -> tuple[float, float, bool]:
        raw_factor = float(np.exp(self.kp * (rho_control - self.rho_star)))
        factor = float(np.clip(raw_factor, self.min_alpha_factor, self.max_alpha_factor))
        trust_region_expanded = (
            self.trust_region_expand
            and backtracks == 0
            and rho_control >= self.trust_region_rho_threshold
            and alpha_used <= self.trust_region_alpha_threshold
        )
        if trust_region_expanded:
            factor = max(factor, self.trust_region_expand_factor)
        alpha_next = float(np.clip(alpha_used * factor, self.alpha_min, self.alpha_max))
        alpha_update_factor = alpha_next / alpha_used
        return alpha_next, alpha_update_factor, trust_region_expanded
